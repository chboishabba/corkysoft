"""Call operations API routes."""

from __future__ import annotations

import base64
import binascii
from typing import Any, Dict, List, Optional, Sequence

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from analytics.db_connection import connection_scope
from analytics.operations_diary import (
    export_operations_diary_observer_events,
    list_observer_outbox_events,
)
from corkysoft.api_shared import (
    CALLS_WRITE_SCOPE,
    WORKER_TIME_WRITE_SCOPE,
    ApiAuthContext,
    _current_db_path,
    record_api_write_receipt,
    require_api_auth_context,
    require_internal_api_read_token,
)
from corkysoft.call_ops import (
    AMBIENT_SESSION_STATUSES,
    CALL_DIRECTIONS,
    CALL_EVENT_KINDS,
    CALL_LEG_KINDS,
    CALL_ROUTING_EVENT_TYPES,
    CALL_SOURCE_CHANNELS,
    CALL_STATUSES,
    CALL_TRANSCRIPT_STATUSES,
    EXTRACTED_ACTION_STATUSES,
    WORKER_TIME_CHANNELS,
    WORKER_TIME_EVENT_TYPES,
    add_call_note,
    add_extracted_action,
    create_ambient_session,
    create_call_event,
    create_call_leg,
    create_call_session,
    decide_extracted_action,
    decide_worker_time_capture_event,
    generate_fake_ambient_transcript_artifact,
    generate_fake_transcript_artifact,
    get_ambient_session,
    get_call_event,
    get_call_session,
    list_ambient_sessions,
    list_ambient_transcript_artifacts,
    list_call_events,
    list_call_legs,
    list_call_notes,
    list_call_routing_events,
    list_call_sessions,
    list_extracted_actions,
    list_state_egress_events,
    list_transcript_artifacts,
    list_worker_time_capture_events,
    log_call_routing_event,
    poll_transcript_artifact,
    record_ambient_transcript_artifact,
    record_transcript_artifact,
    record_worker_time_capture_event,
    resolve_call_links,
    submit_call_audio_for_transcription,
)
from corkysoft.whisperx_adapter import WhisperXAdapterError

router = APIRouter(dependencies=[Depends(require_internal_api_read_token)])
require_calls_write = require_api_auth_context((CALLS_WRITE_SCOPE,))
require_worker_time_write = require_api_auth_context((WORKER_TIME_WRITE_SCOPE,))

_ALLOWED_TRANSCRIPT_AUDIO_EXTENSIONS = {".m4a", ".mp3", ".mp4", ".ogg", ".wav"}
_MAX_TRANSCRIPT_AUDIO_BYTES = 25 * 1024 * 1024
_MAX_TRANSCRIPT_BASE64_CHARS = 36 * 1024 * 1024
_MAX_TRANSCRIPT_TEXT_CHARS = 100_000


def _submit_call_audio_for_transcription(conn, **kwargs: Any) -> dict[str, Any]:
    from corkysoft import api as api_module

    submitter = getattr(
        api_module,
        "submit_call_audio_for_transcription",
        submit_call_audio_for_transcription,
    )
    return submitter(conn, **kwargs)


def _poll_transcript_artifact(conn, *, artifact_id: int) -> dict[str, Any]:
    from corkysoft import api as api_module

    poller = getattr(api_module, "poll_transcript_artifact", poll_transcript_artifact)
    return poller(conn, artifact_id=artifact_id)


def _bound_actor(auth: ApiAuthContext, supplied: Optional[str] = None) -> str:
    if auth.legacy and supplied:
        return supplied
    return auth.actor


def _receipt(
    conn,
    *,
    auth: ApiAuthContext,
    action: str,
    resource_type: str,
    resource_id: object,
    request: Request,
) -> None:
    record_api_write_receipt(
        conn,
        auth=auth,
        action=action,
        resource_type=resource_type,
        resource_id=str(resource_id),
        request=request,
    )


class CallEventCreateRequest(BaseModel):
    eventKind: str = Field(
        ...,
        description=(
            "client_call, ops_call, manager_call, worker_call, clock_on_call, or clock_off_call"
        ),
    )
    direction: str = Field(..., description="inbound, outbound, or internal")
    status: str = Field(default="completed", description="Current call status")
    sourceChannel: str = Field(
        default="telephony",
        description="telephony, whatsapp, manual_note, or imported_recording",
    )
    title: Optional[str] = None
    callerPhone: Optional[str] = None
    calleePhone: Optional[str] = None
    quoteId: Optional[int] = None
    jobId: Optional[int] = None
    segmentId: Optional[int] = None
    workerId: Optional[int] = None
    operatorId: Optional[str] = None
    startedAt: Optional[str] = None
    endedAt: Optional[str] = None
    capturedAt: Optional[str] = None
    correlationId: Optional[str] = None


class CallEventResponse(BaseModel):
    id: int
    eventKind: str
    direction: str
    status: str
    sourceChannel: str
    title: Optional[str] = None
    callerPhone: Optional[str] = None
    callerPhoneNormalized: Optional[str] = None
    calleePhone: Optional[str] = None
    calleePhoneNormalized: Optional[str] = None
    clientId: Optional[int] = None
    clientName: Optional[str] = None
    quoteId: Optional[int] = None
    jobId: Optional[int] = None
    segmentId: Optional[int] = None
    workerId: Optional[int] = None
    workerName: Optional[str] = None
    jobClient: Optional[str] = None
    jobOrigin: Optional[str] = None
    jobDestination: Optional[str] = None
    operatorId: Optional[str] = None
    correlationId: Optional[str] = None
    startedAt: Optional[str] = None
    endedAt: Optional[str] = None
    capturedAt: Optional[str] = None
    processedAt: Optional[str] = None
    createdAt: str
    updatedAt: str
    latestTranscriptStatus: Optional[str] = None
    pendingActionCount: int = 0


class CallSessionCreateRequest(BaseModel):
    eventKind: str
    direction: str
    status: str = "completed"
    sourceChannel: str = "telephony"
    title: Optional[str] = None
    callerPhone: Optional[str] = None
    calleePhone: Optional[str] = None
    quoteId: Optional[int] = None
    jobId: Optional[int] = None
    segmentId: Optional[int] = None
    workerId: Optional[int] = None
    operatorId: Optional[str] = None
    startedAt: Optional[str] = None
    endedAt: Optional[str] = None
    capturedAt: Optional[str] = None
    correlationId: Optional[str] = None
    initialDestinationKind: Optional[str] = None
    initialDestinationLabel: Optional[str] = None


class CallSessionResponse(BaseModel):
    id: int
    rootCallEventId: Optional[int] = None
    eventKind: str
    direction: str
    status: str
    sourceChannel: str
    title: Optional[str] = None
    callerPhone: Optional[str] = None
    callerPhoneNormalized: Optional[str] = None
    calleePhone: Optional[str] = None
    calleePhoneNormalized: Optional[str] = None
    clientId: Optional[int] = None
    clientName: Optional[str] = None
    quoteId: Optional[int] = None
    jobId: Optional[int] = None
    segmentId: Optional[int] = None
    workerId: Optional[int] = None
    workerName: Optional[str] = None
    operatorId: Optional[str] = None
    correlationId: Optional[str] = None
    startedAt: Optional[str] = None
    endedAt: Optional[str] = None
    capturedAt: Optional[str] = None
    processedAt: Optional[str] = None
    createdAt: str
    updatedAt: str
    legCount: int = 0
    pendingActionCount: int = 0


class CallLegCreateRequest(BaseModel):
    legKind: str = "primary"
    direction: str = "inbound"
    status: str = "ringing"
    sourceChannel: str = "telephony"
    destinationKind: Optional[str] = None
    destinationLabel: Optional[str] = None
    operatorId: Optional[str] = None
    callerPhone: Optional[str] = None
    calleePhone: Optional[str] = None
    startedAt: Optional[str] = None
    answeredAt: Optional[str] = None
    endedAt: Optional[str] = None


class CallLegResponse(BaseModel):
    id: int
    callSessionId: int
    rootCallEventId: Optional[int] = None
    legKind: str
    direction: str
    status: str
    sourceChannel: str
    destinationKind: Optional[str] = None
    destinationLabel: Optional[str] = None
    operatorId: Optional[str] = None
    callerPhone: Optional[str] = None
    callerPhoneNormalized: Optional[str] = None
    calleePhone: Optional[str] = None
    calleePhoneNormalized: Optional[str] = None
    startedAt: Optional[str] = None
    answeredAt: Optional[str] = None
    endedAt: Optional[str] = None
    createdAt: str
    updatedAt: str
    latestTranscriptStatus: Optional[str] = None


class CallRoutingEventCreateRequest(BaseModel):
    eventType: str
    callLegId: Optional[int] = None
    fromDestination: Optional[str] = None
    toDestination: Optional[str] = None
    actor: Optional[str] = None
    detail: Optional[str] = None


class CallRoutingEventResponse(BaseModel):
    id: int
    callSessionId: int
    callLegId: Optional[int] = None
    eventType: str
    fromDestination: Optional[str] = None
    toDestination: Optional[str] = None
    actor: Optional[str] = None
    detail: Optional[str] = None
    createdAt: str


class CallNoteCreateRequest(BaseModel):
    author: Optional[str] = None
    noteText: str
    noteKind: str = "operator"
    authoritative: bool = True


class CallNoteResponse(BaseModel):
    id: int
    callEventId: Optional[int] = None
    ambientSessionId: Optional[int] = None
    author: Optional[str] = None
    noteKind: str
    noteText: str
    authoritative: bool
    createdAt: str


class ExtractedActionCreateRequest(BaseModel):
    actionText: str
    sourceEngine: Optional[str] = None
    transcriptArtifactId: Optional[int] = None
    spanStart: Optional[float] = None
    spanEnd: Optional[float] = None


class ExtractedActionDecisionRequest(BaseModel):
    status: str = Field(..., description="accepted or rejected")
    decidedBy: Optional[str] = None
    decisionNote: Optional[str] = None


class ExtractedActionResponse(BaseModel):
    id: int
    callEventId: Optional[int] = None
    ambientSessionId: Optional[int] = None
    transcriptArtifactId: Optional[int] = None
    sourceEngine: Optional[str] = None
    actionText: str
    spanStart: Optional[float] = None
    spanEnd: Optional[float] = None
    status: str
    decidedBy: Optional[str] = None
    decisionNote: Optional[str] = None
    createdAt: str
    decidedAt: Optional[str] = None


class CallLinkResolutionRequest(BaseModel):
    actor: Optional[str] = None
    ambientSessionId: Optional[int] = None
    clientId: Optional[int] = None
    quoteId: Optional[int] = None
    jobId: Optional[int] = None
    segmentId: Optional[int] = None
    workerId: Optional[int] = None
    resolutionNote: Optional[str] = None


class TranscriptArtifactCreateRequest(BaseModel):
    serviceKey: str = "ops"
    status: str = "completed"
    transcriptText: Optional[str] = Field(
        default=None,
        max_length=_MAX_TRANSCRIPT_TEXT_CHARS,
        description="Reviewed transcript text. Raw transcript content is observer-capture data.",
    )
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    isFinal: bool = True


class TranscriptUploadRequest(BaseModel):
    serviceKey: str = Field(default="ops", max_length=64)
    filename: str = Field(..., max_length=255)
    contentBase64: str = Field(..., max_length=_MAX_TRANSCRIPT_BASE64_CHARS)
    language: Optional[str] = Field(default=None, max_length=16)
    diarize: bool = True


def _decode_transcript_audio_upload(
    payload: TranscriptUploadRequest,
) -> tuple[bytes, str]:
    filename = (payload.filename or "call_audio.bin").strip()
    suffix = f".{filename.rsplit('.', 1)[-1].lower()}" if "." in filename else ""
    if suffix not in _ALLOWED_TRANSCRIPT_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="filename must use one of: m4a, mp3, mp4, ogg, wav",
        )
    content_base64 = (payload.contentBase64 or "").strip()
    if not content_base64:
        raise HTTPException(status_code=400, detail="contentBase64 is required")
    if len(content_base64) > _MAX_TRANSCRIPT_BASE64_CHARS:
        raise HTTPException(status_code=413, detail="audio upload is too large")
    try:
        file_bytes = base64.b64decode(content_base64.encode("utf-8"), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="contentBase64 must contain valid base64 data",
        ) from exc
    if not file_bytes:
        raise HTTPException(status_code=400, detail="audio upload cannot be empty")
    if len(file_bytes) > _MAX_TRANSCRIPT_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="audio upload is too large")
    return file_bytes, filename


class FakeTranscriptRequest(BaseModel):
    serviceKey: str = Field(default="ops", max_length=64)
    scenario: Optional[str] = Field(default=None, max_length=10_000)
    operatorGoal: Optional[str] = Field(default=None, max_length=10_000)


class TranscriptArtifactResponse(BaseModel):
    id: int
    callEventId: Optional[int] = None
    callSessionId: Optional[int] = None
    callLegId: Optional[int] = None
    serviceKey: str
    externalTaskId: Optional[str] = None
    status: str
    transcriptText: Optional[str] = None
    transcriptSegments: List[Dict[str, Any]] = Field(default_factory=list)
    diarization: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: Optional[float] = None
    isFinal: bool
    errorMessage: Optional[str] = None
    dataClassification: str
    authorityClass: str
    failureKind: Optional[str] = None
    createdAt: str
    updatedAt: str


class WorkerTimeCaptureCreateRequest(BaseModel):
    callEventId: Optional[int] = None
    callSessionId: Optional[int] = None
    callLegId: Optional[int] = None
    workerId: Optional[int] = None
    workerNameRaw: Optional[str] = None
    employeeCodeRaw: Optional[str] = None
    eventType: str
    channel: str
    effectiveTimestamp: Optional[str] = None
    callerPhone: Optional[str] = None
    jobId: Optional[int] = None
    segmentId: Optional[int] = None
    truckId: Optional[str] = None
    confidence: Optional[float] = None
    rawPayload: Dict[str, Any] = Field(default_factory=dict)


class WorkerTimeCaptureDecisionRequest(BaseModel):
    reviewStatus: str = Field(..., description="accepted or rejected")
    reviewer: Optional[str] = None
    reviewNote: Optional[str] = None
    workerId: Optional[int] = None
    jobId: Optional[int] = None
    segmentId: Optional[int] = None
    truckId: Optional[str] = None


class WorkerTimeCaptureResponse(BaseModel):
    id: int
    callEventId: Optional[int] = None
    callSessionId: Optional[int] = None
    callLegId: Optional[int] = None
    workerId: Optional[int] = None
    workerName: Optional[str] = None
    workerNameRaw: Optional[str] = None
    employeeCodeRaw: Optional[str] = None
    eventType: str
    channel: str
    effectiveTimestamp: Optional[str] = None
    capturedTimestamp: str
    callerPhone: Optional[str] = None
    jobId: Optional[int] = None
    segmentId: Optional[int] = None
    truckId: Optional[str] = None
    confidence: Optional[float] = None
    reviewStatus: str
    reviewer: Optional[str] = None
    reviewNote: Optional[str] = None
    rawPayload: Dict[str, Any] = Field(default_factory=dict)
    createdAt: str
    reviewedAt: Optional[str] = None


class StateEgressEventResponse(BaseModel):
    id: int
    surface: Optional[str] = None
    eventId: str
    sourceComponent: str
    sourceSystem: Optional[str] = None
    sourceEntityId: str
    eventType: str
    eventFamily: Optional[str] = None
    idempotencyKey: str
    correlationId: Optional[str] = None
    correlationKey: Optional[str] = None
    causationId: Optional[str] = None
    actorRef: Optional[str] = None
    authorityClass: str
    summary: Optional[str] = None
    status: Optional[str] = None
    objectRefs: Dict[str, Any] = Field(default_factory=dict)
    provenanceRefs: List[Dict[str, Any]] = Field(default_factory=list)
    evidenceRefs: List[Dict[str, Any]] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)
    payloadHash: str
    eventTime: Optional[str] = None
    occurredAt: str
    recordedAt: Optional[str] = None
    ingestedAt: str
    cursor: Optional[str] = None


class OperationsDiaryObserverExportRequest(BaseModel):
    anchorDate: str
    viewMode: str = "day"
    focusJobId: Optional[int] = None
    actorRef: Optional[str] = None
    includePlanningSnapshot: bool = True
    includeReconciliationExceptions: bool = True


class OperationsDiaryObserverExportResponse(BaseModel):
    anchorDate: str
    viewMode: str
    emittedCount: int
    byFamily: Dict[str, int] = Field(default_factory=dict)
    events: List[StateEgressEventResponse] = Field(default_factory=list)


class AmbientSessionCreateRequest(BaseModel):
    title: Optional[str] = None
    sourceLocation: Optional[str] = None
    sourceDevice: Optional[str] = None
    teamLabel: Optional[str] = None
    status: str = "active"
    clientId: Optional[int] = None
    quoteId: Optional[int] = None
    jobId: Optional[int] = None
    segmentId: Optional[int] = None
    workerId: Optional[int] = None
    operatorId: Optional[str] = None
    startedAt: Optional[str] = None
    endedAt: Optional[str] = None
    capturedAt: Optional[str] = None
    correlationId: Optional[str] = None


class AmbientSessionResponse(BaseModel):
    id: int
    title: Optional[str] = None
    sourceLocation: Optional[str] = None
    sourceDevice: Optional[str] = None
    teamLabel: Optional[str] = None
    status: str
    clientId: Optional[int] = None
    clientName: Optional[str] = None
    quoteId: Optional[int] = None
    jobId: Optional[int] = None
    segmentId: Optional[int] = None
    workerId: Optional[int] = None
    workerName: Optional[str] = None
    operatorId: Optional[str] = None
    correlationId: Optional[str] = None
    startedAt: Optional[str] = None
    endedAt: Optional[str] = None
    capturedAt: Optional[str] = None
    processedAt: Optional[str] = None
    createdAt: str
    updatedAt: str


class AmbientTranscriptArtifactResponse(BaseModel):
    id: int
    ambientSessionId: int
    serviceKey: str
    status: str
    transcriptText: Optional[str] = None
    transcriptSegments: List[Dict[str, Any]] = Field(default_factory=list)
    diarization: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: Optional[float] = None
    isFinal: bool
    errorMessage: Optional[str] = None
    dataClassification: str
    authorityClass: str
    failureKind: Optional[str] = None
    createdAt: str
    updatedAt: str


@router.get(
    "/calls/sessions",
    response_model=List[CallSessionResponse],
    summary="List call sessions with routing-aware context",
)
def get_call_sessions(
    status: Optional[str] = Query(default=None),
    event_kind: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> List[CallSessionResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_call_sessions(conn, status=status, event_kind=event_kind, limit=limit)
    return [CallSessionResponse(**row) for row in rows]


@router.post(
    "/calls/sessions",
    response_model=CallSessionResponse,
    summary="Create a routed call session with an initial leg",
)
def post_call_session(
    request: Request,
    payload: CallSessionCreateRequest,
    auth: ApiAuthContext = Depends(require_calls_write),
) -> CallSessionResponse:
    if payload.eventKind not in CALL_EVENT_KINDS:
        raise HTTPException(status_code=400, detail="Unsupported call event kind")
    if payload.direction not in CALL_DIRECTIONS:
        raise HTTPException(status_code=400, detail="Unsupported call direction")
    if payload.status not in CALL_STATUSES:
        raise HTTPException(status_code=400, detail="Unsupported call status")
    if payload.sourceChannel not in CALL_SOURCE_CHANNELS:
        raise HTTPException(status_code=400, detail="Unsupported call source channel")
    with connection_scope(_current_db_path()) as conn:
        row = create_call_session(
            conn,
            event_kind=payload.eventKind,
            direction=payload.direction,
            status=payload.status,
            source_channel=payload.sourceChannel,
            title=payload.title,
            caller_phone=payload.callerPhone,
            callee_phone=payload.calleePhone,
            quote_id=payload.quoteId,
            job_id=payload.jobId,
            segment_id=payload.segmentId,
            worker_id=payload.workerId,
            operator_id=_bound_actor(auth, payload.operatorId),
            started_at=payload.startedAt,
            ended_at=payload.endedAt,
            captured_at=payload.capturedAt,
            correlation_id=payload.correlationId,
            initial_destination_kind=payload.initialDestinationKind,
            initial_destination_label=payload.initialDestinationLabel,
        )
        _receipt(
            conn,
            auth=auth,
            action="calls.session.create",
            resource_type="call_session",
            resource_id=row["id"],
            request=request,
        )
    return CallSessionResponse(**row)


@router.get(
    "/calls/sessions/{call_session_id}",
    response_model=CallSessionResponse,
    summary="Get a single call session",
)
def get_call_session_by_id(call_session_id: int = Path(...)) -> CallSessionResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = get_call_session(conn, call_session_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return CallSessionResponse(**row)


@router.get(
    "/calls/sessions/{call_session_id}/legs",
    response_model=List[CallLegResponse],
    summary="List call legs for a session",
)
def get_call_session_legs(call_session_id: int = Path(...)) -> List[CallLegResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_call_legs(conn, call_session_id=call_session_id)
    return [CallLegResponse(**row) for row in rows]


@router.post(
    "/calls/sessions/{call_session_id}/legs",
    response_model=CallLegResponse,
    summary="Add a routed or consult leg to a call session",
)
def post_call_session_leg(
    request: Request,
    call_session_id: int = Path(...),
    payload: CallLegCreateRequest = ...,
    auth: ApiAuthContext = Depends(require_calls_write),
) -> CallLegResponse:
    if payload.legKind not in CALL_LEG_KINDS:
        raise HTTPException(status_code=400, detail="Unsupported call leg kind")
    if payload.direction not in CALL_DIRECTIONS:
        raise HTTPException(status_code=400, detail="Unsupported call direction")
    if payload.status not in CALL_STATUSES:
        raise HTTPException(status_code=400, detail="Unsupported call status")
    if payload.sourceChannel not in CALL_SOURCE_CHANNELS:
        raise HTTPException(status_code=400, detail="Unsupported call source channel")
    with connection_scope(_current_db_path()) as conn:
        try:
            row = create_call_leg(
                conn,
                call_session_id=call_session_id,
                leg_kind=payload.legKind,
                direction=payload.direction,
                status=payload.status,
                source_channel=payload.sourceChannel,
                destination_kind=payload.destinationKind,
                destination_label=payload.destinationLabel,
                operator_id=_bound_actor(auth, payload.operatorId),
                caller_phone=payload.callerPhone,
                callee_phone=payload.calleePhone,
                started_at=payload.startedAt,
                answered_at=payload.answeredAt,
                ended_at=payload.endedAt,
            )
            _receipt(
                conn,
                auth=auth,
                action="calls.leg.create",
                resource_type="call_leg",
                resource_id=row["id"],
                request=request,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CallLegResponse(**row)


@router.get(
    "/calls/sessions/{call_session_id}/routing-events",
    response_model=List[CallRoutingEventResponse],
    summary="List routing events for a call session",
)
def get_call_session_routing_events(
    call_session_id: int = Path(...),
) -> List[CallRoutingEventResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_call_routing_events(conn, call_session_id=call_session_id)
    return [CallRoutingEventResponse(**row) for row in rows]


@router.post(
    "/calls/sessions/{call_session_id}/routing-events",
    response_model=CallRoutingEventResponse,
    summary="Record a routing or transfer event for a call session",
)
def post_call_session_routing_event(
    request: Request,
    call_session_id: int = Path(...),
    payload: CallRoutingEventCreateRequest = ...,
    auth: ApiAuthContext = Depends(require_calls_write),
) -> CallRoutingEventResponse:
    if payload.eventType not in CALL_ROUTING_EVENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported routing event type")
    with connection_scope(_current_db_path()) as conn:
        try:
            row = log_call_routing_event(
                conn,
                call_session_id=call_session_id,
                event_type=payload.eventType,
                call_leg_id=payload.callLegId,
                from_destination=payload.fromDestination,
                to_destination=payload.toDestination,
                actor=_bound_actor(auth, payload.actor),
                detail=payload.detail,
            )
            _receipt(
                conn,
                auth=auth,
                action="calls.routing_event.create",
                resource_type="call_routing_event",
                resource_id=row["id"],
                request=request,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CallRoutingEventResponse(**row)


@router.get(
    "/calls/ambient-sessions",
    response_model=List[AmbientSessionResponse],
    summary="List ambient office transcript sessions",
)
def get_ambient_call_sessions(
    limit: int = Query(default=100, ge=1, le=500),
) -> List[AmbientSessionResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_ambient_sessions(conn, limit=limit)
    return [AmbientSessionResponse(**row) for row in rows]


@router.post(
    "/calls/ambient-sessions",
    response_model=AmbientSessionResponse,
    summary="Create an ambient office transcript session",
)
def post_ambient_call_session(
    request: Request,
    payload: AmbientSessionCreateRequest,
    auth: ApiAuthContext = Depends(require_calls_write),
) -> AmbientSessionResponse:
    if payload.status not in AMBIENT_SESSION_STATUSES:
        raise HTTPException(status_code=400, detail="Unsupported ambient session status")
    with connection_scope(_current_db_path()) as conn:
        row = create_ambient_session(
            conn,
            title=payload.title,
            source_location=payload.sourceLocation,
            source_device=payload.sourceDevice,
            team_label=payload.teamLabel,
            status=payload.status,
            client_id=payload.clientId,
            quote_id=payload.quoteId,
            job_id=payload.jobId,
            segment_id=payload.segmentId,
            worker_id=payload.workerId,
            operator_id=_bound_actor(auth, payload.operatorId),
            started_at=payload.startedAt,
            ended_at=payload.endedAt,
            captured_at=payload.capturedAt,
            correlation_id=payload.correlationId,
        )
        _receipt(
            conn,
            auth=auth,
            action="calls.ambient_session.create",
            resource_type="ambient_session",
            resource_id=row["id"],
            request=request,
        )
    return AmbientSessionResponse(**row)


@router.get(
    "/calls/ambient-sessions/{ambient_session_id}",
    response_model=AmbientSessionResponse,
    summary="Get a single ambient office transcript session",
)
def get_ambient_call_session_by_id(
    ambient_session_id: int = Path(...),
) -> AmbientSessionResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = get_ambient_session(conn, ambient_session_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AmbientSessionResponse(**row)


@router.get(
    "/calls/ambient-sessions/{ambient_session_id}/transcripts",
    response_model=List[AmbientTranscriptArtifactResponse],
    summary="List transcript artifacts for an ambient office session",
)
def get_ambient_call_session_transcripts(
    ambient_session_id: int = Path(...),
) -> List[AmbientTranscriptArtifactResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_ambient_transcript_artifacts(conn, ambient_session_id=ambient_session_id)
    return [AmbientTranscriptArtifactResponse(**row) for row in rows]


@router.post(
    "/calls/ambient-sessions/{ambient_session_id}/transcripts/manual",
    response_model=AmbientTranscriptArtifactResponse,
    summary="Create a manual transcript artifact for an ambient session",
)
def post_ambient_call_session_transcript(
    request: Request,
    ambient_session_id: int = Path(...),
    payload: TranscriptArtifactCreateRequest = ...,
    auth: ApiAuthContext = Depends(require_calls_write),
) -> AmbientTranscriptArtifactResponse:
    with connection_scope(_current_db_path()) as conn:
        row = record_ambient_transcript_artifact(
            conn,
            ambient_session_id=ambient_session_id,
            service_key=payload.serviceKey,
            status=payload.status,
            transcript_text=payload.transcriptText,
            confidence=payload.confidence,
            is_final=payload.isFinal,
        )
        _receipt(
            conn,
            auth=auth,
            action="calls.ambient_transcript.create",
            resource_type="ambient_transcript_artifact",
            resource_id=row["id"],
            request=request,
        )
    return AmbientTranscriptArtifactResponse(**row)


@router.post(
    "/calls/ambient-sessions/{ambient_session_id}/transcripts/fake",
    response_model=AmbientTranscriptArtifactResponse,
    summary="Generate a fake transcript artifact for an ambient office session",
)
def post_fake_ambient_call_session_transcript(
    request: Request,
    ambient_session_id: int = Path(...),
    payload: FakeTranscriptRequest = ...,
    auth: ApiAuthContext = Depends(require_calls_write),
) -> AmbientTranscriptArtifactResponse:
    with connection_scope(_current_db_path()) as conn:
        row = generate_fake_ambient_transcript_artifact(
            conn,
            ambient_session_id=ambient_session_id,
            scenario=payload.scenario,
            operator_goal=payload.operatorGoal,
            service_key=payload.serviceKey,
        )
        _receipt(
            conn,
            auth=auth,
            action="calls.ambient_transcript.fake",
            resource_type="ambient_transcript_artifact",
            resource_id=row["id"],
            request=request,
        )
    return AmbientTranscriptArtifactResponse(**row)


@router.get(
    "/calls/events",
    response_model=List[CallEventResponse],
    summary="List operational call events",
)
def get_call_events(
    status: Optional[str] = Query(default=None, description="Optional call status filter"),
    event_kind: Optional[str] = Query(default=None, description="Optional call-kind filter"),
    unresolved_only: bool = Query(default=False, description="Only list unresolved call events"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum rows to return"),
) -> List[CallEventResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_call_events(
            conn,
            limit=limit,
            status=status,
            event_kind=event_kind,
            unresolved_only=unresolved_only,
        )
    return [CallEventResponse(**row) for row in rows]


@router.post(
    "/calls/events",
    response_model=CallEventResponse,
    summary="Create a call event and auto-link by phone where possible",
)
def post_call_event(
    request: Request,
    payload: CallEventCreateRequest,
    auth: ApiAuthContext = Depends(require_calls_write),
) -> CallEventResponse:
    if payload.eventKind not in CALL_EVENT_KINDS:
        raise HTTPException(status_code=400, detail="Unsupported call event kind")
    if payload.direction not in CALL_DIRECTIONS:
        raise HTTPException(status_code=400, detail="Unsupported call direction")
    if payload.status not in CALL_STATUSES:
        raise HTTPException(status_code=400, detail="Unsupported call status")
    if payload.sourceChannel not in CALL_SOURCE_CHANNELS:
        raise HTTPException(status_code=400, detail="Unsupported call source channel")
    with connection_scope(_current_db_path()) as conn:
        row = create_call_event(
            conn,
            event_kind=payload.eventKind,
            direction=payload.direction,
            status=payload.status,
            source_channel=payload.sourceChannel,
            title=payload.title,
            caller_phone=payload.callerPhone,
            callee_phone=payload.calleePhone,
            quote_id=payload.quoteId,
            job_id=payload.jobId,
            segment_id=payload.segmentId,
            worker_id=payload.workerId,
            operator_id=_bound_actor(auth, payload.operatorId),
            started_at=payload.startedAt,
            ended_at=payload.endedAt,
            captured_at=payload.capturedAt,
            correlation_id=payload.correlationId,
        )
        _receipt(
            conn,
            auth=auth,
            action="calls.event.create",
            resource_type="call_event",
            resource_id=row["id"],
            request=request,
        )
    return CallEventResponse(**row)


@router.get(
    "/calls/events/{call_event_id}",
    response_model=CallEventResponse,
    summary="Get a single operational call event",
)
def get_call_event_by_id(
    call_event_id: int = Path(..., description="Call event identifier"),
) -> CallEventResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = get_call_event(conn, call_event_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return CallEventResponse(**row)


@router.get(
    "/calls/events/{call_event_id}/notes",
    response_model=List[CallNoteResponse],
    summary="List notes attached to a call event",
)
def get_call_notes_for_event(
    call_event_id: int = Path(..., description="Call event identifier"),
) -> List[CallNoteResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_call_notes(conn, call_event_id=call_event_id)
    return [CallNoteResponse(**row) for row in rows]


@router.post(
    "/calls/events/{call_event_id}/notes",
    response_model=CallNoteResponse,
    summary="Add an authoritative or advisory note to a call event",
)
def post_call_note(
    request: Request,
    call_event_id: int = Path(..., description="Call event identifier"),
    payload: CallNoteCreateRequest = ...,
    auth: ApiAuthContext = Depends(require_calls_write),
) -> CallNoteResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = add_call_note(
                conn,
                call_event_id=call_event_id,
                author=_bound_actor(auth, payload.author),
                note_text=payload.noteText,
                note_kind=payload.noteKind,
                authoritative=payload.authoritative,
            )
            _receipt(
                conn,
                auth=auth,
                action="calls.note.create",
                resource_type="call_note",
                resource_id=row["id"],
                request=request,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CallNoteResponse(**row)


@router.get(
    "/calls/events/{call_event_id}/extracted-actions",
    response_model=List[ExtractedActionResponse],
    summary="List extracted action candidates for a call event",
)
def get_call_extracted_actions(
    call_event_id: int = Path(..., description="Call event identifier"),
) -> List[ExtractedActionResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_extracted_actions(conn, call_event_id=call_event_id)
    return [ExtractedActionResponse(**row) for row in rows]


@router.post(
    "/calls/events/{call_event_id}/extracted-actions",
    response_model=ExtractedActionResponse,
    summary="Add an extracted action candidate for review",
)
def post_call_extracted_action(
    request: Request,
    call_event_id: int = Path(..., description="Call event identifier"),
    payload: ExtractedActionCreateRequest = ...,
    auth: ApiAuthContext = Depends(require_calls_write),
) -> ExtractedActionResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = add_extracted_action(
                conn,
                call_event_id=call_event_id,
                action_text=payload.actionText,
                source_engine=payload.sourceEngine,
                transcript_artifact_id=payload.transcriptArtifactId,
                span_start=payload.spanStart,
                span_end=payload.spanEnd,
            )
            _receipt(
                conn,
                auth=auth,
                action="calls.extracted_action.create",
                resource_type="extracted_action",
                resource_id=row["id"],
                request=request,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExtractedActionResponse(**row)


@router.post(
    "/calls/extracted-actions/{action_id}/decision",
    response_model=ExtractedActionResponse,
    summary="Accept or reject an extracted action candidate",
)
def post_call_extracted_action_decision(
    request: Request,
    action_id: int = Path(..., description="Extracted action identifier"),
    payload: ExtractedActionDecisionRequest = ...,
    auth: ApiAuthContext = Depends(require_calls_write),
) -> ExtractedActionResponse:
    if payload.status not in EXTRACTED_ACTION_STATUSES or payload.status == "pending":
        raise HTTPException(
            status_code=400,
            detail="Decision status must be accepted or rejected",
        )
    with connection_scope(_current_db_path()) as conn:
        try:
            row = decide_extracted_action(
                conn,
                action_id=action_id,
                status=payload.status,
                decided_by=_bound_actor(auth, payload.decidedBy),
                decision_note=payload.decisionNote,
            )
            _receipt(
                conn,
                auth=auth,
                action="calls.extracted_action.decide",
                resource_type="extracted_action",
                resource_id=action_id,
                request=request,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExtractedActionResponse(**row)


@router.post(
    "/calls/events/{call_event_id}/resolve",
    response_model=CallEventResponse,
    summary="Resolve a call event to client/job/segment/worker context",
)
def post_call_link_resolution(
    request: Request,
    call_event_id: int = Path(..., description="Call event identifier"),
    payload: CallLinkResolutionRequest = ...,
    auth: ApiAuthContext = Depends(require_calls_write),
) -> CallEventResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = resolve_call_links(
                conn,
                call_event_id=call_event_id,
                actor=_bound_actor(auth, payload.actor),
                client_id=payload.clientId,
                quote_id=payload.quoteId,
                job_id=payload.jobId,
                segment_id=payload.segmentId,
                worker_id=payload.workerId,
                resolution_note=payload.resolutionNote,
            )
            _receipt(
                conn,
                auth=auth,
                action="calls.event.resolve",
                resource_type="call_event",
                resource_id=call_event_id,
                request=request,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CallEventResponse(**row)


@router.get(
    "/calls/legs/{call_leg_id}/transcripts",
    response_model=List[TranscriptArtifactResponse],
    summary="List transcript artifacts for a call leg",
)
def get_call_leg_transcripts(
    call_leg_id: int = Path(..., description="Call leg identifier"),
) -> List[TranscriptArtifactResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_transcript_artifacts(conn, call_leg_id=call_leg_id)
    return [TranscriptArtifactResponse(**row) for row in rows]


@router.post(
    "/calls/legs/{call_leg_id}/transcripts/manual",
    response_model=TranscriptArtifactResponse,
    summary="Create a manual transcript artifact for a call leg",
)
def post_manual_call_leg_transcript(
    request: Request,
    call_leg_id: int = Path(..., description="Call leg identifier"),
    payload: TranscriptArtifactCreateRequest = ...,
    auth: ApiAuthContext = Depends(require_calls_write),
) -> TranscriptArtifactResponse:
    with connection_scope(_current_db_path()) as conn:
        row = record_transcript_artifact(
            conn,
            call_leg_id=call_leg_id,
            service_key=payload.serviceKey,
            status=payload.status,
            transcript_text=payload.transcriptText,
            confidence=payload.confidence,
            is_final=payload.isFinal,
        )
        _receipt(
            conn,
            auth=auth,
            action="calls.transcript.create",
            resource_type="transcript_artifact",
            resource_id=row["id"],
            request=request,
        )
    return TranscriptArtifactResponse(**row)


@router.post(
    "/calls/legs/{call_leg_id}/transcripts/fake",
    response_model=TranscriptArtifactResponse,
    summary="Generate a fake transcript artifact for a call leg",
)
def post_fake_call_leg_transcript(
    request: Request,
    call_leg_id: int = Path(..., description="Call leg identifier"),
    payload: FakeTranscriptRequest = ...,
    auth: ApiAuthContext = Depends(require_calls_write),
) -> TranscriptArtifactResponse:
    with connection_scope(_current_db_path()) as conn:
        row = generate_fake_transcript_artifact(
            conn,
            call_leg_id=call_leg_id,
            scenario=payload.scenario,
            operator_goal=payload.operatorGoal,
            service_key=payload.serviceKey,
        )
        _receipt(
            conn,
            auth=auth,
            action="calls.transcript.fake",
            resource_type="transcript_artifact",
            resource_id=row["id"],
            request=request,
        )
    return TranscriptArtifactResponse(**row)


@router.post(
    "/calls/legs/{call_leg_id}/transcripts/upload",
    response_model=TranscriptArtifactResponse,
    summary="Submit audio to WhisperX and create a queued transcript artifact for a call leg",
)
def post_call_leg_transcript_upload(
    request: Request,
    call_leg_id: int = Path(..., description="Call leg identifier"),
    payload: TranscriptUploadRequest = ...,
    auth: ApiAuthContext = Depends(require_calls_write),
) -> TranscriptArtifactResponse:
    file_bytes, filename = _decode_transcript_audio_upload(payload)
    with connection_scope(_current_db_path()) as conn:
        try:
            row = _submit_call_audio_for_transcription(
                conn,
                call_leg_id=call_leg_id,
                service_key=payload.serviceKey,
                file_bytes=file_bytes,
                filename=filename,
                language=payload.language,
                diarize=payload.diarize,
            )
            _receipt(
                conn,
                auth=auth,
                action="calls.transcript.upload",
                resource_type="transcript_artifact",
                resource_id=row["id"],
                request=request,
            )
        except WhisperXAdapterError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TranscriptArtifactResponse(**row)


@router.get(
    "/calls/events/{call_event_id}/transcripts",
    response_model=List[TranscriptArtifactResponse],
    summary="List transcript artifacts for a call event",
)
def get_call_transcripts(
    call_event_id: int = Path(..., description="Call event identifier"),
) -> List[TranscriptArtifactResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_transcript_artifacts(conn, call_event_id=call_event_id)
    return [TranscriptArtifactResponse(**row) for row in rows]


@router.post(
    "/calls/events/{call_event_id}/transcripts/manual",
    response_model=TranscriptArtifactResponse,
    summary="Create a manual transcript artifact",
)
def post_manual_call_transcript(
    request: Request,
    call_event_id: int = Path(..., description="Call event identifier"),
    payload: TranscriptArtifactCreateRequest = ...,
    auth: ApiAuthContext = Depends(require_calls_write),
) -> TranscriptArtifactResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = record_transcript_artifact(
                conn,
                call_event_id=call_event_id,
                service_key=payload.serviceKey,
                status=payload.status,
                transcript_text=payload.transcriptText,
                confidence=payload.confidence,
                is_final=payload.isFinal,
            )
            _receipt(
                conn,
                auth=auth,
                action="calls.transcript.create",
                resource_type="transcript_artifact",
                resource_id=row["id"],
                request=request,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TranscriptArtifactResponse(**row)


@router.post(
    "/calls/events/{call_event_id}/transcripts/fake",
    response_model=TranscriptArtifactResponse,
    summary="Generate a fake transcript artifact for workflow testing",
)
def post_fake_call_transcript(
    request: Request,
    call_event_id: int = Path(..., description="Call event identifier"),
    payload: FakeTranscriptRequest = ...,
    auth: ApiAuthContext = Depends(require_calls_write),
) -> TranscriptArtifactResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = generate_fake_transcript_artifact(
                conn,
                call_event_id=call_event_id,
                scenario=payload.scenario,
                operator_goal=payload.operatorGoal,
                service_key=payload.serviceKey,
            )
            _receipt(
                conn,
                auth=auth,
                action="calls.transcript.fake",
                resource_type="transcript_artifact",
                resource_id=row["id"],
                request=request,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TranscriptArtifactResponse(**row)


@router.post(
    "/calls/events/{call_event_id}/transcripts/upload",
    response_model=TranscriptArtifactResponse,
    summary="Submit audio to WhisperX and create a queued transcript artifact",
)
def post_call_transcript_upload(
    request: Request,
    call_event_id: int = Path(..., description="Call event identifier"),
    payload: TranscriptUploadRequest = ...,
    auth: ApiAuthContext = Depends(require_calls_write),
) -> TranscriptArtifactResponse:
    file_bytes, filename = _decode_transcript_audio_upload(payload)
    with connection_scope(_current_db_path()) as conn:
        try:
            row = _submit_call_audio_for_transcription(
                conn,
                call_event_id=call_event_id,
                service_key=payload.serviceKey,
                file_bytes=file_bytes,
                filename=filename,
                language=payload.language,
                diarize=payload.diarize,
            )
            _receipt(
                conn,
                auth=auth,
                action="calls.transcript.upload",
                resource_type="transcript_artifact",
                resource_id=row["id"],
                request=request,
            )
        except WhisperXAdapterError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TranscriptArtifactResponse(**row)


@router.post(
    "/calls/transcripts/{artifact_id}/poll",
    response_model=TranscriptArtifactResponse,
    summary="Poll WhisperX for transcript task completion",
)
def post_call_transcript_poll(
    request: Request,
    artifact_id: int = Path(..., description="Transcript artifact identifier"),
    auth: ApiAuthContext = Depends(require_calls_write),
) -> TranscriptArtifactResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = _poll_transcript_artifact(conn, artifact_id=artifact_id)
            _receipt(
                conn,
                auth=auth,
                action="calls.transcript.poll",
                resource_type="transcript_artifact",
                resource_id=artifact_id,
                request=request,
            )
        except WhisperXAdapterError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TranscriptArtifactResponse(**row)


@router.get(
    "/worker-time/events",
    response_model=List[WorkerTimeCaptureResponse],
    summary="List worker time-capture events and review state",
)
def get_worker_time_events(
    review_status: Optional[str] = Query(default=None, description="Optional review-status filter"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum rows to return"),
) -> List[WorkerTimeCaptureResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_worker_time_capture_events(
            conn, review_status=review_status, limit=limit
        )
    return [WorkerTimeCaptureResponse(**row) for row in rows]


@router.post(
    "/worker-time/events",
    response_model=WorkerTimeCaptureResponse,
    summary="Record a worker time-capture event from app, WhatsApp, or voice call",
)
def post_worker_time_event(
    request: Request,
    payload: WorkerTimeCaptureCreateRequest,
    auth: ApiAuthContext = Depends(require_worker_time_write),
) -> WorkerTimeCaptureResponse:
    if payload.eventType not in WORKER_TIME_EVENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported worker time event type")
    if payload.channel not in WORKER_TIME_CHANNELS:
        raise HTTPException(status_code=400, detail="Unsupported worker time channel")
    with connection_scope(_current_db_path()) as conn:
        try:
            row = record_worker_time_capture_event(
                conn,
                call_event_id=payload.callEventId,
                call_session_id=payload.callSessionId,
                call_leg_id=payload.callLegId,
                worker_id=payload.workerId,
                worker_name_raw=payload.workerNameRaw,
                employee_code_raw=payload.employeeCodeRaw,
                event_type=payload.eventType,
                channel=payload.channel,
                effective_timestamp=payload.effectiveTimestamp,
                caller_phone=payload.callerPhone,
                job_id=payload.jobId,
                segment_id=payload.segmentId,
                truck_id=payload.truckId,
                confidence=payload.confidence,
                raw_payload=payload.rawPayload,
            )
            _receipt(
                conn,
                auth=auth,
                action="worker_time.event.create",
                resource_type="worker_time_event",
                resource_id=row["id"],
                request=request,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WorkerTimeCaptureResponse(**row)


@router.post(
    "/worker-time/events/{event_id}/decision",
    response_model=WorkerTimeCaptureResponse,
    summary="Accept or reject a worker time-capture event after review",
)
def post_worker_time_event_decision(
    request: Request,
    event_id: int = Path(..., description="Worker time event identifier"),
    payload: WorkerTimeCaptureDecisionRequest = ...,
    auth: ApiAuthContext = Depends(require_worker_time_write),
) -> WorkerTimeCaptureResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = decide_worker_time_capture_event(
                conn,
                event_id=event_id,
                review_status=payload.reviewStatus,
                reviewer=_bound_actor(auth, payload.reviewer),
                review_note=payload.reviewNote,
                worker_id=payload.workerId,
                job_id=payload.jobId,
                segment_id=payload.segmentId,
                truck_id=payload.truckId,
            )
            _receipt(
                conn,
                auth=auth,
                action="worker_time.event.decide",
                resource_type="worker_time_event",
                resource_id=event_id,
                request=request,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WorkerTimeCaptureResponse(**row)


@router.get(
    "/state-egress/events",
    response_model=List[StateEgressEventResponse],
    summary="List append-only downstream state events prepared for StatiBaker-like consumers",
)
def get_state_egress_events(
    limit: int = Query(default=100, ge=1, le=500, description="Maximum rows to return"),
    surface: str = Query(default="all", description="all, call_ops, or observer"),
    event_family: Optional[str] = Query(
        default=None,
        alias="eventFamily",
        description="Optional observer event family",
    ),
    job_id: Optional[int] = Query(
        default=None,
        alias="jobId",
        description="Optional job identifier filter",
    ),
    start_time: Optional[str] = Query(
        default=None,
        alias="startTime",
        description="Earliest event time to return",
    ),
    end_time: Optional[str] = Query(
        default=None,
        alias="endTime",
        description="Latest event time to return",
    ),
    cursor: Optional[str] = Query(
        default=None,
        description="Opaque cursor returned by a previous listing call",
    ),
) -> List[StateEgressEventResponse]:
    with connection_scope(_current_db_path()) as conn:
        payload: list[dict[str, Any]] = []
        normalized_surface = str(surface or "all").strip().lower()
        include_call_ops = normalized_surface in {"all", "call_ops"}
        include_observer = normalized_surface in {"all", "observer"}
        if include_call_ops:
            for row in list_state_egress_events(conn, limit=max(limit * 3, limit)):
                row = {
                    **row,
                    "surface": "call_ops",
                    "sourceSystem": row["sourceComponent"],
                    "eventFamily": row["eventType"],
                    "correlationKey": row.get("correlationId"),
                    "actorRef": None,
                    "summary": None,
                    "status": None,
                    "objectRefs": {},
                    "provenanceRefs": [],
                    "evidenceRefs": [],
                    "eventTime": row["occurredAt"],
                    "recordedAt": row["ingestedAt"],
                    "cursor": f"{row['occurredAt']}|call_ops|{row['eventId']}",
                }
                payload.append(row)
        if include_observer:
            payload.extend(
                list_observer_outbox_events(
                    conn,
                    limit=max(limit * 3, limit),
                    event_family=event_family,
                    job_id=job_id,
                    start_time=start_time,
                    end_time=end_time,
                )
            )
        if job_id is not None and include_call_ops:
            target = int(job_id)
            payload = [
                row
                for row in payload
                if row.get("surface") != "call_ops"
                or int(row["payload"].get("jobId") or -1) == target
            ]
        if start_time is not None:
            payload = [
                row
                for row in payload
                if str(row.get("eventTime") or row["occurredAt"]) >= start_time
            ]
        if end_time is not None:
            payload = [
                row
                for row in payload
                if str(row.get("eventTime") or row["occurredAt"]) <= end_time
            ]
        payload.sort(
            key=lambda row: (
                str(row.get("eventTime") or row["occurredAt"]),
                str(row.get("surface") or ""),
                str(row["eventId"]),
            ),
            reverse=True,
        )
        if cursor:
            payload = [row for row in payload if str(row.get("cursor")) < cursor]
        rows = payload[:limit]
    return [StateEgressEventResponse(**row) for row in rows]


@router.post(
    "/state-egress/operations-diary-export",
    response_model=OperationsDiaryObserverExportResponse,
    summary="Emit append-only observer envelopes for operations-diary planning snapshots and reconciliation exceptions",
)
def post_operations_diary_export(
    request: Request,
    payload: OperationsDiaryObserverExportRequest,
    auth: ApiAuthContext = Depends(require_calls_write),
) -> OperationsDiaryObserverExportResponse:
    with connection_scope(_current_db_path()) as conn:
        row = export_operations_diary_observer_events(
            conn,
            anchor_date=payload.anchorDate,
            view_mode=payload.viewMode,
            focus_job_id=payload.focusJobId,
            actor_ref=_bound_actor(auth, payload.actorRef),
            include_planning_snapshot=payload.includePlanningSnapshot,
            include_reconciliation_exceptions=payload.includeReconciliationExceptions,
        )
        _receipt(
            conn,
            auth=auth,
            action="state_egress.operations_diary_export",
            resource_type="operations_diary_export",
            resource_id=f"{row['anchorDate']}:{row['viewMode']}",
            request=request,
        )
    return OperationsDiaryObserverExportResponse(
        anchorDate=row["anchorDate"],
        viewMode=row["viewMode"],
        emittedCount=row["emittedCount"],
        byFamily=row["byFamily"],
        events=[StateEgressEventResponse(**item) for item in row["events"]],
    )
