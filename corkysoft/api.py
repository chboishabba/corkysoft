"""Minimal REST API surface for Corkysoft integrations."""

from __future__ import annotations

import os
import sqlite3
import base64
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query
from pydantic import BaseModel, Field

from analytics.db import fetch_driver_shifts
from analytics.db.inventory import (
    allocate_inventory_to_segment,
    list_segment_inventory_coordination,
)
from analytics.db_connection import connection_scope
from analytics.kent_ams_import import import_kent_ams_records
from analytics.moveware_import import import_moveware_records
from analytics.operations_assignment import (
    approve_operations_cutover_promotion,
    apply_operations_cutover_recommendation,
    assign_segment_resources,
    assign_worker_compliance,
    assign_worker_role,
    ensure_segment,
    ensure_worker_compliance,
    ensure_worker_role,
    get_operations_policy,
    list_job_operations_board,
    list_operations_cutover_events,
    list_operations_cutover_rollout,
    list_operations_cutover_workflows,
    list_operational_readiness_items,
    list_operational_conflicts,
    list_segment_readiness,
    reject_operations_cutover_promotion,
    record_operations_cutover_event,
    request_operations_cutover_promotion,
    upsert_operations_cutover_workflow,
    update_operations_policy,
)
from analytics.operations_workbook import sync_operations_workbook
from analytics.operations_diary import (
    export_operations_diary_observer_events,
    list_observer_outbox_events,
)
from corkysoft.call_ops import (
    AMBIENT_SESSION_STATUSES,
    CALL_DIRECTIONS,
    CALL_EVENT_KINDS,
    CALL_LEG_KINDS,
    CALL_ROUTING_EVENT_TYPES,
    CALL_SOURCE_CHANNELS,
    CALL_STATUSES,
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
from corkysoft.api_kent import router as kent_router
from corkysoft.api_labor import router as labor_router


def _current_db_path() -> str:
    """Return the SQLite database path configured for the API."""

    return (
        os.environ.get("CORKYSOFT_DB")
        or os.environ.get("ROUTES_DB")
        or "routes.db"
    )


def _required_internal_api_token() -> str:
    token = os.environ.get("CORKYSOFT_API_TOKEN")
    if not token:
        raise HTTPException(
            status_code=503,
            detail="CORKYSOFT_API_TOKEN is not configured for mutating API routes",
        )
    return token


def require_internal_api_token(
    x_corkysoft_api_key: Optional[str] = Header(
        default=None, alias="X-Corkysoft-Api-Key"
    ),
) -> None:
    expected = _required_internal_api_token()
    if x_corkysoft_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid internal API token")


def _run_import_with_optional_dry_run(
    importer,
    resource: str,
    records: List[Dict[str, Any]],
    *,
    dry_run: bool,
) -> None:
    with connection_scope(_current_db_path()) as conn:
        if not dry_run:
            importer(conn, resource, records, dry_run=False)
            return
        shadow = sqlite3.connect(":memory:")
        shadow.row_factory = sqlite3.Row
        try:
            conn.backup(shadow)
            importer(shadow, resource, records, dry_run=True)
        finally:
            shadow.close()


class JobType(BaseModel):
    """Descriptor for the type/category of a job."""

    code: str = Field(..., description="Short code representing the job type")
    text: Optional[str] = Field(
        default=None,
        description="Human readable text describing the job type",
    )


class BillingDetails(BaseModel):
    """Contact information for the billing entity."""

    name: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    address2: Optional[str] = None
    suburb: Optional[str] = None
    postcode: Optional[str] = None
    state: Optional[str] = None
    attention: Optional[str] = None
    code: Optional[str] = None


class ServiceDescriptor(BaseModel):
    """Identifier for the underlying service or product."""

    code: Optional[str] = None
    text: Optional[str] = None


class JobResponse(BaseModel):
    """Response payload for ``GET /jobs/{jobId}``."""

    id: str = Field(..., description="Primary identifier for the job")
    externalId: Optional[str] = Field(
        default=None,
        description="External system identifier for the job",
    )
    created: Optional[str] = Field(
        default=None,
        description="ISO8601 timestamp when the job was created",
    )
    lastUpdated: Optional[str] = Field(
        default=None,
        description="ISO8601 timestamp when the job was last updated",
    )
    lastUpdatedBy: Optional[str] = Field(
        default=None,
        description="User identifier responsible for the latest update",
    )
    type: JobType = Field(
        default_factory=lambda: JobType(code="R", text="Move"),
        description="Categorisation metadata for the job",
    )
    billing: Optional[BillingDetails] = Field(
        default=None,
        description="Billing contact details for the job",
    )
    service: Optional[ServiceDescriptor] = Field(
        default=None,
        description="Service information associated with the job",
    )
    origin: Optional[str] = Field(
        default=None,
        description="Free form origin description captured for the job",
    )
    destination: Optional[str] = Field(
        default=None,
        description="Free form destination description captured for the job",
    )
    revenue: Optional[float] = Field(
        default=None,
        description="Recorded revenue attributable to the job",
    )
    revenueTotal: Optional[float] = Field(
        default=None,
        description="Total revenue including adjustments",
    )
    volume: Optional[float] = Field(
        default=None,
        description="Volume captured for the job in native units",
    )
    volumeM3: Optional[float] = Field(
        default=None,
        description="Volume captured for the job expressed in cubic metres",
    )
    distanceKm: Optional[float] = Field(
        default=None,
        description="Distance for the job measured in kilometres",
    )
    finalCost: Optional[float] = Field(
        default=None,
        description="Final cost assigned to the job",
    )


class DriverShiftResponse(BaseModel):
    """Representation of a driver shift entry."""

    id: int = Field(..., description="Primary identifier for the shift")
    shiftDate: str = Field(..., description="Shift date in ISO format")
    truckId: Optional[str] = Field(
        default=None, description="Truck identifier attached to the shift"
    )
    truckName: Optional[str] = Field(
        default=None, description="Friendly truck name if available"
    )
    workerId: Optional[int] = Field(
        default=None, description="Worker identifier attached to the shift"
    )
    workerName: Optional[str] = Field(
        default=None, description="Driver or worker name for the shift"
    )
    ticketNumbers: Optional[str] = Field(
        default=None, description="Ticket numbers or references logged"
    )
    shiftStart: Optional[str] = Field(
        default=None, description="Recorded start time for the shift"
    )
    shiftEnd: Optional[str] = Field(
        default=None, description="Recorded finish time for the shift"
    )
    shiftWindowStart: Optional[str] = Field(
        default=None, description="Planned or rostered window start"
    )
    shiftWindowEnd: Optional[str] = Field(
        default=None, description="Planned or rostered window end"
    )
    role: Optional[str] = Field(default=None, description="Role assigned for the shift")
    hours: Optional[float] = Field(
        default=None, description="Duration of the shift in hours"
    )
    hourlyRate: Optional[float] = Field(
        default=None, description="Rate applied to the shift"
    )
    costTotal: Optional[float] = Field(
        default=None, description="Total cost recorded for the shift"
    )
    jobId: Optional[int] = Field(
        default=None, description="Job identifier the shift is linked to"
    )
    shipmentId: Optional[int] = Field(
        default=None, description="Shipment identifier the shift is linked to"
    )
    jobOrigin: Optional[str] = Field(
        default=None, description="Origin of the linked job if available"
    )
    jobDestination: Optional[str] = Field(
        default=None, description="Destination of the linked job if available"
    )
    source: Optional[str] = Field(
        default=None, description="Origin of the shift record (e.g. sheet tab)"
    )
    importedAt: Optional[str] = Field(
        default=None, description="Timestamp when the shift was last imported"
    )


class MovewareImportRequest(BaseModel):
    """Request payload for the MoveWare importer endpoint."""

    records: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Collection of raw MoveWare job dictionaries",
    )
    dry_run: bool = Field(
        default=False,
        description="When true, only validate the payload without persisting",
    )


class ImportSummary(BaseModel):
    """Summary of a MoveWare import invocation."""

    resource: str = Field(
        ..., description="Specific MoveWare resource targeted by the import"
    )
    imported: int = Field(
        ..., description="Number of records supplied to the importer"
    )
    dry_run: bool = Field(
        ..., description="Whether the invocation was executed in dry-run mode"
    )


class OperationsPolicyResponse(BaseModel):
    regoWarningDays: int
    coiWarningDays: int
    serviceWarningDays: int
    complianceWarningDays: int
    serviceOverdueBlocks: bool
    conflictBlocks: bool
    serviceOverrideAllowed: bool
    conflictOverrideAllowed: bool


class OperationsPolicyUpdateRequest(BaseModel):
    regoWarningDays: int
    coiWarningDays: int
    serviceWarningDays: int
    complianceWarningDays: int
    serviceOverdueBlocks: bool
    conflictBlocks: bool
    serviceOverrideAllowed: bool
    conflictOverrideAllowed: bool


class SegmentWorkerAssignmentRequest(BaseModel):
    workerId: int
    roleId: Optional[int] = None
    requiredComplianceIds: List[int] = Field(default_factory=list)
    startTime: Optional[str] = None
    endTime: Optional[str] = None


class SegmentAssignmentRequest(BaseModel):
    truckIds: List[str] = Field(default_factory=list)
    workerAssignments: List[SegmentWorkerAssignmentRequest] = Field(default_factory=list)
    override: bool = False
    overrideReasonCode: Optional[str] = None
    overrideNote: Optional[str] = None


class SegmentEnsureRequest(BaseModel):
    jobId: int
    segmentSequence: int
    fromLocation: Optional[str] = None
    toLocation: Optional[str] = None
    plannedStart: Optional[str] = None
    plannedEnd: Optional[str] = None


class SegmentTruckAssignmentResponse(BaseModel):
    truckId: str
    truckName: Optional[str] = None
    sourceImportedAt: Optional[str] = None


class SegmentWorkerAssignmentResponse(BaseModel):
    workerId: int
    workerName: Optional[str] = None
    roleId: Optional[int] = None
    requiredComplianceIds: List[str] = Field(default_factory=list)
    sourceImportedAt: Optional[str] = None


class SegmentReadinessResponse(BaseModel):
    segmentId: int
    jobId: int
    jobClient: Optional[str] = None
    jobOrigin: Optional[str] = None
    jobDestination: Optional[str] = None
    segmentSequence: int
    fromLocation: Optional[str] = None
    toLocation: Optional[str] = None
    plannedStart: Optional[str] = None
    plannedEnd: Optional[str] = None
    assignmentStatus: str
    warningFlags: List[str] = Field(default_factory=list)
    blockingFlags: List[str] = Field(default_factory=list)
    overrideableFlags: List[str] = Field(default_factory=list)
    overrideRequired: bool = False
    overrideReasonCode: Optional[str] = None
    overrideNote: Optional[str] = None
    truckAssignments: List[SegmentTruckAssignmentResponse] = Field(default_factory=list)
    workerAssignments: List[SegmentWorkerAssignmentResponse] = Field(default_factory=list)


class OperationalConflictResponse(BaseModel):
    segmentId: int
    jobId: int
    assignmentStatus: str
    flag: str


class OperationalReadinessItemResponse(BaseModel):
    resourceType: str
    resourceId: str
    resourceName: str
    status: str
    ruleType: str
    dueAt: Optional[str] = None
    overrideable: bool = False
    sourceImportedAt: Optional[str] = None
    details: Optional[str] = None


class WorkerRoleAssignmentRequest(BaseModel):
    roleId: Optional[int] = None
    roleName: Optional[str] = None
    description: Optional[str] = None


class WorkerComplianceAssignmentRequest(BaseModel):
    complianceId: Optional[int] = None
    complianceName: Optional[str] = None
    description: Optional[str] = None
    expiryDate: Optional[str] = None


class SegmentInventoryCoordinationResponse(BaseModel):
    segmentId: int
    jobId: int
    segmentSequence: int
    fromLocation: Optional[str] = None
    toLocation: Optional[str] = None
    plannedStart: Optional[str] = None
    plannedEnd: Optional[str] = None
    assignmentStatus: Optional[str] = None
    shipmentCount: int
    allocatedQuantity: float
    inventoryNames: List[str] = Field(default_factory=list)
    supplierNames: List[str] = Field(default_factory=list)


class SegmentInventoryAllocationRequest(BaseModel):
    inventoryItemId: int
    quantity: float
    status: str = "planned"


class JobBoardSegmentResponse(BaseModel):
    segmentId: int
    segmentSequence: int
    fromLocation: Optional[str] = None
    toLocation: Optional[str] = None
    plannedStart: Optional[str] = None
    plannedEnd: Optional[str] = None
    assignmentStatus: Optional[str] = None
    warningCount: int
    blockingCount: int
    overrideableCount: int
    truckIds: List[str] = Field(default_factory=list)
    workerNames: List[str] = Field(default_factory=list)
    inventoryNames: List[str] = Field(default_factory=list)
    supplierNames: List[str] = Field(default_factory=list)
    shipmentCount: int


class JobOperationsBoardResponse(BaseModel):
    jobId: int
    jobClient: Optional[str] = None
    jobOrigin: Optional[str] = None
    jobDestination: Optional[str] = None
    segmentCount: int
    plannedStart: Optional[str] = None
    plannedEnd: Optional[str] = None
    jobStatus: str
    warningCount: int
    blockingCount: int
    overrideableCount: int
    truckIds: List[str] = Field(default_factory=list)
    workerNames: List[str] = Field(default_factory=list)
    inventoryNames: List[str] = Field(default_factory=list)
    supplierNames: List[str] = Field(default_factory=list)
    segments: List[JobBoardSegmentResponse] = Field(default_factory=list)


class OperationsCutoverChecklistResponse(BaseModel):
    nativeReady: bool
    dualRunComplete: bool
    fallbackDrillComplete: bool
    operatorTrained: bool


class OperationsCutoverApprovalResponse(BaseModel):
    targetStatus: Optional[str] = None
    status: str
    requestPending: bool
    approvalSatisfied: bool
    blockedByApproval: bool
    requestedAt: Optional[str] = None
    requestedBy: Optional[str] = None
    requestNote: Optional[str] = None
    approvedAt: Optional[str] = None
    approvedBy: Optional[str] = None
    approvalNote: Optional[str] = None
    rejectedAt: Optional[str] = None
    rejectedBy: Optional[str] = None
    rejectionNote: Optional[str] = None


class OperationsCutoverRecommendationResponse(BaseModel):
    recommendedStatus: str
    actionable: bool
    reason: str
    approvalRequired: bool = False
    approvalSatisfied: bool = False
    blockedByApproval: bool = False


class OperationsCutoverWorkflowResponse(BaseModel):
    workflowKey: str
    label: str
    nativeSurface: str
    spreadsheetSource: str
    cutoverStatus: str
    ownerRole: Optional[str] = None
    snapshotMode: str
    snapshotFields: List[str] = Field(default_factory=list)
    fallbackMode: str
    metrics: Dict[str, Any]
    checklist: OperationsCutoverChecklistResponse
    allChecksComplete: bool
    targetMet: bool
    approval: OperationsCutoverApprovalResponse
    recommendation: OperationsCutoverRecommendationResponse
    lastDrillAt: Optional[str] = None
    rollbackInstructions: Optional[str] = None
    notes: Optional[str] = None
    updatedAt: Optional[str] = None


class OperationsCutoverWorkflowUpdateRequest(BaseModel):
    cutoverStatus: str
    ownerRole: Optional[str] = None
    snapshotMode: str = "none"
    snapshotFields: List[str] = Field(default_factory=list)
    fallbackMode: str = "import_only"
    cutoverTargetPercent: float = 100.0
    nativeReady: bool = False
    dualRunComplete: bool = False
    fallbackDrillComplete: bool = False
    operatorTrained: bool = False
    rollbackInstructions: Optional[str] = None
    notes: Optional[str] = None


class OperationsCutoverEventResponse(BaseModel):
    id: int
    workflowKey: str
    eventType: str
    actor: Optional[str] = None
    note: Optional[str] = None
    eventValue: Optional[str] = None
    createdAt: str


class OperationsCutoverEventRequest(BaseModel):
    eventType: str
    actor: Optional[str] = None
    note: Optional[str] = None
    eventValue: Optional[str] = None
    createdAt: Optional[str] = None


class OperationsCutoverTransitionRequest(BaseModel):
    actor: Optional[str] = None
    note: Optional[str] = None


class OperationsCutoverPromotionRequest(BaseModel):
    actor: str
    note: Optional[str] = None
    targetStatus: Optional[str] = None


class OperationsCutoverPromotionDecisionRequest(BaseModel):
    actor: str
    note: Optional[str] = None
    targetStatus: Optional[str] = None


class OperationsSyncResponse(BaseModel):
    fleetImported: int
    staffInserted: int
    staffUpdated: int
    suppliersImported: int
    staffSheetName: str
    suppliersSheetName: str


class OperationsSyncRequest(BaseModel):
    reference: Optional[str] = None


class CallEventCreateRequest(BaseModel):
    eventKind: str = Field(..., description="client_call, ops_call, manager_call, worker_call, clock_on_call, or clock_off_call")
    direction: str = Field(..., description="inbound, outbound, or internal")
    status: str = Field(default="completed", description="Current call status")
    sourceChannel: str = Field(default="telephony", description="telephony, whatsapp, manual_note, or imported_recording")
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
    transcriptText: Optional[str] = None
    confidence: Optional[float] = None
    isFinal: bool = True


class TranscriptUploadRequest(BaseModel):
    serviceKey: str = "ops"
    filename: str
    contentBase64: str
    language: Optional[str] = None
    diarize: bool = True


class FakeTranscriptRequest(BaseModel):
    serviceKey: str = "ops"
    scenario: Optional[str] = None
    operatorGoal: Optional[str] = None


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
    createdAt: str
    updatedAt: str


app = FastAPI(title="Corkysoft API", version="0.1.0")
app.include_router(kent_router)
app.include_router(labor_router)


def _optional_column(row: Any, column: str) -> Optional[Any]:
    """Return ``row[column]`` if the column exists, otherwise ``None``."""

    if hasattr(row, "keys") and column in row.keys():
        return row[column]
    try:
        return row[column]
    except (KeyError, TypeError, IndexError):
        return None


def _build_job_response(row: Any) -> JobResponse:
    """Transform a SQLite row into a :class:`JobResponse`."""

    job_type_code = _optional_column(row, "job_type_code") or "R"
    job_type_text = _optional_column(row, "job_type_text")
    if not job_type_text and job_type_code == "R":
        job_type_text = "Move"

    billing = BillingDetails(
        name=_optional_column(row, "billing_name")
        or _optional_column(row, "client"),
        email=_optional_column(row, "billing_email"),
        address=_optional_column(row, "billing_address"),
        address2=_optional_column(row, "billing_address2"),
        suburb=_optional_column(row, "billing_suburb"),
        postcode=_optional_column(row, "billing_postcode"),
        state=_optional_column(row, "billing_state"),
        attention=_optional_column(row, "billing_attention"),
        code=_optional_column(row, "billing_code"),
    )
    billing_payload = billing if billing.model_dump(exclude_none=True) else None

    service = ServiceDescriptor(
        code=_optional_column(row, "service_code"),
        text=_optional_column(row, "service_text"),
    )
    service_payload = service if service.model_dump(exclude_none=True) else None

    return JobResponse(
        id=str(_optional_column(row, "id")),
        externalId=_optional_column(row, "external_id"),
        created=_optional_column(row, "created_at")
        or _optional_column(row, "job_date"),
        lastUpdated=_optional_column(row, "updated_at"),
        lastUpdatedBy=_optional_column(row, "last_updated_by"),
        type=JobType(code=job_type_code, text=job_type_text),
        billing=billing_payload,
        service=service_payload,
        origin=_optional_column(row, "origin"),
        destination=_optional_column(row, "destination"),
        revenue=_optional_column(row, "revenue"),
        revenueTotal=_optional_column(row, "revenue_total"),
        volume=_optional_column(row, "volume"),
        volumeM3=_optional_column(row, "volume_m3"),
        distanceKm=_optional_column(row, "distance_km"),
        finalCost=_optional_column(row, "final_cost"),
    )


@app.post(
    "/importers/moveware/{resource}",
    response_model=ImportSummary,
    summary="Import MoveWare data",
)
def import_moveware_resource(
    resource: str = Path(..., description="MoveWare resource identifier"),
    payload: MovewareImportRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> ImportSummary:
    """Return a summary for an incoming MoveWare import request."""

    _run_import_with_optional_dry_run(
        import_moveware_records,
        resource,
        payload.records,
        dry_run=payload.dry_run,
    )
    return ImportSummary(
        resource=resource,
        imported=len(payload.records),
        dry_run=payload.dry_run,
    )


@app.post(
    "/importers/kent-ams/{resource}",
    response_model=ImportSummary,
    summary="Import Kent AMS adapter data",
)
def import_kent_ams_resource(
    resource: str = Path(..., description="Kent AMS resource identifier"),
    payload: MovewareImportRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> ImportSummary:
    """Return a summary for an incoming Kent AMS adapter import request."""

    _run_import_with_optional_dry_run(
        import_kent_ams_records,
        resource,
        payload.records,
        dry_run=payload.dry_run,
    )
    return ImportSummary(
        resource=resource,
        imported=len(payload.records),
        dry_run=payload.dry_run,
    )


@app.get(
    "/operations/policy",
    response_model=OperationsPolicyResponse,
    summary="Get operational readiness policy defaults",
)
def get_operations_assignment_policy() -> OperationsPolicyResponse:
    with connection_scope(_current_db_path()) as conn:
        payload = get_operations_policy(conn)
    return OperationsPolicyResponse(**payload)


@app.put(
    "/operations/policy",
    response_model=OperationsPolicyResponse,
    summary="Update operational readiness policy defaults",
)
def put_operations_assignment_policy(
    payload: OperationsPolicyUpdateRequest,
    _auth: None = Depends(require_internal_api_token),
) -> OperationsPolicyResponse:
    with connection_scope(_current_db_path()) as conn:
        updated = update_operations_policy(
            conn,
            rego_warning_days=payload.regoWarningDays,
            coi_warning_days=payload.coiWarningDays,
            service_warning_days=payload.serviceWarningDays,
            compliance_warning_days=payload.complianceWarningDays,
            service_overdue_blocks=payload.serviceOverdueBlocks,
            conflict_blocks=payload.conflictBlocks,
            service_override_allowed=payload.serviceOverrideAllowed,
            conflict_override_allowed=payload.conflictOverrideAllowed,
        )
    return OperationsPolicyResponse(**updated)


@app.post(
    "/operations/sync",
    response_model=OperationsSyncResponse,
    summary="Sync the shared operations workbook",
)
def post_operations_sync(
    payload: OperationsSyncRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> OperationsSyncResponse:
    with connection_scope(_current_db_path()) as conn:
        summary = sync_operations_workbook(conn, sheet_id_or_url=payload.reference)
    return OperationsSyncResponse(**summary)


@app.post(
    "/operations/segments",
    response_model=SegmentReadinessResponse,
    summary="Create or update a job segment for planning",
)
def post_operations_segment(
    payload: SegmentEnsureRequest,
    _auth: None = Depends(require_internal_api_token),
) -> SegmentReadinessResponse:
    with connection_scope(_current_db_path()) as conn:
        segment = ensure_segment(
            conn,
            job_id=payload.jobId,
            segment_sequence=payload.segmentSequence,
            from_location=payload.fromLocation,
            to_location=payload.toLocation,
            planned_start=payload.plannedStart,
            planned_end=payload.plannedEnd,
        )
        row = next(
            item
            for item in list_segment_readiness(conn, job_id=payload.jobId)
            if item["segmentId"] == int(segment["id"])
        )
    return SegmentReadinessResponse(**row)


@app.get(
    "/operations/segments/readiness",
    response_model=List[SegmentReadinessResponse],
    summary="List segment readiness and assignments",
)
def get_operations_segment_readiness(
    job_id: Optional[int] = Query(default=None, description="Optional job filter"),
    assignment_status: Optional[str] = Query(
        default=None,
        description="Optional assignment status filter",
    ),
) -> List[SegmentReadinessResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_segment_readiness(
            conn,
            job_id=job_id,
            assignment_status=assignment_status,
        )
    return [SegmentReadinessResponse(**row) for row in rows]


@app.post(
    "/operations/segments/{segment_id}/assign",
    response_model=SegmentReadinessResponse,
    summary="Assign trucks and workers to a job segment",
)
def post_operations_segment_assignment(
    segment_id: int = Path(..., description="Segment identifier"),
    payload: SegmentAssignmentRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> SegmentReadinessResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            assign_segment_resources(
                conn,
                segment_id=segment_id,
                truck_ids=payload.truckIds,
                worker_assignments=[item.model_dump() for item in payload.workerAssignments],
                override=payload.override,
                override_reason_code=payload.overrideReasonCode,
                override_note=payload.overrideNote,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        row = next(
            item
            for item in list_segment_readiness(conn)
            if item["segmentId"] == segment_id
        )
    return SegmentReadinessResponse(**row)


@app.get(
    "/operations/conflicts",
    response_model=List[OperationalConflictResponse],
    summary="List operational conflicts across planned segments",
)
def get_operations_conflicts(
    job_id: Optional[int] = Query(default=None, description="Optional job filter"),
) -> List[OperationalConflictResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_operational_conflicts(conn, job_id=job_id)
    return [OperationalConflictResponse(**row) for row in rows]


@app.get(
    "/operations/readiness/resources",
    response_model=List[OperationalReadinessItemResponse],
    summary="List due-soon and blocked operational readiness items",
)
def get_operations_readiness_resources(
    resource_type: Optional[str] = Query(
        default=None,
        description="Optional filter: vehicle or worker",
    ),
    status: Optional[str] = Query(
        default=None,
        description="Optional filter: warning or blocked",
    ),
) -> List[OperationalReadinessItemResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_operational_readiness_items(
            conn,
            resource_type=resource_type,
            status=status,
        )
    return [OperationalReadinessItemResponse(**row) for row in rows]


@app.post(
    "/operations/workers/{worker_id}/roles",
    summary="Assign a role to a worker for operational planning",
)
def post_operations_worker_role(
    worker_id: int = Path(..., description="Worker identifier"),
    payload: WorkerRoleAssignmentRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> dict[str, Any]:
    if payload.roleId is None and not (payload.roleName or "").strip():
        raise HTTPException(status_code=400, detail="Provide roleId or roleName")
    with connection_scope(_current_db_path()) as conn:
        try:
            role_id = (
                int(payload.roleId)
                if payload.roleId is not None
                else ensure_worker_role(
                    conn,
                    name=str(payload.roleName).strip(),
                    description=payload.description or "",
                )
            )
            assign_worker_role(conn, worker_id=worker_id, role_id=role_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"workerId": worker_id, "roleId": role_id}


@app.post(
    "/operations/workers/{worker_id}/compliances",
    summary="Assign a compliance to a worker for operational planning",
)
def post_operations_worker_compliance(
    worker_id: int = Path(..., description="Worker identifier"),
    payload: WorkerComplianceAssignmentRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> dict[str, Any]:
    if payload.complianceId is None and not (payload.complianceName or "").strip():
        raise HTTPException(status_code=400, detail="Provide complianceId or complianceName")
    with connection_scope(_current_db_path()) as conn:
        try:
            compliance_id = (
                int(payload.complianceId)
                if payload.complianceId is not None
                else ensure_worker_compliance(
                    conn,
                    name=str(payload.complianceName).strip(),
                    description=payload.description or "",
                )
            )
            assign_worker_compliance(
                conn,
                worker_id=worker_id,
                compliance_id=compliance_id,
                expiry_date=payload.expiryDate,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"workerId": worker_id, "complianceId": compliance_id, "expiryDate": payload.expiryDate}


@app.get(
    "/operations/inventory/segments",
    response_model=List[SegmentInventoryCoordinationResponse],
    summary="List segment-linked inventory and supplier coordination",
)
def get_operations_inventory_segments(
    job_id: Optional[int] = Query(default=None, description="Optional job filter"),
) -> List[SegmentInventoryCoordinationResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_segment_inventory_coordination(conn, job_id=job_id)
    return [SegmentInventoryCoordinationResponse(**row) for row in rows]


@app.get(
    "/operations/jobs/board",
    response_model=List[JobOperationsBoardResponse],
    summary="List job-centric operational board rows",
)
def get_operations_jobs_board(
    job_id: Optional[int] = Query(default=None, description="Optional job filter"),
) -> List[JobOperationsBoardResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_job_operations_board(conn, job_id=job_id)
    return [JobOperationsBoardResponse(**row) for row in rows]


@app.get(
    "/operations/cutover/workflows",
    response_model=List[OperationsCutoverWorkflowResponse],
    summary="List spreadsheet-to-native workflow cutover status",
)
def get_operations_cutover_workflows() -> List[OperationsCutoverWorkflowResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_operations_cutover_rollout(conn)
    return [OperationsCutoverWorkflowResponse(**row) for row in rows]


@app.put(
    "/operations/cutover/workflows/{workflow_key}",
    response_model=OperationsCutoverWorkflowResponse,
    summary="Update workflow cutover status and fallback rules",
)
def put_operations_cutover_workflow(
    workflow_key: str = Path(..., description="Workflow identifier"),
    payload: OperationsCutoverWorkflowUpdateRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> OperationsCutoverWorkflowResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = upsert_operations_cutover_workflow(
                conn,
                workflow_key=workflow_key,
                cutover_status=payload.cutoverStatus,
                owner_role=payload.ownerRole,
                snapshot_mode=payload.snapshotMode,
                snapshot_fields=payload.snapshotFields,
                fallback_mode=payload.fallbackMode,
                cutover_target_percent=payload.cutoverTargetPercent,
                native_ready=payload.nativeReady,
                dual_run_complete=payload.dualRunComplete,
                fallback_drill_complete=payload.fallbackDrillComplete,
                operator_trained=payload.operatorTrained,
                rollback_instructions=payload.rollbackInstructions,
                notes=payload.notes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        row = next(item for item in list_operations_cutover_rollout(conn) if item["workflowKey"] == workflow_key)
    return OperationsCutoverWorkflowResponse(**row)


@app.post(
    "/operations/cutover/workflows/{workflow_key}/request-promotion",
    response_model=OperationsCutoverWorkflowResponse,
    summary="Request the next guarded cutover promotion",
)
def post_operations_cutover_request_promotion(
    workflow_key: str = Path(..., description="Workflow identifier"),
    payload: OperationsCutoverPromotionRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> OperationsCutoverWorkflowResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = request_operations_cutover_promotion(
                conn,
                workflow_key=workflow_key,
                actor=payload.actor,
                note=payload.note,
                target_status=payload.targetStatus,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return OperationsCutoverWorkflowResponse(**row)


@app.post(
    "/operations/cutover/workflows/{workflow_key}/approve-promotion",
    response_model=OperationsCutoverWorkflowResponse,
    summary="Approve the next guarded cutover promotion",
)
def post_operations_cutover_approve_promotion(
    workflow_key: str = Path(..., description="Workflow identifier"),
    payload: OperationsCutoverPromotionDecisionRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> OperationsCutoverWorkflowResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = approve_operations_cutover_promotion(
                conn,
                workflow_key=workflow_key,
                actor=payload.actor,
                note=payload.note,
                target_status=payload.targetStatus,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return OperationsCutoverWorkflowResponse(**row)


@app.post(
    "/operations/cutover/workflows/{workflow_key}/reject-promotion",
    response_model=OperationsCutoverWorkflowResponse,
    summary="Reject the next guarded cutover promotion",
)
def post_operations_cutover_reject_promotion(
    workflow_key: str = Path(..., description="Workflow identifier"),
    payload: OperationsCutoverPromotionDecisionRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> OperationsCutoverWorkflowResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = reject_operations_cutover_promotion(
                conn,
                workflow_key=workflow_key,
                actor=payload.actor,
                note=payload.note or "",
                target_status=payload.targetStatus,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return OperationsCutoverWorkflowResponse(**row)


@app.post(
    "/operations/cutover/workflows/{workflow_key}/apply-recommendation",
    response_model=OperationsCutoverWorkflowResponse,
    summary="Apply the guarded recommended cutover transition",
)
def post_operations_cutover_apply_recommendation(
    workflow_key: str = Path(..., description="Workflow identifier"),
    payload: OperationsCutoverTransitionRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> OperationsCutoverWorkflowResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = apply_operations_cutover_recommendation(
                conn,
                workflow_key=workflow_key,
                actor=payload.actor,
                note=payload.note,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return OperationsCutoverWorkflowResponse(**row)


@app.get(
    "/operations/cutover/events",
    response_model=List[OperationsCutoverEventResponse],
    summary="List spreadsheet cutover events",
)
def get_operations_cutover_events(
    workflow_key: Optional[str] = Query(default=None, description="Optional workflow filter"),
    event_type: Optional[str] = Query(default=None, description="Optional event-type filter"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum events to return"),
) -> List[OperationsCutoverEventResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_operations_cutover_events(
            conn,
            workflow_key=workflow_key,
            event_type=event_type,
            limit=limit,
        )
    return [OperationsCutoverEventResponse(**row) for row in rows]


@app.post(
    "/operations/cutover/workflows/{workflow_key}/events",
    response_model=OperationsCutoverEventResponse,
    summary="Record a spreadsheet cutover event",
)
def post_operations_cutover_event(
    workflow_key: str = Path(..., description="Workflow identifier"),
    payload: OperationsCutoverEventRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> OperationsCutoverEventResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = record_operations_cutover_event(
                conn,
                workflow_key=workflow_key,
                event_type=payload.eventType,
                actor=payload.actor,
                note=payload.note,
                event_value=payload.eventValue,
                created_at=payload.createdAt,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return OperationsCutoverEventResponse(**row)


@app.post(
    "/operations/inventory/segments/{segment_id}/allocate",
    summary="Allocate inventory to a planned job segment",
)
def post_operations_inventory_allocation(
    segment_id: int = Path(..., description="Segment identifier"),
    payload: SegmentInventoryAllocationRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> dict[str, Any]:
    with connection_scope(_current_db_path()) as conn:
        try:
            shipment = allocate_inventory_to_segment(
                conn,
                segment_id=segment_id,
                inventory_item_id=payload.inventoryItemId,
                quantity=payload.quantity,
                status=payload.status,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "segmentId": segment_id,
        "shipmentId": int(shipment["id"]),
        "inventoryItemId": payload.inventoryItemId,
        "quantity": payload.quantity,
        "status": payload.status,
    }


@app.get(
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


@app.post(
    "/calls/sessions",
    response_model=CallSessionResponse,
    summary="Create a routed call session with an initial leg",
)
def post_call_session(
    payload: CallSessionCreateRequest,
    _auth: None = Depends(require_internal_api_token),
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
            operator_id=payload.operatorId,
            started_at=payload.startedAt,
            ended_at=payload.endedAt,
            captured_at=payload.capturedAt,
            correlation_id=payload.correlationId,
            initial_destination_kind=payload.initialDestinationKind,
            initial_destination_label=payload.initialDestinationLabel,
        )
    return CallSessionResponse(**row)


@app.get(
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


@app.get(
    "/calls/sessions/{call_session_id}/legs",
    response_model=List[CallLegResponse],
    summary="List call legs for a session",
)
def get_call_session_legs(call_session_id: int = Path(...)) -> List[CallLegResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_call_legs(conn, call_session_id=call_session_id)
    return [CallLegResponse(**row) for row in rows]


@app.post(
    "/calls/sessions/{call_session_id}/legs",
    response_model=CallLegResponse,
    summary="Add a routed or consult leg to a call session",
)
def post_call_session_leg(
    call_session_id: int = Path(...),
    payload: CallLegCreateRequest = ...,
    _auth: None = Depends(require_internal_api_token),
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
                operator_id=payload.operatorId,
                caller_phone=payload.callerPhone,
                callee_phone=payload.calleePhone,
                started_at=payload.startedAt,
                answered_at=payload.answeredAt,
                ended_at=payload.endedAt,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CallLegResponse(**row)


@app.get(
    "/calls/sessions/{call_session_id}/routing-events",
    response_model=List[CallRoutingEventResponse],
    summary="List routing events for a call session",
)
def get_call_session_routing_events(call_session_id: int = Path(...)) -> List[CallRoutingEventResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_call_routing_events(conn, call_session_id=call_session_id)
    return [CallRoutingEventResponse(**row) for row in rows]


@app.post(
    "/calls/sessions/{call_session_id}/routing-events",
    response_model=CallRoutingEventResponse,
    summary="Record a routing or transfer event for a call session",
)
def post_call_session_routing_event(
    call_session_id: int = Path(...),
    payload: CallRoutingEventCreateRequest = ...,
    _auth: None = Depends(require_internal_api_token),
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
                actor=payload.actor,
                detail=payload.detail,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CallRoutingEventResponse(**row)


@app.get(
    "/calls/ambient-sessions",
    response_model=List[AmbientSessionResponse],
    summary="List ambient office transcript sessions",
)
def get_ambient_call_sessions(limit: int = Query(default=100, ge=1, le=500)) -> List[AmbientSessionResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_ambient_sessions(conn, limit=limit)
    return [AmbientSessionResponse(**row) for row in rows]


@app.post(
    "/calls/ambient-sessions",
    response_model=AmbientSessionResponse,
    summary="Create an ambient office transcript session",
)
def post_ambient_call_session(
    payload: AmbientSessionCreateRequest,
    _auth: None = Depends(require_internal_api_token),
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
            operator_id=payload.operatorId,
            started_at=payload.startedAt,
            ended_at=payload.endedAt,
            captured_at=payload.capturedAt,
            correlation_id=payload.correlationId,
        )
    return AmbientSessionResponse(**row)


@app.get(
    "/calls/ambient-sessions/{ambient_session_id}",
    response_model=AmbientSessionResponse,
    summary="Get a single ambient office transcript session",
)
def get_ambient_call_session_by_id(ambient_session_id: int = Path(...)) -> AmbientSessionResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = get_ambient_session(conn, ambient_session_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AmbientSessionResponse(**row)


@app.get(
    "/calls/ambient-sessions/{ambient_session_id}/transcripts",
    response_model=List[AmbientTranscriptArtifactResponse],
    summary="List transcript artifacts for an ambient office session",
)
def get_ambient_call_session_transcripts(ambient_session_id: int = Path(...)) -> List[AmbientTranscriptArtifactResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_ambient_transcript_artifacts(conn, ambient_session_id=ambient_session_id)
    return [AmbientTranscriptArtifactResponse(**row) for row in rows]


@app.post(
    "/calls/ambient-sessions/{ambient_session_id}/transcripts/manual",
    response_model=AmbientTranscriptArtifactResponse,
    summary="Create a manual transcript artifact for an ambient session",
)
def post_ambient_call_session_transcript(
    ambient_session_id: int = Path(...),
    payload: TranscriptArtifactCreateRequest = ...,
    _auth: None = Depends(require_internal_api_token),
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
    return AmbientTranscriptArtifactResponse(**row)


@app.post(
    "/calls/ambient-sessions/{ambient_session_id}/transcripts/fake",
    response_model=AmbientTranscriptArtifactResponse,
    summary="Generate a fake transcript artifact for an ambient office session",
)
def post_fake_ambient_call_session_transcript(
    ambient_session_id: int = Path(...),
    payload: FakeTranscriptRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> AmbientTranscriptArtifactResponse:
    with connection_scope(_current_db_path()) as conn:
        row = generate_fake_ambient_transcript_artifact(
            conn,
            ambient_session_id=ambient_session_id,
            scenario=payload.scenario,
            operator_goal=payload.operatorGoal,
            service_key=payload.serviceKey,
        )
    return AmbientTranscriptArtifactResponse(**row)


@app.get(
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


@app.post(
    "/calls/events",
    response_model=CallEventResponse,
    summary="Create a call event and auto-link by phone where possible",
)
def post_call_event(
    payload: CallEventCreateRequest,
    _auth: None = Depends(require_internal_api_token),
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
            operator_id=payload.operatorId,
            started_at=payload.startedAt,
            ended_at=payload.endedAt,
            captured_at=payload.capturedAt,
            correlation_id=payload.correlationId,
        )
    return CallEventResponse(**row)


@app.get(
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


@app.get(
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


@app.post(
    "/calls/events/{call_event_id}/notes",
    response_model=CallNoteResponse,
    summary="Add an authoritative or advisory note to a call event",
)
def post_call_note(
    call_event_id: int = Path(..., description="Call event identifier"),
    payload: CallNoteCreateRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> CallNoteResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = add_call_note(
                conn,
                call_event_id=call_event_id,
                author=payload.author,
                note_text=payload.noteText,
                note_kind=payload.noteKind,
                authoritative=payload.authoritative,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CallNoteResponse(**row)


@app.get(
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


@app.post(
    "/calls/events/{call_event_id}/extracted-actions",
    response_model=ExtractedActionResponse,
    summary="Add an extracted action candidate for review",
)
def post_call_extracted_action(
    call_event_id: int = Path(..., description="Call event identifier"),
    payload: ExtractedActionCreateRequest = ...,
    _auth: None = Depends(require_internal_api_token),
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
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExtractedActionResponse(**row)


@app.post(
    "/calls/extracted-actions/{action_id}/decision",
    response_model=ExtractedActionResponse,
    summary="Accept or reject an extracted action candidate",
)
def post_call_extracted_action_decision(
    action_id: int = Path(..., description="Extracted action identifier"),
    payload: ExtractedActionDecisionRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> ExtractedActionResponse:
    if payload.status not in EXTRACTED_ACTION_STATUSES or payload.status == "pending":
        raise HTTPException(status_code=400, detail="Decision status must be accepted or rejected")
    with connection_scope(_current_db_path()) as conn:
        try:
            row = decide_extracted_action(
                conn,
                action_id=action_id,
                status=payload.status,
                decided_by=payload.decidedBy,
                decision_note=payload.decisionNote,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExtractedActionResponse(**row)


@app.post(
    "/calls/events/{call_event_id}/resolve",
    response_model=CallEventResponse,
    summary="Resolve a call event to client/job/segment/worker context",
)
def post_call_link_resolution(
    call_event_id: int = Path(..., description="Call event identifier"),
    payload: CallLinkResolutionRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> CallEventResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = resolve_call_links(
                conn,
                call_event_id=call_event_id,
                actor=payload.actor,
                client_id=payload.clientId,
                quote_id=payload.quoteId,
                job_id=payload.jobId,
                segment_id=payload.segmentId,
                worker_id=payload.workerId,
                resolution_note=payload.resolutionNote,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CallEventResponse(**row)


@app.get(
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


@app.post(
    "/calls/legs/{call_leg_id}/transcripts/manual",
    response_model=TranscriptArtifactResponse,
    summary="Create a manual transcript artifact for a call leg",
)
def post_manual_call_leg_transcript(
    call_leg_id: int = Path(..., description="Call leg identifier"),
    payload: TranscriptArtifactCreateRequest = ...,
    _auth: None = Depends(require_internal_api_token),
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
    return TranscriptArtifactResponse(**row)


@app.post(
    "/calls/legs/{call_leg_id}/transcripts/fake",
    response_model=TranscriptArtifactResponse,
    summary="Generate a fake transcript artifact for a call leg",
)
def post_fake_call_leg_transcript(
    call_leg_id: int = Path(..., description="Call leg identifier"),
    payload: FakeTranscriptRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> TranscriptArtifactResponse:
    with connection_scope(_current_db_path()) as conn:
        row = generate_fake_transcript_artifact(
            conn,
            call_leg_id=call_leg_id,
            scenario=payload.scenario,
            operator_goal=payload.operatorGoal,
            service_key=payload.serviceKey,
        )
    return TranscriptArtifactResponse(**row)


@app.post(
    "/calls/legs/{call_leg_id}/transcripts/upload",
    response_model=TranscriptArtifactResponse,
    summary="Submit audio to WhisperX and create a queued transcript artifact for a call leg",
)
def post_call_leg_transcript_upload(
    call_leg_id: int = Path(..., description="Call leg identifier"),
    payload: TranscriptUploadRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> TranscriptArtifactResponse:
    try:
        file_bytes = base64.b64decode(payload.contentBase64.encode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="contentBase64 must contain valid base64 data") from exc
    with connection_scope(_current_db_path()) as conn:
        row = submit_call_audio_for_transcription(
            conn,
            call_leg_id=call_leg_id,
            service_key=payload.serviceKey,
            file_bytes=file_bytes,
            filename=payload.filename or "call_audio.bin",
            language=payload.language,
            diarize=payload.diarize,
        )
    return TranscriptArtifactResponse(**row)


@app.get(
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


@app.post(
    "/calls/events/{call_event_id}/transcripts/manual",
    response_model=TranscriptArtifactResponse,
    summary="Create a manual transcript artifact",
)
def post_manual_call_transcript(
    call_event_id: int = Path(..., description="Call event identifier"),
    payload: TranscriptArtifactCreateRequest = ...,
    _auth: None = Depends(require_internal_api_token),
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
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TranscriptArtifactResponse(**row)


@app.post(
    "/calls/events/{call_event_id}/transcripts/fake",
    response_model=TranscriptArtifactResponse,
    summary="Generate a fake transcript artifact for workflow testing",
)
def post_fake_call_transcript(
    call_event_id: int = Path(..., description="Call event identifier"),
    payload: FakeTranscriptRequest = ...,
    _auth: None = Depends(require_internal_api_token),
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
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TranscriptArtifactResponse(**row)


@app.post(
    "/calls/events/{call_event_id}/transcripts/upload",
    response_model=TranscriptArtifactResponse,
    summary="Submit audio to WhisperX and create a queued transcript artifact",
)
def post_call_transcript_upload(
    call_event_id: int = Path(..., description="Call event identifier"),
    payload: TranscriptUploadRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> TranscriptArtifactResponse:
    try:
        file_bytes = base64.b64decode(payload.contentBase64.encode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="contentBase64 must contain valid base64 data") from exc
    with connection_scope(_current_db_path()) as conn:
        try:
            row = submit_call_audio_for_transcription(
                conn,
                call_event_id=call_event_id,
                service_key=payload.serviceKey,
                file_bytes=file_bytes,
                filename=payload.filename or "call_audio.bin",
                language=payload.language,
                diarize=payload.diarize,
            )
        except WhisperXAdapterError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TranscriptArtifactResponse(**row)


@app.post(
    "/calls/transcripts/{artifact_id}/poll",
    response_model=TranscriptArtifactResponse,
    summary="Poll WhisperX for transcript task completion",
)
def post_call_transcript_poll(
    artifact_id: int = Path(..., description="Transcript artifact identifier"),
    _auth: None = Depends(require_internal_api_token),
) -> TranscriptArtifactResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = poll_transcript_artifact(conn, artifact_id=artifact_id)
        except WhisperXAdapterError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TranscriptArtifactResponse(**row)


@app.get(
    "/worker-time/events",
    response_model=List[WorkerTimeCaptureResponse],
    summary="List worker time-capture events and review state",
)
def get_worker_time_events(
    review_status: Optional[str] = Query(default=None, description="Optional review-status filter"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum rows to return"),
) -> List[WorkerTimeCaptureResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_worker_time_capture_events(conn, review_status=review_status, limit=limit)
    return [WorkerTimeCaptureResponse(**row) for row in rows]


@app.post(
    "/worker-time/events",
    response_model=WorkerTimeCaptureResponse,
    summary="Record a worker time-capture event from app, WhatsApp, or voice call",
)
def post_worker_time_event(
    payload: WorkerTimeCaptureCreateRequest,
    _auth: None = Depends(require_internal_api_token),
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
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WorkerTimeCaptureResponse(**row)


@app.post(
    "/worker-time/events/{event_id}/decision",
    response_model=WorkerTimeCaptureResponse,
    summary="Accept or reject a worker time-capture event after review",
)
def post_worker_time_event_decision(
    event_id: int = Path(..., description="Worker time event identifier"),
    payload: WorkerTimeCaptureDecisionRequest = ...,
    _auth: None = Depends(require_internal_api_token),
) -> WorkerTimeCaptureResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = decide_worker_time_capture_event(
                conn,
                event_id=event_id,
                review_status=payload.reviewStatus,
                reviewer=payload.reviewer,
                review_note=payload.reviewNote,
                worker_id=payload.workerId,
                job_id=payload.jobId,
                segment_id=payload.segmentId,
                truck_id=payload.truckId,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WorkerTimeCaptureResponse(**row)


@app.get(
    "/state-egress/events",
    response_model=List[StateEgressEventResponse],
    summary="List append-only downstream state events prepared for StatiBaker-like consumers",
)
def get_state_egress_events(
    limit: int = Query(default=100, ge=1, le=500, description="Maximum rows to return"),
    surface: str = Query(default="all", description="all, call_ops, or observer"),
    event_family: Optional[str] = Query(default=None, alias="eventFamily", description="Optional observer event family"),
    job_id: Optional[int] = Query(default=None, alias="jobId", description="Optional job identifier filter"),
    start_time: Optional[str] = Query(default=None, alias="startTime", description="Earliest event time to return"),
    end_time: Optional[str] = Query(default=None, alias="endTime", description="Latest event time to return"),
    cursor: Optional[str] = Query(default=None, description="Opaque cursor returned by a previous listing call"),
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
                if row.get("surface") != "call_ops" or int(row["payload"].get("jobId") or -1) == target
            ]
        if start_time is not None:
            payload = [row for row in payload if str(row.get("eventTime") or row["occurredAt"]) >= start_time]
        if end_time is not None:
            payload = [row for row in payload if str(row.get("eventTime") or row["occurredAt"]) <= end_time]
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


@app.post(
    "/state-egress/operations-diary-export",
    response_model=OperationsDiaryObserverExportResponse,
    summary="Emit append-only observer envelopes for operations-diary planning snapshots and reconciliation exceptions",
)
def post_operations_diary_export(
    payload: OperationsDiaryObserverExportRequest,
    _auth: None = Depends(require_internal_api_token),
) -> OperationsDiaryObserverExportResponse:
    with connection_scope(_current_db_path()) as conn:
        row = export_operations_diary_observer_events(
            conn,
            anchor_date=payload.anchorDate,
            view_mode=payload.viewMode,
            focus_job_id=payload.focusJobId,
            actor_ref=payload.actorRef,
            include_planning_snapshot=payload.includePlanningSnapshot,
            include_reconciliation_exceptions=payload.includeReconciliationExceptions,
        )
    return OperationsDiaryObserverExportResponse(
        anchorDate=row["anchorDate"],
        viewMode=row["viewMode"],
        emittedCount=row["emittedCount"],
        byFamily=row["byFamily"],
        events=[StateEgressEventResponse(**item) for item in row["events"]],
    )


@app.get(
    "/jobs/{jobId}",
    response_model=JobResponse,
    summary="Fetch a job by its identifier",
)
def get_job(jobId: str = Path(..., description="Unique job identifier")) -> JobResponse:
    """Return a job from the ``jobs`` table."""

    with connection_scope(_current_db_path()) as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (jobId,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    job_id = _optional_column(row, "id")
    if job_id is None:
        raise HTTPException(status_code=500, detail="Job row missing identifier")
    return _build_job_response(row)


@app.get(
    "/driver-shifts",
    response_model=List[DriverShiftResponse],
    summary="List driver shifts with optional filters",
)
def list_driver_shifts(
    start_date: Optional[str] = Query(
        default=None, description="Earliest shift date (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = Query(
        default=None, description="Latest shift date (YYYY-MM-DD)"
    ),
    workers: Optional[List[str]] = Query(
        default=None, description="Filter results to specific worker names"
    ),
    trucks: Optional[List[str]] = Query(
        default=None, description="Filter results to specific truck identifiers"
    ),
) -> List[DriverShiftResponse]:
    """Return driver shifts stored in the ``driver_shifts`` table."""

    with connection_scope(_current_db_path()) as conn:
        rows = fetch_driver_shifts(
            conn,
            start_date=start_date,
            end_date=end_date,
            worker_names=workers,
            truck_ids=trucks,
        )

    responses: List[DriverShiftResponse] = []
    for row in rows:
        responses.append(
            DriverShiftResponse(
                id=int(row["id"]),
                shiftDate=row["shift_date"],
                truckId=row["truck_id"],
                truckName=row["truck_name"],
                workerId=row["worker_id"],
                workerName=row["worker_name"],
                ticketNumbers=row["ticket_numbers"],
                shiftStart=row["shift_start"],
                shiftEnd=row["shift_end"],
                shiftWindowStart=_optional_column(row, "shift_window_start"),
                shiftWindowEnd=_optional_column(row, "shift_window_end"),
                role=_optional_column(row, "role"),
                hours=row["hours"],
                hourlyRate=row["hourly_rate"],
                costTotal=row["cost_total"],
                jobId=_optional_column(row, "linked_job_id"),
                shipmentId=_optional_column(row, "shipment_id"),
                jobOrigin=_optional_column(row, "job_origin"),
                jobDestination=_optional_column(row, "job_destination"),
                source=row["source"],
                importedAt=row["imported_at"],
            )
        )
    return responses


__all__ = [
    "app",
    "JobResponse",
    "DriverShiftResponse",
    "KentTenderCalibrationBandResponse",
    "KentTenderCalibrationResponse",
    "KentTenderCalibrationSummaryResponse",
    "KentTenderPriorityResponse",
    "MovewareImportRequest",
    "ImportSummary",
]
