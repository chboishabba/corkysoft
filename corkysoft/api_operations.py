from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from analytics.db.inventory import (
    allocate_inventory_to_segment,
    list_segment_inventory_coordination,
)
from analytics.db_connection import connection_scope
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
    list_operational_conflicts,
    list_operational_readiness_items,
    list_segment_readiness,
    record_operations_cutover_event,
    reject_operations_cutover_promotion,
    request_operations_cutover_promotion,
    update_operations_policy,
    upsert_operations_cutover_workflow,
)
from analytics.operations_workbook import sync_operations_workbook
from corkysoft.api_shared import (
    OPERATIONS_CUTOVER_APPROVE_SCOPE,
    OPERATIONS_CUTOVER_WRITE_SCOPE,
    OPERATIONS_WRITE_SCOPE,
    ApiAuthContext,
    _current_db_path,
    record_api_write_receipt,
    require_api_auth_context,
    require_internal_api_read_token,
)

router = APIRouter(dependencies=[Depends(require_internal_api_read_token)])
require_operations_cutover_write = require_api_auth_context(
    (OPERATIONS_CUTOVER_WRITE_SCOPE,)
)
require_operations_cutover_approval = require_api_auth_context(
    (OPERATIONS_CUTOVER_APPROVE_SCOPE,)
)
require_operations_write = require_api_auth_context((OPERATIONS_WRITE_SCOPE,))


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


@router.get("/operations/policy", response_model=OperationsPolicyResponse, summary="Get operational readiness policy defaults")
def get_operations_assignment_policy() -> OperationsPolicyResponse:
    with connection_scope(_current_db_path()) as conn:
        payload = get_operations_policy(conn)
    return OperationsPolicyResponse(**payload)


@router.put("/operations/policy", response_model=OperationsPolicyResponse, summary="Update operational readiness policy defaults")
def put_operations_assignment_policy(
    request: Request,
    payload: OperationsPolicyUpdateRequest,
    auth: ApiAuthContext = Depends(require_operations_write),
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
        _receipt(
            conn,
            auth=auth,
            action="operations.policy.update",
            resource_type="operations_policy",
            resource_id="default",
            request=request,
        )
    return OperationsPolicyResponse(**updated)


@router.post("/operations/sync", response_model=OperationsSyncResponse, summary="Sync the shared operations workbook")
def post_operations_sync(
    request: Request,
    payload: OperationsSyncRequest = ...,
    auth: ApiAuthContext = Depends(require_operations_write),
) -> OperationsSyncResponse:
    from corkysoft import api as api_module

    sync_workbook = getattr(api_module, "sync_operations_workbook", sync_operations_workbook)
    with connection_scope(_current_db_path()) as conn:
        summary = sync_workbook(conn, sheet_id_or_url=payload.reference)
        _receipt(
            conn,
            auth=auth,
            action="operations.workbook.sync",
            resource_type="operations_workbook",
            resource_id=payload.reference or "configured",
            request=request,
        )
    return OperationsSyncResponse(**summary)


@router.post("/operations/segments", response_model=SegmentReadinessResponse, summary="Create or update a job segment for planning")
def post_operations_segment(
    request: Request,
    payload: SegmentEnsureRequest,
    auth: ApiAuthContext = Depends(require_operations_write),
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
        row = next(item for item in list_segment_readiness(conn, job_id=payload.jobId) if item["segmentId"] == int(segment["id"]))
        _receipt(
            conn,
            auth=auth,
            action="operations.segment.ensure",
            resource_type="job_segment",
            resource_id=segment["id"],
            request=request,
        )
    return SegmentReadinessResponse(**row)


@router.get("/operations/segments/readiness", response_model=List[SegmentReadinessResponse], summary="List segment readiness and assignments")
def get_operations_segment_readiness(
    job_id: Optional[int] = Query(default=None, description="Optional job filter"),
    assignment_status: Optional[str] = Query(default=None, description="Optional assignment status filter"),
) -> List[SegmentReadinessResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_segment_readiness(conn, job_id=job_id, assignment_status=assignment_status)
    return [SegmentReadinessResponse(**row) for row in rows]


@router.post("/operations/segments/{segment_id}/assign", response_model=SegmentReadinessResponse, summary="Assign trucks and workers to a job segment")
def post_operations_segment_assignment(
    request: Request,
    segment_id: int = Path(..., description="Segment identifier"),
    payload: SegmentAssignmentRequest = ...,
    auth: ApiAuthContext = Depends(require_operations_write),
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
            _receipt(
                conn,
                auth=auth,
                action="operations.segment.assign",
                resource_type="job_segment",
                resource_id=segment_id,
                request=request,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        row = next(item for item in list_segment_readiness(conn) if item["segmentId"] == segment_id)
    return SegmentReadinessResponse(**row)


@router.get("/operations/conflicts", response_model=List[OperationalConflictResponse], summary="List operational conflicts across planned segments")
def get_operations_conflicts(
    job_id: Optional[int] = Query(default=None, description="Optional job filter"),
) -> List[OperationalConflictResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_operational_conflicts(conn, job_id=job_id)
    return [OperationalConflictResponse(**row) for row in rows]


@router.get("/operations/readiness/resources", response_model=List[OperationalReadinessItemResponse], summary="List due-soon and blocked operational readiness items")
def get_operations_readiness_resources(
    resource_type: Optional[str] = Query(default=None, description="Optional filter: vehicle or worker"),
    status: Optional[str] = Query(default=None, description="Optional filter: warning or blocked"),
) -> List[OperationalReadinessItemResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_operational_readiness_items(conn, resource_type=resource_type, status=status)
    return [OperationalReadinessItemResponse(**row) for row in rows]


@router.post("/operations/workers/{worker_id}/roles", summary="Assign a role to a worker for operational planning")
def post_operations_worker_role(
    request: Request,
    worker_id: int = Path(..., description="Worker identifier"),
    payload: WorkerRoleAssignmentRequest = ...,
    auth: ApiAuthContext = Depends(require_operations_write),
) -> dict[str, Any]:
    if payload.roleId is None and not (payload.roleName or "").strip():
        raise HTTPException(status_code=400, detail="Provide roleId or roleName")
    with connection_scope(_current_db_path()) as conn:
        try:
            role_id = int(payload.roleId) if payload.roleId is not None else ensure_worker_role(conn, name=str(payload.roleName).strip(), description=payload.description or "")
            assign_worker_role(conn, worker_id=worker_id, role_id=role_id)
            _receipt(
                conn,
                auth=auth,
                action="operations.worker_role.assign",
                resource_type="worker",
                resource_id=worker_id,
                request=request,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"workerId": worker_id, "roleId": role_id}


@router.post("/operations/workers/{worker_id}/compliances", summary="Assign a compliance to a worker for operational planning")
def post_operations_worker_compliance(
    request: Request,
    worker_id: int = Path(..., description="Worker identifier"),
    payload: WorkerComplianceAssignmentRequest = ...,
    auth: ApiAuthContext = Depends(require_operations_write),
) -> dict[str, Any]:
    if payload.complianceId is None and not (payload.complianceName or "").strip():
        raise HTTPException(status_code=400, detail="Provide complianceId or complianceName")
    with connection_scope(_current_db_path()) as conn:
        try:
            compliance_id = int(payload.complianceId) if payload.complianceId is not None else ensure_worker_compliance(conn, name=str(payload.complianceName).strip(), description=payload.description or "")
            assign_worker_compliance(conn, worker_id=worker_id, compliance_id=compliance_id, expiry_date=payload.expiryDate)
            _receipt(
                conn,
                auth=auth,
                action="operations.worker_compliance.assign",
                resource_type="worker",
                resource_id=worker_id,
                request=request,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"workerId": worker_id, "complianceId": compliance_id, "expiryDate": payload.expiryDate}


@router.get("/operations/inventory/segments", response_model=List[SegmentInventoryCoordinationResponse], summary="List segment-linked inventory and supplier coordination")
def get_operations_inventory_segments(
    job_id: Optional[int] = Query(default=None, description="Optional job filter"),
) -> List[SegmentInventoryCoordinationResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_segment_inventory_coordination(conn, job_id=job_id)
    return [SegmentInventoryCoordinationResponse(**row) for row in rows]


@router.get("/operations/jobs/board", response_model=List[JobOperationsBoardResponse], summary="List job-centric operational board rows")
def get_operations_jobs_board(
    job_id: Optional[int] = Query(default=None, description="Optional job filter"),
) -> List[JobOperationsBoardResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_job_operations_board(conn, job_id=job_id)
    return [JobOperationsBoardResponse(**row) for row in rows]


@router.get("/operations/cutover/workflows", response_model=List[OperationsCutoverWorkflowResponse], summary="List spreadsheet-to-native workflow cutover status")
def get_operations_cutover_workflows() -> List[OperationsCutoverWorkflowResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_operations_cutover_rollout(conn)
    return [OperationsCutoverWorkflowResponse(**row) for row in rows]


@router.put("/operations/cutover/workflows/{workflow_key}", response_model=OperationsCutoverWorkflowResponse, summary="Update workflow cutover status and fallback rules")
def put_operations_cutover_workflow(
    request: Request,
    workflow_key: str = Path(..., description="Workflow identifier"),
    payload: OperationsCutoverWorkflowUpdateRequest = ...,
    auth: ApiAuthContext = Depends(require_operations_cutover_write),
) -> OperationsCutoverWorkflowResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            upsert_operations_cutover_workflow(
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
        record_api_write_receipt(
            conn,
            auth=auth,
            action="operations_cutover_workflow_updated",
            resource_type="operations_cutover_workflow",
            resource_id=workflow_key,
            request=request,
        )
        row = next(item for item in list_operations_cutover_rollout(conn) if item["workflowKey"] == workflow_key)
    return OperationsCutoverWorkflowResponse(**row)


@router.post("/operations/cutover/workflows/{workflow_key}/request-promotion", response_model=OperationsCutoverWorkflowResponse, summary="Request the next guarded cutover promotion")
def post_operations_cutover_request_promotion(
    request: Request,
    workflow_key: str = Path(..., description="Workflow identifier"),
    payload: OperationsCutoverPromotionRequest = ...,
    auth: ApiAuthContext = Depends(require_operations_cutover_write),
) -> OperationsCutoverWorkflowResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = request_operations_cutover_promotion(conn, workflow_key=workflow_key, actor=auth.actor, note=payload.note, target_status=payload.targetStatus)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        record_api_write_receipt(
            conn,
            auth=auth,
            action="operations_cutover_promotion_requested",
            resource_type="operations_cutover_workflow",
            resource_id=workflow_key,
            request=request,
        )
    return OperationsCutoverWorkflowResponse(**row)


@router.post("/operations/cutover/workflows/{workflow_key}/approve-promotion", response_model=OperationsCutoverWorkflowResponse, summary="Approve the next guarded cutover promotion")
def post_operations_cutover_approve_promotion(
    request: Request,
    workflow_key: str = Path(..., description="Workflow identifier"),
    payload: OperationsCutoverPromotionDecisionRequest = ...,
    auth: ApiAuthContext = Depends(require_operations_cutover_approval),
) -> OperationsCutoverWorkflowResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = approve_operations_cutover_promotion(conn, workflow_key=workflow_key, actor=auth.actor, note=payload.note, target_status=payload.targetStatus)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        record_api_write_receipt(
            conn,
            auth=auth,
            action="operations_cutover_promotion_approved",
            resource_type="operations_cutover_workflow",
            resource_id=workflow_key,
            request=request,
        )
    return OperationsCutoverWorkflowResponse(**row)


@router.post("/operations/cutover/workflows/{workflow_key}/reject-promotion", response_model=OperationsCutoverWorkflowResponse, summary="Reject the next guarded cutover promotion")
def post_operations_cutover_reject_promotion(
    request: Request,
    workflow_key: str = Path(..., description="Workflow identifier"),
    payload: OperationsCutoverPromotionDecisionRequest = ...,
    auth: ApiAuthContext = Depends(require_operations_cutover_approval),
) -> OperationsCutoverWorkflowResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = reject_operations_cutover_promotion(conn, workflow_key=workflow_key, actor=auth.actor, note=payload.note or "", target_status=payload.targetStatus)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        record_api_write_receipt(
            conn,
            auth=auth,
            action="operations_cutover_promotion_rejected",
            resource_type="operations_cutover_workflow",
            resource_id=workflow_key,
            request=request,
        )
    return OperationsCutoverWorkflowResponse(**row)


@router.post("/operations/cutover/workflows/{workflow_key}/apply-recommendation", response_model=OperationsCutoverWorkflowResponse, summary="Apply the guarded recommended cutover transition")
def post_operations_cutover_apply_recommendation(
    request: Request,
    workflow_key: str = Path(..., description="Workflow identifier"),
    payload: OperationsCutoverTransitionRequest = ...,
    auth: ApiAuthContext = Depends(require_operations_cutover_write),
) -> OperationsCutoverWorkflowResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = apply_operations_cutover_recommendation(conn, workflow_key=workflow_key, actor=auth.actor, note=payload.note)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        record_api_write_receipt(
            conn,
            auth=auth,
            action="operations_cutover_recommendation_applied",
            resource_type="operations_cutover_workflow",
            resource_id=workflow_key,
            request=request,
        )
    return OperationsCutoverWorkflowResponse(**row)


@router.get("/operations/cutover/events", response_model=List[OperationsCutoverEventResponse], summary="List spreadsheet cutover events")
def get_operations_cutover_events(
    workflow_key: Optional[str] = Query(default=None, description="Optional workflow filter"),
    event_type: Optional[str] = Query(default=None, description="Optional event-type filter"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum events to return"),
) -> List[OperationsCutoverEventResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_operations_cutover_events(conn, workflow_key=workflow_key, event_type=event_type, limit=limit)
    return [OperationsCutoverEventResponse(**row) for row in rows]


@router.post("/operations/cutover/workflows/{workflow_key}/events", response_model=OperationsCutoverEventResponse, summary="Record a spreadsheet cutover event")
def post_operations_cutover_event(
    request: Request,
    workflow_key: str = Path(..., description="Workflow identifier"),
    payload: OperationsCutoverEventRequest = ...,
    auth: ApiAuthContext = Depends(require_operations_cutover_write),
) -> OperationsCutoverEventResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = record_operations_cutover_event(
                conn,
                workflow_key=workflow_key,
                event_type=payload.eventType,
                actor=auth.actor,
                note=payload.note,
                event_value=payload.eventValue,
                created_at=payload.createdAt,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        record_api_write_receipt(
            conn,
            auth=auth,
            action=f"operations_cutover_event:{payload.eventType}",
            resource_type="operations_cutover_workflow",
            resource_id=workflow_key,
            request=request,
        )
    return OperationsCutoverEventResponse(**row)


@router.post("/operations/inventory/segments/{segment_id}/allocate", summary="Allocate inventory to a planned job segment")
def post_operations_inventory_allocation(
    request: Request,
    segment_id: int = Path(..., description="Segment identifier"),
    payload: SegmentInventoryAllocationRequest = ...,
    auth: ApiAuthContext = Depends(require_operations_write),
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
            _receipt(
                conn,
                auth=auth,
                action="operations.inventory.allocate",
                resource_type="job_segment_shipment",
                resource_id=shipment["id"],
                request=request,
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
