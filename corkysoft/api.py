"""Minimal REST API surface for Corkysoft integrations."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

from analytics.db import fetch_driver_shifts
from analytics.db_connection import connection_scope
from analytics.kent_ams_import import (
    get_kent_tender_policy_config,
    get_tender_calibration,
    import_kent_ams_records,
    list_kent_override_reason_codes,
    list_kent_tender_override_history,
    list_prioritized_tenders,
    record_kent_tender_override,
    update_kent_tender_policy_config,
    upsert_kent_override_reason_code,
)
from analytics.moveware_import import import_moveware_records


def _current_db_path() -> str:
    """Return the SQLite database path configured for the API."""

    return (
        os.environ.get("CORKYSOFT_DB")
        or os.environ.get("ROUTES_DB")
        or "routes.db"
    )


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


class KentTenderPriorityResponse(BaseModel):
    """Ranked tender entry used for Kent AMS pre-scoring triage."""

    tenderExternalId: str
    jobNumber: str
    clientName: Optional[str] = None
    origin: Optional[str] = None
    destination: Optional[str] = None
    expectedRevenue: Optional[float] = None
    estimatedCost: Optional[float] = None
    estimatedMargin: Optional[float] = None
    estimatedMarginPct: Optional[float] = None
    requiredTrucks: Optional[int] = None
    requiredWorkers: Optional[int] = None
    dueAt: Optional[str] = None
    moveDate: Optional[str] = None
    status: str
    scoreTotal: float
    scoreProfitability: float
    scoreCapacity: float
    scoreUrgency: float
    scoreSeasonality: float
    scoreRouteLocation: float
    scoreSpareCapacity: float
    recommendedAction: str
    updatedAt: Optional[str] = None
    profitRuleMode: str
    absoluteMarginThreshold: float
    marginPercentThreshold: float
    policyMatched: bool
    policyFailReasons: List[str] = Field(default_factory=list)
    lossAlert: bool = False
    overrideableFlags: List[str] = Field(default_factory=list)
    hardBlockFlags: List[str] = Field(default_factory=list)
    freshnessState: str
    confidenceScore: float


class KentTenderPolicyConfigResponse(BaseModel):
    ruleMode: str
    absoluteMarginThreshold: float
    marginPercentThreshold: float
    lossAlertFloor: float
    updatedAt: Optional[str] = None


class KentTenderPolicyConfigUpdateRequest(BaseModel):
    ruleMode: str = Field(..., description="ABS_ONLY, PCT_ONLY, EITHER, or BOTH")
    absoluteMarginThreshold: float = Field(..., description="Minimum expected margin amount in AUD")
    marginPercentThreshold: float = Field(..., description="Minimum expected margin percentage")
    lossAlertFloor: float = Field(default=0.0, description="Alert floor for clearly loss-making tenders")


class KentTenderOverrideReasonCodeResponse(BaseModel):
    code: str
    label: str
    description: Optional[str] = None
    active: bool
    systemSeeded: bool
    updatedAt: Optional[str] = None


class KentTenderOverrideReasonCodeUpsertRequest(BaseModel):
    code: str
    label: str
    description: Optional[str] = None
    active: bool = True


class KentTenderOverrideRequest(BaseModel):
    action: str = Field(..., description="Operator action taken for the tender")
    operatorId: str = Field(..., description="Operator identifier recorded for the override")
    reasonCode: str = Field(..., description="Configured reason code for the override")
    note: Optional[str] = Field(default=None, description="Optional free-text operator note")


class KentTenderOverrideResponse(BaseModel):
    id: int
    tenderExternalId: str
    action: str
    operatorId: str
    reasonCode: str
    note: Optional[str] = None
    overrideableFlags: List[str] = Field(default_factory=list)
    hardBlockFlags: List[str] = Field(default_factory=list)
    policyMatched: bool
    policyFailReasons: List[str] = Field(default_factory=list)
    lossAlert: bool
    createdAt: str


class KentTenderCalibrationBandResponse(BaseModel):
    """Calibration metrics for one tender score band."""

    scoreBand: str
    tenders: int
    wins: int
    winRate: float
    avgPredictedMargin: Optional[float] = None
    avgRealizedMargin: Optional[float] = None
    meanAbsMarginError: Optional[float] = None


class KentTenderCalibrationSummaryResponse(BaseModel):
    """Top-level summary of tender scoring calibration."""

    lookbackDays: int
    tenders: int
    wins: int
    overallWinRate: float
    avgRealizedMargin: Optional[float] = None
    meanAbsMarginError: Optional[float] = None


class KentTenderCalibrationResponse(BaseModel):
    """Full calibration payload containing summary and per-band metrics."""

    summary: KentTenderCalibrationSummaryResponse
    bands: List[KentTenderCalibrationBandResponse]


app = FastAPI(title="Corkysoft API", version="0.1.0")


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
) -> ImportSummary:
    """Return a summary for an incoming MoveWare import request."""

    with connection_scope(_current_db_path()) as conn:
        import_moveware_records(conn, resource, payload.records, dry_run=payload.dry_run)
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
) -> ImportSummary:
    """Return a summary for an incoming Kent AMS adapter import request."""

    with connection_scope(_current_db_path()) as conn:
        import_kent_ams_records(conn, resource, payload.records, dry_run=payload.dry_run)
    return ImportSummary(
        resource=resource,
        imported=len(payload.records),
        dry_run=payload.dry_run,
    )


@app.get(
    "/kent-ams/tenders/prioritized",
    response_model=List[KentTenderPriorityResponse],
    summary="List ranked Kent AMS tenders for operator focus",
)
def get_prioritized_kent_tenders(
    status: str = Query(
        default="open",
        description="Tender status filter (`open`, `awarded`, `closed`, or `all`)",
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of ranked tenders to return",
    ),
) -> List[KentTenderPriorityResponse]:
    """Return Kent tenders ranked by profitability/capacity/seasonality urgency."""

    with connection_scope(_current_db_path()) as conn:
        rows = list_prioritized_tenders(conn, status=status.strip().lower(), limit=limit)
    return [KentTenderPriorityResponse(**row) for row in rows]


@app.get(
    "/kent-ams/config",
    response_model=KentTenderPolicyConfigResponse,
    summary="Get default Kent tender policy config",
)
def get_kent_tender_policy() -> KentTenderPolicyConfigResponse:
    """Return the stored default Kent tender policy configuration."""

    with connection_scope(_current_db_path()) as conn:
        payload = get_kent_tender_policy_config(conn)
    return KentTenderPolicyConfigResponse(**payload)


@app.put(
    "/kent-ams/config",
    response_model=KentTenderPolicyConfigResponse,
    summary="Update default Kent tender policy config",
)
def put_kent_tender_policy(
    payload: KentTenderPolicyConfigUpdateRequest,
) -> KentTenderPolicyConfigResponse:
    """Persist the default Kent tender policy configuration in global parameters."""

    with connection_scope(_current_db_path()) as conn:
        updated = update_kent_tender_policy_config(
            conn,
            rule_mode=payload.ruleMode,
            absolute_margin_threshold=payload.absoluteMarginThreshold,
            margin_percent_threshold=payload.marginPercentThreshold,
            loss_alert_floor=payload.lossAlertFloor,
        )
    return KentTenderPolicyConfigResponse(**updated)


@app.get(
    "/kent-ams/override-reasons",
    response_model=List[KentTenderOverrideReasonCodeResponse],
    summary="List Kent tender override reason codes",
)
def get_kent_tender_override_reasons() -> List[KentTenderOverrideReasonCodeResponse]:
    """Return seeded and admin-managed override reason codes."""

    with connection_scope(_current_db_path()) as conn:
        payload = list_kent_override_reason_codes(conn)
    return [KentTenderOverrideReasonCodeResponse(**row) for row in payload]


@app.put(
    "/kent-ams/override-reasons/{code}",
    response_model=KentTenderOverrideReasonCodeResponse,
    summary="Create or update a Kent tender override reason code",
)
def put_kent_tender_override_reason(
    code: str = Path(..., description="Reason code identifier"),
    payload: KentTenderOverrideReasonCodeUpsertRequest = ...,
) -> KentTenderOverrideReasonCodeResponse:
    """Upsert an admin-manageable Kent tender override reason code."""

    if code != payload.code:
        raise HTTPException(status_code=400, detail="Path code must match payload code")
    with connection_scope(_current_db_path()) as conn:
        try:
            row = upsert_kent_override_reason_code(
                conn,
                code=payload.code,
                label=payload.label,
                description=payload.description,
                active=payload.active,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return KentTenderOverrideReasonCodeResponse(**row)


@app.post(
    "/kent-ams/tenders/{tender_external_id}/override",
    response_model=KentTenderOverrideResponse,
    summary="Record an operator override for a Kent tender",
)
def post_kent_tender_override(
    tender_external_id: str = Path(..., description="Kent tender external id"),
    payload: KentTenderOverrideRequest = ...,
) -> KentTenderOverrideResponse:
    """Persist an operator override event for a tender."""

    with connection_scope(_current_db_path()) as conn:
        try:
            row = record_kent_tender_override(
                conn,
                tender_external_id=tender_external_id,
                action=payload.action,
                operator_id=payload.operatorId,
                reason_code=payload.reasonCode,
                note=payload.note,
            )
        except ValueError as exc:
            detail = str(exc)
            status_code = 404 if "not found" in detail.lower() else 400
            raise HTTPException(status_code=status_code, detail=detail) from exc
    return KentTenderOverrideResponse(**row)


@app.get(
    "/kent-ams/tenders/{tender_external_id}/overrides",
    response_model=List[KentTenderOverrideResponse],
    summary="List operator override history for a Kent tender",
)
def get_kent_tender_override_history(
    tender_external_id: str = Path(..., description="Kent tender external id"),
) -> List[KentTenderOverrideResponse]:
    """Return override audit history for a Kent tender."""

    with connection_scope(_current_db_path()) as conn:
        rows = list_kent_tender_override_history(conn, tender_external_id=tender_external_id)
    return [KentTenderOverrideResponse(**row) for row in rows]


@app.get(
    "/kent-ams/tenders/calibration",
    response_model=KentTenderCalibrationResponse,
    summary="Get tender scoring calibration metrics",
)
def get_kent_tender_calibration(
    lookback_days: int = Query(
        default=180,
        ge=1,
        le=3650,
        description="Number of trailing days to include in calibration metrics",
    ),
) -> KentTenderCalibrationResponse:
    """Return score-band win-rate and realized-margin calibration metrics."""

    with connection_scope(_current_db_path()) as conn:
        payload = get_tender_calibration(conn, lookback_days=lookback_days)
    return KentTenderCalibrationResponse(**payload)


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
