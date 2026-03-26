"""Labor analytics and worker absence API routes."""

from __future__ import annotations

import os
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from analytics.db import (
    ABSENCE_RECORD_STATUSES,
    ABSENCE_RECORD_TYPES,
    create_worker_absence_record,
    list_worker_absence_records,
)
from analytics.db_connection import connection_scope
from analytics.labor_analytics import build_payroll_labor_analytics
from analytics.operations_assignment import (
    list_labor_reconciliation,
    list_planned_labor_assignments,
)

router = APIRouter()


def _current_db_path() -> str:
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


class PlannedLaborAssignmentResponse(BaseModel):
    segmentId: int
    jobId: int
    jobClient: Optional[str] = None
    segmentSequence: int
    workerId: Optional[int] = None
    workerName: Optional[str] = None
    roleId: Optional[int] = None
    truckIds: List[str] = Field(default_factory=list)
    truckNames: List[str] = Field(default_factory=list)
    plannedStart: Optional[str] = None
    plannedEnd: Optional[str] = None
    fromLocation: Optional[str] = None
    toLocation: Optional[str] = None
    assignmentStatus: Optional[str] = None


class LaborReconciliationResponse(BaseModel):
    status: str
    workerId: Optional[int] = None
    workerName: Optional[str] = None
    truckIds: List[str] = Field(default_factory=list)
    jobId: Optional[int] = None
    segmentId: Optional[int] = None
    plannedStart: Optional[str] = None
    plannedEnd: Optional[str] = None
    shiftDate: Optional[str] = None
    source: Optional[str] = None


class LaborAnalyticsSummaryResponse(BaseModel):
    plannedHours: float
    plannedExposure: float
    importedHours: float
    importedCost: float
    reviewedActualCost: float
    workerCount: int
    confidenceScore: int
    confidenceLabel: str
    absenceModelStatus: str
    absenceRecordCount: int
    confirmedAbsenceCount: int
    overtimeDailyHours: float


class PayForecastRowResponse(BaseModel):
    workerName: str
    plannedHours: float
    plannedExposure: float
    importedHours: float
    importedCost: float
    reviewedActualCost: float
    acceptedEventCount: int
    hourlyRateBasis: float
    absenceDays: float
    absenceHours: float


class ExportReadyLaborSummaryRowResponse(BaseModel):
    workerName: str
    dateRangeStart: str
    dateRangeEnd: str
    plannedExposure: float
    importedCost: float
    reviewedActualCost: float
    importedHours: float
    overtimeHours: float
    absenceDays: float
    absenceHours: float
    acceptedEventCount: int
    pendingReviewCount: int
    hourlyRateBasis: float
    exportReady: bool


class OvertimeDistributionRowResponse(BaseModel):
    workerName: str
    date: str
    totalHours: float
    overtimeHours: float
    totalCost: float
    shiftCount: int


class LaborConfidenceSummaryResponse(BaseModel):
    pendingReviewCount: int
    acceptedEventCount: int
    rejectedEventCount: int
    duplicateEventCount: int
    missingPriorClockOnCount: int
    plannedOnlyCount: int
    importedOnlyCount: int
    matchedPlanImportCount: int
    acceptedUnmatchedCount: int
    confidenceScore: int
    confidenceLabel: str


class LaborCostDriverRowResponse(BaseModel):
    dimension: str
    dimensionValue: str
    totalHours: float
    totalCost: float
    shiftCount: int


class LaborAbsenceSummaryResponse(BaseModel):
    recordCount: int
    confirmedCount: int
    plannedCount: int
    cancelledCount: int
    sickDays: float
    annualLeaveDays: float
    personalLeaveDays: float
    unpaidLeaveDays: float
    carersLeaveDays: float
    otherDays: float


class WorkerAbsenceRecordResponse(BaseModel):
    id: int
    workerId: int
    workerName: str
    startDate: str
    endDate: str
    absenceType: str
    status: str
    hoursPerDay: Optional[float] = None
    note: Optional[str] = None
    source: Optional[str] = None
    recordedBy: Optional[str] = None
    createdAt: str
    updatedAt: str


class WorkerAbsenceRecordRequest(BaseModel):
    workerId: int
    startDate: str
    endDate: Optional[str] = None
    absenceType: str = Field(default="other", description="One of the supported absence/leave types")
    status: str = Field(default="confirmed", description="One of the supported absence statuses")
    hoursPerDay: Optional[float] = None
    note: Optional[str] = None
    source: Optional[str] = None
    recordedBy: Optional[str] = None


@router.get(
    "/operations/labor/roster",
    response_model=List[PlannedLaborAssignmentResponse],
    summary="List native planned labor assignments by date, worker, or truck",
)
def get_operations_labor_roster(
    start_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD lower bound"),
    end_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD upper bound"),
    worker_id: Optional[int] = Query(default=None, description="Optional worker filter"),
    truck_id: Optional[str] = Query(default=None, description="Optional truck filter"),
) -> List[PlannedLaborAssignmentResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_planned_labor_assignments(
            conn,
            start_date=start_date,
            end_date=end_date,
            worker_id=worker_id,
            truck_id=truck_id,
        )
    return [PlannedLaborAssignmentResponse(**row) for row in rows]


@router.get(
    "/operations/labor/reconciliation",
    response_model=List[LaborReconciliationResponse],
    summary="Compare native planned labor with imported VEHICLE_DRIVER shifts",
)
def get_operations_labor_reconciliation(
    start_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD lower bound"),
    end_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD upper bound"),
) -> List[LaborReconciliationResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_labor_reconciliation(
            conn,
            start_date=start_date,
            end_date=end_date,
        )
    return [LaborReconciliationResponse(**row) for row in rows]


def _build_labor_payload(
    *,
    start_date: Optional[str],
    end_date: Optional[str],
    overtime_daily_hours: Optional[float],
) -> dict:
    with connection_scope(_current_db_path()) as conn:
        return build_payroll_labor_analytics(
            conn,
            start_date=start_date,
            end_date=end_date,
            overtime_daily_hours=overtime_daily_hours,
        )


@router.get(
    "/labor-analytics/summary",
    response_model=LaborAnalyticsSummaryResponse,
    summary="Summarize payroll-preparation and labor analytics for a date range",
)
def get_labor_analytics_summary(
    start_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD lower bound"),
    end_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD upper bound"),
    overtime_daily_hours: Optional[float] = Query(
        default=None, description="Optional daily overtime threshold in hours"
    ),
) -> LaborAnalyticsSummaryResponse:
    payload = _build_labor_payload(
        start_date=start_date,
        end_date=end_date,
        overtime_daily_hours=overtime_daily_hours,
    )
    return LaborAnalyticsSummaryResponse(**payload["summary"])


@router.get(
    "/labor-analytics/pay-forecast",
    response_model=List[PayForecastRowResponse],
    summary="List pay-forecast rows by worker for a date range",
)
def get_labor_analytics_pay_forecast(
    start_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD lower bound"),
    end_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD upper bound"),
    overtime_daily_hours: Optional[float] = Query(
        default=None, description="Optional daily overtime threshold in hours"
    ),
) -> List[PayForecastRowResponse]:
    payload = _build_labor_payload(
        start_date=start_date,
        end_date=end_date,
        overtime_daily_hours=overtime_daily_hours,
    )
    return [PayForecastRowResponse(**row) for row in payload["payForecastRows"]]


@router.get(
    "/labor-analytics/export-summary",
    response_model=List[ExportReadyLaborSummaryRowResponse],
    summary="List export-ready labor summary rows for payroll/accounting handoff",
)
def get_labor_analytics_export_summary(
    start_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD lower bound"),
    end_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD upper bound"),
    overtime_daily_hours: Optional[float] = Query(
        default=None, description="Optional daily overtime threshold in hours"
    ),
) -> List[ExportReadyLaborSummaryRowResponse]:
    payload = _build_labor_payload(
        start_date=start_date,
        end_date=end_date,
        overtime_daily_hours=overtime_daily_hours,
    )
    return [
        ExportReadyLaborSummaryRowResponse(**row)
        for row in payload["exportReadyLaborSummaries"]
    ]


@router.get(
    "/labor-analytics/overtime",
    response_model=List[OvertimeDistributionRowResponse],
    summary="List daily overtime distribution rows for a date range",
)
def get_labor_analytics_overtime(
    start_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD lower bound"),
    end_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD upper bound"),
    overtime_daily_hours: Optional[float] = Query(
        default=None, description="Optional daily overtime threshold in hours"
    ),
) -> List[OvertimeDistributionRowResponse]:
    payload = _build_labor_payload(
        start_date=start_date,
        end_date=end_date,
        overtime_daily_hours=overtime_daily_hours,
    )
    return [OvertimeDistributionRowResponse(**row) for row in payload["overtimeRows"]]


@router.get(
    "/labor-analytics/confidence",
    response_model=LaborConfidenceSummaryResponse,
    summary="Summarize payroll-prep confidence and worker-time anomalies for a date range",
)
def get_labor_analytics_confidence(
    start_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD lower bound"),
    end_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD upper bound"),
    overtime_daily_hours: Optional[float] = Query(
        default=None, description="Optional daily overtime threshold in hours"
    ),
) -> LaborConfidenceSummaryResponse:
    payload = _build_labor_payload(
        start_date=start_date,
        end_date=end_date,
        overtime_daily_hours=overtime_daily_hours,
    )
    return LaborConfidenceSummaryResponse(**payload["confidence"])


@router.get(
    "/labor-analytics/absence",
    response_model=LaborAbsenceSummaryResponse,
    summary="Summarize recorded absence and leave rows for a date range",
)
def get_labor_analytics_absence(
    start_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD lower bound"),
    end_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD upper bound"),
    overtime_daily_hours: Optional[float] = Query(
        default=None, description="Optional daily overtime threshold in hours"
    ),
) -> LaborAbsenceSummaryResponse:
    payload = _build_labor_payload(
        start_date=start_date,
        end_date=end_date,
        overtime_daily_hours=overtime_daily_hours,
    )
    return LaborAbsenceSummaryResponse(**payload["absenceSummary"])


@router.get(
    "/labor-analytics/cost-drivers",
    response_model=List[LaborCostDriverRowResponse],
    summary="List labor cost drivers grouped by worker, client, corridor, truck, or job",
)
def get_labor_analytics_cost_drivers(
    dimension: str = Query(
        default="worker",
        description="Grouping dimension: worker, client, corridor, truck, or job",
    ),
    start_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD lower bound"),
    end_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD upper bound"),
    overtime_daily_hours: Optional[float] = Query(
        default=None, description="Optional daily overtime threshold in hours"
    ),
) -> List[LaborCostDriverRowResponse]:
    normalized_dimension = dimension.strip().lower()
    if normalized_dimension not in {"worker", "client", "corridor", "truck", "job"}:
        raise HTTPException(status_code=400, detail="Unsupported labor cost-driver dimension")
    payload = _build_labor_payload(
        start_date=start_date,
        end_date=end_date,
        overtime_daily_hours=overtime_daily_hours,
    )
    return [
        LaborCostDriverRowResponse(**row)
        for row in payload["laborCostDrivers"][normalized_dimension]
    ]


@router.get(
    "/worker-absence/records",
    response_model=List[WorkerAbsenceRecordResponse],
    summary="List worker absence and leave records",
)
def get_worker_absence_records(
    worker_id: Optional[int] = Query(default=None),
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
) -> List[WorkerAbsenceRecordResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_worker_absence_records(
            conn,
            worker_id=worker_id,
            start_date=start_date,
            end_date=end_date,
            status=status,
        )
    return [WorkerAbsenceRecordResponse(**row) for row in rows]


@router.post(
    "/worker-absence/records",
    response_model=WorkerAbsenceRecordResponse,
    dependencies=[Depends(require_internal_api_token)],
    summary="Create a worker absence or leave record",
)
def create_worker_absence(
    payload: WorkerAbsenceRecordRequest,
) -> WorkerAbsenceRecordResponse:
    if payload.absenceType.strip().lower() not in ABSENCE_RECORD_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported absence type")
    if payload.status.strip().lower() not in ABSENCE_RECORD_STATUSES:
        raise HTTPException(status_code=400, detail="Unsupported absence status")
    with connection_scope(_current_db_path()) as conn:
        row = create_worker_absence_record(
            conn,
            worker_id=payload.workerId,
            start_date=payload.startDate,
            end_date=payload.endDate,
            absence_type=payload.absenceType,
            status=payload.status,
            hours_per_day=payload.hoursPerDay,
            note=payload.note,
            source=payload.source,
            recorded_by=payload.recordedBy,
        )
    return WorkerAbsenceRecordResponse(
        **{
            "id": int(row["id"]),
            "workerId": int(row["worker_id"]),
            "workerName": row["worker_name"],
            "startDate": row["start_date"],
            "endDate": row["end_date"],
            "absenceType": row["absence_type"],
            "status": row["status"],
            "hoursPerDay": row["hours_per_day"],
            "note": row["note"],
            "source": row["source"],
            "recordedBy": row["recorded_by"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }
    )
