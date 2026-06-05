"""Kent AMS tender API routes."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from analytics.db_connection import connection_scope
from analytics.kent_ams_import import (
    get_kent_tender_policy_config,
    get_tender_calibration,
    list_kent_override_reason_codes,
    list_kent_tender_override_history,
    list_prioritized_tenders,
    record_kent_tender_override,
    update_kent_tender_policy_config,
    upsert_kent_override_reason_code,
)
from corkysoft.api_shared import (
    KENT_WRITE_SCOPE,
    ApiAuthContext,
    _current_db_path,
    record_api_write_receipt,
    require_api_auth_context,
    require_internal_api_read_token,
)

router = APIRouter(dependencies=[Depends(require_internal_api_read_token)])
require_kent_write = require_api_auth_context((KENT_WRITE_SCOPE,))


def _kent_actor(auth: ApiAuthContext, supplied: Optional[str] = None) -> str:
    if auth.legacy and supplied:
        return supplied
    return auth.actor


def _receipt(conn, *, auth: ApiAuthContext, action: str, resource_type: str, resource_id: object, request: Request) -> None:
    record_api_write_receipt(
        conn,
        auth=auth,
        action=action,
        resource_type=resource_type,
        resource_id=str(resource_id),
        request=request,
    )


class KentTenderPriorityResponse(BaseModel):
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
    scoreBand: str
    tenders: int
    wins: int
    winRate: float
    avgPredictedMargin: Optional[float] = None
    avgRealizedMargin: Optional[float] = None
    meanAbsMarginError: Optional[float] = None


class KentTenderCalibrationSummaryResponse(BaseModel):
    lookbackDays: int
    tenders: int
    wins: int
    overallWinRate: float
    avgRealizedMargin: Optional[float] = None
    meanAbsMarginError: Optional[float] = None


class KentTenderCalibrationResponse(BaseModel):
    summary: KentTenderCalibrationSummaryResponse
    bands: List[KentTenderCalibrationBandResponse] = Field(default_factory=list)


@router.get(
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
    with connection_scope(_current_db_path()) as conn:
        rows = list_prioritized_tenders(conn, status=status.strip().lower(), limit=limit)
    return [KentTenderPriorityResponse(**row) for row in rows]


@router.get(
    "/kent-ams/config",
    response_model=KentTenderPolicyConfigResponse,
    summary="Get default Kent tender policy config",
)
def get_kent_tender_policy() -> KentTenderPolicyConfigResponse:
    with connection_scope(_current_db_path()) as conn:
        payload = get_kent_tender_policy_config(conn)
    return KentTenderPolicyConfigResponse(**payload)


@router.put(
    "/kent-ams/config",
    response_model=KentTenderPolicyConfigResponse,
    summary="Update default Kent tender policy config",
)
def put_kent_tender_policy(
    request: Request,
    payload: KentTenderPolicyConfigUpdateRequest,
    auth: ApiAuthContext = Depends(require_kent_write),
) -> KentTenderPolicyConfigResponse:
    with connection_scope(_current_db_path()) as conn:
        updated = update_kent_tender_policy_config(
            conn,
            rule_mode=payload.ruleMode,
            absolute_margin_threshold=payload.absoluteMarginThreshold,
            margin_percent_threshold=payload.marginPercentThreshold,
            loss_alert_floor=payload.lossAlertFloor,
        )
        _receipt(
            conn,
            auth=auth,
            action="kent.policy.update",
            resource_type="kent_policy_config",
            resource_id="default",
            request=request,
        )
    return KentTenderPolicyConfigResponse(**updated)


@router.get(
    "/kent-ams/override-reasons",
    response_model=List[KentTenderOverrideReasonCodeResponse],
    summary="List Kent tender override reason codes",
)
def get_kent_tender_override_reasons() -> List[KentTenderOverrideReasonCodeResponse]:
    with connection_scope(_current_db_path()) as conn:
        payload = list_kent_override_reason_codes(conn)
    return [KentTenderOverrideReasonCodeResponse(**row) for row in payload]


@router.put(
    "/kent-ams/override-reasons/{code}",
    response_model=KentTenderOverrideReasonCodeResponse,
    summary="Create or update a Kent tender override reason code",
)
def put_kent_tender_override_reason(
    request: Request,
    code: str = Path(..., description="Reason code identifier"),
    payload: KentTenderOverrideReasonCodeUpsertRequest = ...,
    auth: ApiAuthContext = Depends(require_kent_write),
) -> KentTenderOverrideReasonCodeResponse:
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
            _receipt(
                conn,
                auth=auth,
                action="kent.override_reason.upsert",
                resource_type="kent_override_reason",
                resource_id=payload.code,
                request=request,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return KentTenderOverrideReasonCodeResponse(**row)


@router.post(
    "/kent-ams/tenders/{tender_external_id}/override",
    response_model=KentTenderOverrideResponse,
    summary="Record an operator override for a Kent tender",
)
def post_kent_tender_override(
    request: Request,
    tender_external_id: str = Path(..., description="Kent tender external id"),
    payload: KentTenderOverrideRequest = ...,
    auth: ApiAuthContext = Depends(require_kent_write),
) -> KentTenderOverrideResponse:
    with connection_scope(_current_db_path()) as conn:
        try:
            row = record_kent_tender_override(
                conn,
                tender_external_id=tender_external_id,
                action=payload.action,
                operator_id=_kent_actor(auth, payload.operatorId),
                reason_code=payload.reasonCode,
                note=payload.note,
            )
            _receipt(
                conn,
                auth=auth,
                action="kent.tender_override.create",
                resource_type="kent_tender_override",
                resource_id=row["id"],
                request=request,
            )
        except ValueError as exc:
            detail = str(exc)
            status_code = 404 if "not found" in detail.lower() else 400
            raise HTTPException(status_code=status_code, detail=detail) from exc
    return KentTenderOverrideResponse(**row)


@router.get(
    "/kent-ams/tenders/{tender_external_id}/overrides",
    response_model=List[KentTenderOverrideResponse],
    summary="List operator override history for a Kent tender",
)
def get_kent_tender_override_history(
    tender_external_id: str = Path(..., description="Kent tender external id"),
) -> List[KentTenderOverrideResponse]:
    with connection_scope(_current_db_path()) as conn:
        rows = list_kent_tender_override_history(conn, tender_external_id=tender_external_id)
    return [KentTenderOverrideResponse(**row) for row in rows]


@router.get(
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
    with connection_scope(_current_db_path()) as conn:
        payload = get_tender_calibration(conn, lookback_days=lookback_days)
    return KentTenderCalibrationResponse(**payload)
