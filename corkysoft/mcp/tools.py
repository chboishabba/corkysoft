from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date
from typing import Any, Mapping

import pandas as pd

from analytics.db.connection import connection_scope
from analytics.operations_assignment import list_operational_share_opportunities
from analytics.operations_diary import build_operations_diary, list_observer_outbox_events
from analytics.price_distribution import load_historical_jobs
from analytics.quote_guidance import build_quote_benchmark_overlay
from analytics.margin_regression import (
    build_corridor_margin_preview,
    summarise_corridor_margin_model,
    summarise_corridor_margin_validation,
)
from .contracts import JsonDict, ToolInputError, ToolSpec


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolInputError(f"{key} must be a string", details={"field": key})
    text = value.strip()
    return text or None


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = _optional_str(payload, key)
    if value is None:
        raise ToolInputError(f"{key} is required", details={"field": key})
    return value


def _optional_int(payload: Mapping[str, Any], key: str, *, minimum: int | None = None) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ToolInputError(f"{key} must be an integer", details={"field": key})
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ToolInputError(f"{key} must be an integer", details={"field": key}) from exc
    if minimum is not None and numeric < minimum:
        raise ToolInputError(
            f"{key} must be >= {minimum}",
            details={"field": key, "minimum": minimum},
        )
    return numeric


def _optional_float(payload: Mapping[str, Any], key: str, *, minimum: float | None = None) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ToolInputError(f"{key} must be numeric", details={"field": key})
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ToolInputError(f"{key} must be numeric", details={"field": key}) from exc
    if minimum is not None and numeric < minimum:
        raise ToolInputError(
            f"{key} must be >= {minimum}",
            details={"field": key, "minimum": minimum},
        )
    return numeric


def _optional_bool(payload: Mapping[str, Any], key: str, *, default: bool = False) -> bool:
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ToolInputError(f"{key} must be a boolean", details={"field": key})


def _optional_date(payload: Mapping[str, Any], key: str) -> pd.Timestamp | None:
    value = _optional_str(payload, key)
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ToolInputError(f"{key} must be an ISO date", details={"field": key})
    return parsed


def _db_path(payload: Mapping[str, Any]) -> str | None:
    path = _optional_str(payload, "db_path")
    return path or None


def _jsonify(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonify(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return []
        records = value.to_dict(orient="records")
        return [_jsonify(item) for item in records]
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if value is pd.NA:
        return None
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _jsonify(value.item())
        except Exception:
            pass
    return value


def profitability_summary_tool(payload: Mapping[str, Any]) -> JsonDict:
    start_date = _optional_date(payload, "start_date")
    end_date = _optional_date(payload, "end_date")
    corridor = _optional_str(payload, "corridor")
    postcode_prefix = _optional_str(payload, "postcode_prefix")
    target_column = _optional_str(payload, "target_column") or "margin_per_m3"
    min_corridor_jobs = _optional_int(payload, "min_corridor_jobs", minimum=1) or 2
    max_corridors = _optional_int(payload, "max_corridors", minimum=1) or 8
    preview_max_corridors = _optional_int(payload, "preview_max_corridors", minimum=1) or 4
    preview_season = _optional_str(payload, "preview_season")

    with connection_scope(_db_path(payload)) as conn:
        df, _ = load_historical_jobs(
            conn,
            start_date=start_date,
            end_date=end_date,
            corridor=corridor,
            postcode_prefix=postcode_prefix,
        )

    if df.empty:
        return {
            "jobCount": 0,
            "targetColumn": target_column,
            "filters": {
                "startDate": start_date.isoformat() if start_date is not None else None,
                "endDate": end_date.isoformat() if end_date is not None else None,
                "corridor": corridor,
                "postcodePrefix": postcode_prefix,
            },
            "model": None,
            "validation": None,
            "preview": [],
        }

    model = summarise_corridor_margin_model(
        df,
        target_column=target_column,
        min_corridor_jobs=min_corridor_jobs,
        max_corridors=max_corridors,
    )
    validation = summarise_corridor_margin_validation(
        df,
        target_column=target_column,
        min_corridor_jobs=min_corridor_jobs,
        max_corridors=max_corridors,
    )
    preview = build_corridor_margin_preview(
        model,
        season=preview_season,
        max_corridors=preview_max_corridors,
    )
    top_corridors = sorted(
        model.corridor_job_counts.items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )[:preview_max_corridors]
    return {
        "jobCount": int(len(df)),
        "targetColumn": target_column,
        "filters": {
            "startDate": start_date.isoformat() if start_date is not None else None,
            "endDate": end_date.isoformat() if end_date is not None else None,
            "corridor": corridor,
            "postcodePrefix": postcode_prefix,
        },
        "model": {
            "fittedJobCount": int(model.fitted_job_count),
            "rSquared": float(model.r_squared),
            "rmse": float(model.rmse),
            "baselineRSquared": float(model.baseline_r_squared),
            "baselineRmse": float(model.baseline_rmse),
            "distanceCoeffPer100km": float(model.distance_coeff_per_100km),
            "baselineSeason": model.baseline_season,
            "baselineCorridor": model.baseline_corridor,
            "seasonalEffects": _jsonify(model.seasonal_effects),
            "corridorEffects": _jsonify({name: model.corridor_effects.get(name, 0.0) for name, _ in top_corridors}),
            "corridorJobCounts": _jsonify({name: count for name, count in top_corridors}),
        },
        "validation": {
            "trainJobCount": int(validation.train_job_count),
            "holdoutJobCount": int(validation.holdout_job_count),
            "holdoutRmse": float(validation.holdout_rmse),
            "holdoutMae": float(validation.holdout_mae),
            "baselineHoldoutRmse": float(validation.baseline_holdout_rmse),
            "baselineHoldoutMae": float(validation.baseline_holdout_mae),
            "rmseImprovement": float(validation.rmse_improvement),
            "maeImprovement": float(validation.mae_improvement),
            "novelHoldoutCorridorCount": int(validation.novel_holdout_corridor_count),
            "trustedCorridorCount": int(validation.trusted_corridor_count),
            "trustLabel": validation.trust_label,
        },
        "preview": _jsonify(preview),
    }


def dispatch_recommendations_tool(payload: Mapping[str, Any]) -> JsonDict:
    job_id = _optional_int(payload, "job_id", minimum=1)
    limit = _optional_int(payload, "limit", minimum=1) or 25

    with connection_scope(_db_path(payload)) as conn:
        opportunities = list_operational_share_opportunities(conn, job_id=job_id)

    counts_by_type: dict[str, int] = {}
    counts_by_response: dict[str, int] = {}
    for row in opportunities:
        opportunity_type = str(row.get("opportunityType") or "unknown")
        counts_by_type[opportunity_type] = counts_by_type.get(opportunity_type, 0) + 1
        response = str(row.get("utilizationResponse") or "unknown")
        counts_by_response[response] = counts_by_response.get(response, 0) + 1

    return {
        "jobId": job_id,
        "totalCount": len(opportunities),
        "countsByType": counts_by_type,
        "countsByResponse": counts_by_response,
        "opportunities": _jsonify(opportunities[:limit]),
    }


def operations_diary_summary_tool(payload: Mapping[str, Any]) -> JsonDict:
    anchor_date = _optional_str(payload, "anchor_date") or date.today().isoformat()
    view_mode = _optional_str(payload, "view_mode") or "day"
    if view_mode not in {"day", "week"}:
        raise ToolInputError(
            "view_mode must be 'day' or 'week'",
            details={"field": "view_mode"},
        )
    focus_job_id = _optional_int(payload, "focus_job_id", minimum=1)
    include_observer = _optional_bool(payload, "include_observer", default=False)
    observer_limit = _optional_int(payload, "observer_limit", minimum=1) or 10

    with connection_scope(_db_path(payload)) as conn:
        diary = build_operations_diary(
            conn,
            anchor_date=anchor_date,
            view_mode=view_mode,
            focus_job_id=focus_job_id,
        )
        observer_events = (
            list_observer_outbox_events(conn, limit=observer_limit, job_id=focus_job_id)
            if include_observer
            else []
        )

    jobs = [
        {
            "jobId": int(row["jobId"]),
            "jobClient": row.get("jobClient"),
            "jobOrigin": row.get("jobOrigin"),
            "jobDestination": row.get("jobDestination"),
            "jobStatus": row.get("jobStatus"),
            "taskCount": int(row.get("taskCount") or 0),
            "invoiceStatus": row.get("invoiceStatus"),
            "billStatus": row.get("billStatus"),
            "plannedStart": row.get("plannedStart"),
            "plannedEnd": row.get("plannedEnd"),
        }
        for row in diary["jobs"]
    ]

    return {
        "anchorDate": diary["anchorDate"],
        "viewMode": diary["viewMode"],
        "startDate": diary["startDate"],
        "endDate": diary["endDate"],
        "focusJobId": focus_job_id,
        "summary": _jsonify(diary["summary"]),
        "jobs": jobs,
        "observerEvents": _jsonify(observer_events),
    }


def quote_guidance_preview_tool(payload: Mapping[str, Any]) -> JsonDict:
    origin_resolved = _required_str(payload, "origin_resolved")
    destination_resolved = _required_str(payload, "destination_resolved")
    cubic_m = _optional_float(payload, "cubic_m", minimum=0.0)
    current_quote_total = _optional_float(payload, "current_quote_total", minimum=0.0)
    if cubic_m is None:
        raise ToolInputError("cubic_m is required", details={"field": "cubic_m"})
    if current_quote_total is None:
        raise ToolInputError(
            "current_quote_total is required",
            details={"field": "current_quote_total"},
        )
    origin_postcode = _optional_str(payload, "origin_postcode")
    destination_postcode = _optional_str(payload, "destination_postcode")
    spare_capacity_signal = payload.get("spare_capacity_signal")
    if spare_capacity_signal is not None and not isinstance(spare_capacity_signal, dict):
        raise ToolInputError(
            "spare_capacity_signal must be an object",
            details={"field": "spare_capacity_signal"},
        )

    with connection_scope(_db_path(payload)) as conn:
        overlay = build_quote_benchmark_overlay(
            conn,
            origin_resolved=origin_resolved,
            destination_resolved=destination_resolved,
            origin_postcode=origin_postcode,
            destination_postcode=destination_postcode,
            cubic_m=float(cubic_m),
            current_quote_total=float(current_quote_total),
            spare_capacity_signal=spare_capacity_signal,
        )

    return _jsonify(asdict(overlay))


def get_corkysoft_tools() -> list[tuple[ToolSpec, Any]]:
    return [
        (
            ToolSpec(
                name="corkysoft.profitability_summary",
                title="Profitability Summary",
                description="Read-only profitability and corridor-model summary over historical jobs.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "db_path": {"type": "string"},
                        "start_date": {"type": "string", "format": "date"},
                        "end_date": {"type": "string", "format": "date"},
                        "corridor": {"type": "string"},
                        "postcode_prefix": {"type": "string"},
                        "target_column": {"type": "string"},
                        "min_corridor_jobs": {"type": "integer", "minimum": 1},
                        "max_corridors": {"type": "integer", "minimum": 1},
                        "preview_max_corridors": {"type": "integer", "minimum": 1},
                        "preview_season": {"type": "string"},
                    },
                },
            ),
            profitability_summary_tool,
        ),
        (
            ToolSpec(
                name="corkysoft.dispatch_recommendations",
                title="Dispatch Recommendations",
                description="Read-only operational share and utilisation recommendation feed.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "db_path": {"type": "string"},
                        "job_id": {"type": "integer", "minimum": 1},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                },
            ),
            dispatch_recommendations_tool,
        ),
        (
            ToolSpec(
                name="corkysoft.operations_diary_summary",
                title="Operations Diary Summary",
                description="Read-only day/week operations-diary summary with optional observer-outbox rows.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "db_path": {"type": "string"},
                        "anchor_date": {"type": "string", "format": "date"},
                        "view_mode": {"type": "string", "enum": ["day", "week"]},
                        "focus_job_id": {"type": "integer", "minimum": 1},
                        "include_observer": {"type": "boolean"},
                        "observer_limit": {"type": "integer", "minimum": 1},
                    },
                },
            ),
            operations_diary_summary_tool,
        ),
        (
            ToolSpec(
                name="corkysoft.quote_guidance_preview",
                title="Quote Guidance Preview",
                description="Read-only benchmark and quote-guidance preview for a proposed quote.",
                input_schema={
                    "type": "object",
                    "required": [
                        "origin_resolved",
                        "destination_resolved",
                        "cubic_m",
                        "current_quote_total",
                    ],
                    "properties": {
                        "db_path": {"type": "string"},
                        "origin_resolved": {"type": "string"},
                        "destination_resolved": {"type": "string"},
                        "origin_postcode": {"type": "string"},
                        "destination_postcode": {"type": "string"},
                        "cubic_m": {"type": "number", "minimum": 0},
                        "current_quote_total": {"type": "number", "minimum": 0},
                        "spare_capacity_signal": {"type": "object"},
                    },
                },
            ),
            quote_guidance_preview_tool,
        ),
    ]
