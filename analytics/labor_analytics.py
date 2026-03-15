"""Derived payroll-preparation and labor analytics helpers."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any, Literal

import pandas as pd

from analytics.db import fetch_driver_shifts, list_worker_absence_records
from analytics.operations_assignment import (
    list_labor_reconciliation,
    list_planned_labor_assignments,
)
from corkysoft.call_ops import list_worker_time_capture_events

OVERTIME_DAILY_HOURS_DEFAULT = float(os.environ.get("CORKYSOFT_OVERTIME_DAILY_HOURS", "8.0"))

CostDriverDimension = Literal["worker", "client", "corridor", "truck", "job"]


def _norm(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def _to_dataframe(rows: list[Any]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, sqlite3.Row):
            normalized.append(dict(row))
        elif isinstance(row, dict):
            normalized.append(dict(row))
        else:
            normalized.append(dict(row))
    return pd.DataFrame(normalized)


def _corridor(origin: Any, destination: Any) -> str:
    origin_text = _norm(origin)
    destination_text = _norm(destination)
    if origin_text and destination_text:
        return f"{origin_text} -> {destination_text}"
    return ""


def _safe_float(value: Any) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_datetime(value: Any) -> pd.Timestamp | pd.NaT:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return parsed
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_localize(None)
    return parsed


def _confidence_label(score: int) -> str:
    if score >= 85:
        return "high"
    if score >= 60:
        return "medium"
    return "low"


def _date_in_selected_range(
    value: Any,
    *,
    start_date: str | None,
    end_date: str | None,
) -> bool:
    if value in (None, ""):
        return False
    parsed = _safe_datetime(value)
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return False
    current_date = parsed.date().isoformat()
    if start_date and current_date < start_date:
        return False
    if end_date and current_date > end_date:
        return False
    return True


def _inclusive_day_count(start_date: Any, end_date: Any) -> float:
    start = pd.to_datetime(start_date, errors="coerce")
    end = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return 0.0
    if getattr(start, "tzinfo", None) is not None:
        start = start.tz_localize(None)
    if getattr(end, "tzinfo", None) is not None:
        end = end.tz_localize(None)
    days = (end.normalize() - start.normalize()).days + 1
    return float(max(days, 0))


def build_payroll_labor_analytics(
    conn: sqlite3.Connection,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    overtime_daily_hours: float | None = None,
) -> dict[str, Any]:
    overtime_threshold = (
        float(overtime_daily_hours)
        if overtime_daily_hours is not None
        else OVERTIME_DAILY_HOURS_DEFAULT
    )

    planned_df = _to_dataframe(
        list_planned_labor_assignments(conn)
    )
    imported_df = _to_dataframe(
        fetch_driver_shifts(conn, start_date=start_date, end_date=end_date)
    )
    reconciliation_df = _to_dataframe(
        list_labor_reconciliation(conn)
    )
    worker_time_df = _to_dataframe(list_worker_time_capture_events(conn, limit=5000))
    absence_df = _to_dataframe(
        list_worker_absence_records(
            conn,
            start_date=start_date,
            end_date=end_date,
            limit=5000,
        )
    )

    if not planned_df.empty:
        planned_df["plannedStart"] = planned_df["plannedStart"].apply(_safe_datetime)
        planned_df["plannedEnd"] = planned_df["plannedEnd"].apply(_safe_datetime)
        planned_df["plannedDate"] = planned_df["plannedStart"].dt.date
        if start_date or end_date:
            planned_df = planned_df[
                planned_df["plannedStart"].apply(
                    lambda value: _date_in_selected_range(
                        value,
                        start_date=start_date,
                        end_date=end_date,
                    )
                )
            ].copy()
        planned_df["plannedHours"] = (
            (planned_df["plannedEnd"] - planned_df["plannedStart"]).dt.total_seconds()
            / 3600.0
        ).fillna(0.0)
        planned_df["workerName"] = planned_df["workerName"].fillna("").astype(str).str.strip()
        planned_df["jobClient"] = planned_df["jobClient"].fillna("").astype(str).str.strip()
        planned_df["corridor"] = planned_df.apply(
            lambda row: _corridor(row.get("fromLocation"), row.get("toLocation")), axis=1
        )
        planned_df["truckKey"] = planned_df["truckIds"].apply(
            lambda values: ", ".join(values) if isinstance(values, list) else _norm(values)
        )

    if not imported_df.empty:
        imported_df["shiftDate"] = pd.to_datetime(imported_df["shift_date"], errors="coerce").dt.date
        imported_df["workerName"] = imported_df["worker_name"].fillna("").astype(str).str.strip()
        imported_df["truckId"] = imported_df["truck_id"].fillna("").astype(str).str.strip()
        imported_df["jobId"] = imported_df["linked_job_id"].apply(_norm)
        imported_df["hours"] = imported_df["hours"].apply(_safe_float)
        imported_df["hourlyRate"] = imported_df["hourly_rate"].apply(_safe_float)
        imported_df["costTotal"] = imported_df["cost_total"].apply(_safe_float)
        imported_df.loc[
            imported_df["costTotal"] == 0.0, "costTotal"
        ] = imported_df["hours"] * imported_df["hourlyRate"]
        imported_df["jobClient"] = imported_df.get("job_client", pd.Series(dtype=str))
        if "jobClient" not in imported_df.columns:
            imported_df["jobClient"] = ""
        imported_df["corridor"] = imported_df.apply(
            lambda row: _corridor(row.get("job_origin"), row.get("job_destination")), axis=1
        )
        imported_df["shiftWindowStart"] = imported_df["shift_window_start"].fillna("").astype(str).str.strip()
        imported_df["shiftWindowEnd"] = imported_df["shift_window_end"].fillna("").astype(str).str.strip()

    if not worker_time_df.empty:
        worker_time_df["workerName"] = worker_time_df["workerName"].fillna("").astype(str).str.strip()
        worker_time_df["effectiveTimestamp"] = worker_time_df["effectiveTimestamp"].apply(_safe_datetime)
        worker_time_df["effectiveDate"] = worker_time_df["effectiveTimestamp"].dt.date
        worker_time_df["truckId"] = worker_time_df["truckId"].apply(_norm)
        worker_time_df["jobId"] = worker_time_df["jobId"].apply(_norm)
        worker_time_df["reviewStatus"] = worker_time_df["reviewStatus"].fillna("").astype(str)
        worker_time_df["anomalyFlags"] = worker_time_df["rawPayload"].apply(
            lambda payload: (payload or {}).get("anomalyFlags", [])
            if isinstance(payload, dict)
            else []
        )
        if start_date or end_date:
            worker_time_df = worker_time_df[
                worker_time_df["effectiveTimestamp"].apply(
                    lambda value: _date_in_selected_range(
                        value,
                        start_date=start_date,
                        end_date=end_date,
                    )
                )
            ].copy()

    if not absence_df.empty:
        absence_df["workerName"] = absence_df["workerName"].fillna("").astype(str).str.strip()
        absence_df["startDate"] = pd.to_datetime(absence_df["startDate"], errors="coerce").dt.date
        absence_df["endDate"] = pd.to_datetime(absence_df["endDate"], errors="coerce").dt.date
        absence_df["days"] = absence_df.apply(
            lambda row: _inclusive_day_count(row.get("startDate"), row.get("endDate")),
            axis=1,
        )
        absence_df["hoursPerDay"] = absence_df["hoursPerDay"].apply(_safe_float)
        absence_df["estimatedHours"] = absence_df.apply(
            lambda row: _safe_float(row.get("hoursPerDay")) * _safe_float(row.get("days"))
            if _safe_float(row.get("hoursPerDay")) > 0
            else 0.0,
            axis=1,
        )
        absence_df["absenceType"] = absence_df["absenceType"].fillna("").astype(str).str.strip()
        absence_df["status"] = absence_df["status"].fillna("").astype(str).str.strip()

    if not reconciliation_df.empty and (start_date or end_date):
        if "shiftDate" in reconciliation_df.columns:
            reconciliation_df = reconciliation_df[
                reconciliation_df["shiftDate"].fillna("").astype(str).apply(
                    lambda value: _date_in_selected_range(
                        value,
                        start_date=start_date,
                        end_date=end_date,
                    )
                )
                | reconciliation_df["plannedStart"].apply(
                    lambda value: _date_in_selected_range(
                        value,
                        start_date=start_date,
                        end_date=end_date,
                    )
                )
            ].copy()

    accepted_events_df = worker_time_df[
        worker_time_df["reviewStatus"] == "accepted"
    ].copy() if not worker_time_df.empty else pd.DataFrame()
    pending_events_df = worker_time_df[
        worker_time_df["reviewStatus"] == "pending_review"
    ].copy() if not worker_time_df.empty else pd.DataFrame()
    rejected_events_df = worker_time_df[
        worker_time_df["reviewStatus"] == "rejected"
    ].copy() if not worker_time_df.empty else pd.DataFrame()

    rate_by_worker: dict[str, float] = {}
    if not imported_df.empty:
        rate_series = (
            imported_df[imported_df["hourlyRate"] > 0]
            .groupby("workerName")["hourlyRate"]
            .mean()
        )
        rate_by_worker = {str(worker): float(rate) for worker, rate in rate_series.items()}

    planned_forecast_rows: list[dict[str, Any]] = []
    if not planned_df.empty:
        for _, row in planned_df.iterrows():
            worker_name = _norm(row.get("workerName"))
            hourly_rate = rate_by_worker.get(worker_name, 0.0)
            planned_hours = _safe_float(row.get("plannedHours"))
            planned_forecast_rows.append(
                {
                    "workerName": worker_name,
                    "date": str(row.get("plannedDate") or ""),
                    "jobId": _norm(row.get("jobId")),
                    "jobClient": _norm(row.get("jobClient")),
                    "corridor": _norm(row.get("corridor")),
                    "truckKey": _norm(row.get("truckKey")),
                    "plannedHours": planned_hours,
                    "avgHourlyRate": hourly_rate,
                    "plannedExposure": planned_hours * hourly_rate,
                }
            )
    planned_forecast_df = pd.DataFrame(planned_forecast_rows)

    matched_import_indexes: set[int] = set()
    if not imported_df.empty and not accepted_events_df.empty:
        for event_index, event_row in accepted_events_df.iterrows():
            mask = (
                imported_df["shiftDate"].astype(str) == str(event_row.get("effectiveDate") or "")
            ) & (imported_df["workerName"] == _norm(event_row.get("workerName")))
            candidates = imported_df[mask]
            if candidates.empty:
                continue
            preferred = candidates
            event_truck = _norm(event_row.get("truckId"))
            event_job = _norm(event_row.get("jobId"))
            if event_truck:
                truck_matches = preferred[preferred["truckId"] == event_truck]
                if not truck_matches.empty:
                    preferred = truck_matches
            if event_job:
                job_matches = preferred[preferred["jobId"] == event_job]
                if not job_matches.empty:
                    preferred = job_matches
            matched_import_indexes.add(int(preferred.index[0]))

    imported_df["reviewMatched"] = imported_df.index.isin(matched_import_indexes) if not imported_df.empty else False

    pay_forecast_rows: list[dict[str, Any]] = []
    worker_names = sorted(
        {
            *([] if planned_forecast_df.empty else planned_forecast_df["workerName"].dropna().tolist()),
            *([] if imported_df.empty else imported_df["workerName"].dropna().tolist()),
            *([] if accepted_events_df.empty else accepted_events_df["workerName"].dropna().tolist()),
            *([] if absence_df.empty else absence_df["workerName"].dropna().tolist()),
        }
    )
    for worker_name in worker_names:
        planned_subset = (
            planned_forecast_df[planned_forecast_df["workerName"] == worker_name]
            if not planned_forecast_df.empty
            else pd.DataFrame()
        )
        imported_subset = (
            imported_df[imported_df["workerName"] == worker_name]
            if not imported_df.empty
            else pd.DataFrame()
        )
        accepted_subset = (
            accepted_events_df[accepted_events_df["workerName"] == worker_name]
            if not accepted_events_df.empty
            else pd.DataFrame()
        )
        absence_subset = (
            absence_df[
                (absence_df["workerName"] == worker_name)
                & (absence_df["status"].isin(["planned", "confirmed"]))
            ]
            if not absence_df.empty
            else pd.DataFrame()
        )
        pay_forecast_rows.append(
            {
                "workerName": worker_name,
                "plannedHours": float(planned_subset["plannedHours"].sum()) if not planned_subset.empty else 0.0,
                "plannedExposure": float(planned_subset["plannedExposure"].sum()) if not planned_subset.empty else 0.0,
                "importedHours": float(imported_subset["hours"].sum()) if not imported_subset.empty else 0.0,
                "importedCost": float(imported_subset["costTotal"].sum()) if not imported_subset.empty else 0.0,
                "reviewedActualCost": float(imported_subset[imported_subset["reviewMatched"]]["costTotal"].sum()) if not imported_subset.empty else 0.0,
                "acceptedEventCount": int(len(accepted_subset.index)) if not accepted_subset.empty else 0,
                "hourlyRateBasis": float(imported_subset["hourlyRate"].mean()) if not imported_subset.empty else 0.0,
                "absenceDays": float(absence_subset["days"].sum()) if not absence_subset.empty else 0.0,
                "absenceHours": float(absence_subset["estimatedHours"].sum()) if not absence_subset.empty else 0.0,
            }
        )

    overtime_rows: list[dict[str, Any]] = []
    if not imported_df.empty:
        grouped = imported_df.groupby(["workerName", "shiftDate"], dropna=False).agg(
            totalHours=("hours", "sum"),
            totalCost=("costTotal", "sum"),
            shiftCount=("id", "count"),
        )
        for (worker_name, shift_date), row in grouped.iterrows():
            total_hours = float(row["totalHours"])
            overtime_hours = max(0.0, total_hours - overtime_threshold)
            overtime_rows.append(
                {
                    "workerName": _norm(worker_name),
                    "date": str(shift_date or ""),
                    "totalHours": total_hours,
                    "overtimeHours": overtime_hours,
                    "totalCost": float(row["totalCost"]),
                    "shiftCount": int(row["shiftCount"]),
                }
            )

    duplicate_count = 0
    missing_prior_clock_on_count = 0
    if not worker_time_df.empty:
        for flags in worker_time_df["anomalyFlags"]:
            if "duplicate_event" in flags:
                duplicate_count += 1
            if "missing_prior_clock_on" in flags:
                missing_prior_clock_on_count += 1

    reconciliation_status_counts = {
        "planned_only": 0,
        "imported_only": 0,
        "matched": 0,
    }
    if not reconciliation_df.empty and "status" in reconciliation_df.columns:
        for key, value in reconciliation_df["status"].value_counts().to_dict().items():
            reconciliation_status_counts[_norm(key)] = int(value)

    accepted_unmatched_count = 0
    if not accepted_events_df.empty:
        accepted_unmatched_count = int(len(accepted_events_df.index)) - int(len(matched_import_indexes))
        accepted_unmatched_count = max(0, accepted_unmatched_count)

    confidence_penalty = (
        len(pending_events_df.index) * 8
        + duplicate_count * 12
        + missing_prior_clock_on_count * 12
        + reconciliation_status_counts.get("planned_only", 0) * 4
        + reconciliation_status_counts.get("imported_only", 0) * 4
        + accepted_unmatched_count * 6
    )
    confidence_score = max(0, 100 - confidence_penalty)

    confirmed_absences_df = (
        absence_df[absence_df["status"] == "confirmed"].copy() if not absence_df.empty else pd.DataFrame()
    )
    planned_absences_df = (
        absence_df[absence_df["status"] == "planned"].copy() if not absence_df.empty else pd.DataFrame()
    )

    absence_summary = {
        "recordCount": int(len(absence_df.index)) if not absence_df.empty else 0,
        "confirmedCount": int(len(confirmed_absences_df.index)) if not confirmed_absences_df.empty else 0,
        "plannedCount": int(len(planned_absences_df.index)) if not planned_absences_df.empty else 0,
        "cancelledCount": int(len(absence_df[absence_df["status"] == "cancelled"].index)) if not absence_df.empty else 0,
        "sickDays": float(
            absence_df[absence_df["absenceType"] == "sick"]["days"].sum()
        ) if not absence_df.empty else 0.0,
        "annualLeaveDays": float(
            absence_df[absence_df["absenceType"] == "annual_leave"]["days"].sum()
        ) if not absence_df.empty else 0.0,
        "personalLeaveDays": float(
            absence_df[absence_df["absenceType"] == "personal_leave"]["days"].sum()
        ) if not absence_df.empty else 0.0,
        "unpaidLeaveDays": float(
            absence_df[absence_df["absenceType"] == "unpaid_leave"]["days"].sum()
        ) if not absence_df.empty else 0.0,
        "carersLeaveDays": float(
            absence_df[absence_df["absenceType"] == "carers_leave"]["days"].sum()
        ) if not absence_df.empty else 0.0,
        "otherDays": float(
            absence_df[absence_df["absenceType"] == "other"]["days"].sum()
        ) if not absence_df.empty else 0.0,
    }

    summary = {
        "plannedHours": float(planned_forecast_df["plannedHours"].sum()) if not planned_forecast_df.empty else 0.0,
        "plannedExposure": float(planned_forecast_df["plannedExposure"].sum()) if not planned_forecast_df.empty else 0.0,
        "importedHours": float(imported_df["hours"].sum()) if not imported_df.empty else 0.0,
        "importedCost": float(imported_df["costTotal"].sum()) if not imported_df.empty else 0.0,
        "reviewedActualCost": float(imported_df[imported_df["reviewMatched"]]["costTotal"].sum()) if not imported_df.empty else 0.0,
        "workerCount": len([name for name in worker_names if name]),
        "confidenceScore": confidence_score,
        "confidenceLabel": _confidence_label(confidence_score),
        "absenceModelStatus": "basic_recorded",
        "absenceRecordCount": absence_summary["recordCount"],
        "confirmedAbsenceCount": absence_summary["confirmedCount"],
        "overtimeDailyHours": overtime_threshold,
    }

    confidence = {
        "pendingReviewCount": int(len(pending_events_df.index)),
        "acceptedEventCount": int(len(accepted_events_df.index)),
        "rejectedEventCount": int(len(rejected_events_df.index)),
        "duplicateEventCount": duplicate_count,
        "missingPriorClockOnCount": missing_prior_clock_on_count,
        "plannedOnlyCount": reconciliation_status_counts.get("planned_only", 0),
        "importedOnlyCount": reconciliation_status_counts.get("imported_only", 0),
        "matchedPlanImportCount": reconciliation_status_counts.get("matched", 0),
        "acceptedUnmatchedCount": accepted_unmatched_count,
        "confidenceScore": confidence_score,
        "confidenceLabel": _confidence_label(confidence_score),
    }

    plan_vs_actual = {
        "plannedOnlyCount": reconciliation_status_counts.get("planned_only", 0),
        "importedOnlyCount": reconciliation_status_counts.get("imported_only", 0),
        "matchedCount": reconciliation_status_counts.get("matched", 0),
        "acceptedMatchedShiftCount": int(len(matched_import_indexes)),
        "acceptedUnmatchedCount": accepted_unmatched_count,
    }

    cost_driver_sources = imported_df.copy() if not imported_df.empty else pd.DataFrame()
    if not cost_driver_sources.empty:
        cost_driver_sources["worker"] = cost_driver_sources["workerName"]
        cost_driver_sources["client"] = cost_driver_sources["jobClient"]
        cost_driver_sources["corridor"] = cost_driver_sources["corridor"]
        cost_driver_sources["truck"] = cost_driver_sources["truckId"]
        cost_driver_sources["job"] = cost_driver_sources["jobId"]

    def _cost_driver_rows(dimension: CostDriverDimension) -> list[dict[str, Any]]:
        if cost_driver_sources.empty:
            return []
        grouped = (
            cost_driver_sources.groupby(dimension, dropna=False)
            .agg(totalHours=("hours", "sum"), totalCost=("costTotal", "sum"), shiftCount=("id", "count"))
            .reset_index()
            .rename(columns={dimension: "dimensionValue"})
            .sort_values(["totalCost", "totalHours"], ascending=[False, False], kind="stable")
        )
        rows: list[dict[str, Any]] = []
        for _, row in grouped.iterrows():
            value = _norm(row.get("dimensionValue")) or "unassigned"
            rows.append(
                {
                    "dimension": dimension,
                    "dimensionValue": value,
                    "totalHours": float(row.get("totalHours") or 0.0),
                    "totalCost": float(row.get("totalCost") or 0.0),
                    "shiftCount": int(row.get("shiftCount") or 0),
                }
            )
        return rows

    export_ready_rows: list[dict[str, Any]] = []
    for row in pay_forecast_rows:
        worker_name = _norm(row.get("workerName"))
        overtime_hours = 0.0
        if overtime_rows:
            overtime_hours = sum(
                _safe_float(item.get("overtimeHours"))
                for item in overtime_rows
                if _norm(item.get("workerName")) == worker_name
            )
        worker_pending_reviews = 0
        if not pending_events_df.empty:
            worker_pending_reviews = int(
                len(
                    pending_events_df[
                        pending_events_df["workerName"] == worker_name
                    ].index
                )
            )
        export_ready_rows.append(
            {
                "workerName": worker_name,
                "dateRangeStart": start_date or "",
                "dateRangeEnd": end_date or "",
                "plannedExposure": _safe_float(row.get("plannedExposure")),
                "importedCost": _safe_float(row.get("importedCost")),
                "reviewedActualCost": _safe_float(row.get("reviewedActualCost")),
                "importedHours": _safe_float(row.get("importedHours")),
                "overtimeHours": overtime_hours,
                "absenceDays": _safe_float(row.get("absenceDays")),
                "absenceHours": _safe_float(row.get("absenceHours")),
                "acceptedEventCount": int(row.get("acceptedEventCount") or 0),
                "pendingReviewCount": worker_pending_reviews,
                "hourlyRateBasis": _safe_float(row.get("hourlyRateBasis")),
                "exportReady": worker_pending_reviews == 0,
            }
        )

    return {
        "summary": summary,
        "payForecastRows": pay_forecast_rows,
        "exportReadyLaborSummaries": export_ready_rows,
        "hoursCostDistributionRows": (
            imported_df[
                [
                    "shiftDate",
                    "workerName",
                    "truckId",
                    "jobId",
                    "jobClient",
                    "corridor",
                    "hours",
                    "hourlyRate",
                    "costTotal",
                    "source",
                ]
            ]
            .rename(
                columns={
                    "shiftDate": "date",
                    "truckId": "truckId",
                    "jobId": "jobId",
                    "jobClient": "jobClient",
                    "corridor": "corridor",
                    "hours": "hours",
                    "hourlyRate": "hourlyRate",
                    "costTotal": "costTotal",
                    "source": "source",
                }
            )
            .to_dict("records")
            if not imported_df.empty
            else []
        ),
        "overtimeRows": overtime_rows,
        "planVsActual": plan_vs_actual,
        "confidence": confidence,
        "absenceSummary": absence_summary,
        "absenceRows": absence_df.to_dict("records") if not absence_df.empty else [],
        "laborCostDrivers": {
            "worker": _cost_driver_rows("worker"),
            "client": _cost_driver_rows("client"),
            "corridor": _cost_driver_rows("corridor"),
            "truck": _cost_driver_rows("truck"),
            "job": _cost_driver_rows("job"),
        },
    }


__all__ = ["OVERTIME_DAILY_HOURS_DEFAULT", "build_payroll_labor_analytics"]
