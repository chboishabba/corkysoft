from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import math
from typing import Literal, Sequence

import pandas as pd
from analytics.live_data import load_active_routes, load_truck_positions
from analytics.operations_assignment import list_job_operations_board
from analytics.operations_diary import build_operations_diary
from analytics.adaptive_policy import PROPOSAL_STATUS_PENDING_REVIEW
from analytics.kent_ams_import import KENT_TENDER_RULE_MODE_KEY
from analytics.profitability_analysis import summarise_distribution, summarise_profitability

FreshnessState = Literal["fresh", "stale", "unknown", "scaffold"]
DecisionGrade = Literal["decision_grade", "advisory", "placeholder"]


@dataclass(frozen=True)
class ShellMetricSignal:
    signal_id: str
    label: str
    value: str
    delta: str | None
    direction: str
    source: str
    owner: str
    refresh_cadence: str
    stale_threshold: str
    freshness_state: FreshnessState
    decision_grade: DecisionGrade
    fallback_behavior: str

    def as_card(self) -> dict[str, str]:
        card = {
            "label": self.label,
            "value": self.value,
            "direction": self.direction,
        }
        if self.delta:
            card["delta"] = self.delta
        return card


@dataclass(frozen=True)
class ShellAlertSignal:
    signal_id: str
    title: str
    message: str
    severity: str
    source: str
    owner: str
    refresh_cadence: str
    stale_threshold: str
    freshness_state: FreshnessState
    decision_grade: DecisionGrade
    fallback_behavior: str


@dataclass(frozen=True)
class ShellSignalBundle:
    scope_label: str
    metrics: Sequence[ShellMetricSignal]
    alert: ShellAlertSignal
    owner: str
    source: str
    refresh_cadence: str
    stale_threshold: str
    freshness_state: FreshnessState
    decision_grade: DecisionGrade
    fallback_behavior: str

    def metric_cards(self) -> list[dict[str, str]]:
        return [metric.as_card() for metric in self.metrics]


def build_shell_signal_bundle(scope_label: str) -> ShellSignalBundle:
    normalized = scope_label.strip().lower()
    builder = _SHELL_SIGNAL_BUILDERS.get(normalized, _build_unknown_bundle)
    return builder(scope_label)


def build_network_shell_signal_bundle(conn: sqlite3.Connection) -> ShellSignalBundle:
    scope_label = "Network"
    owner = "Network control"
    source = "analytics.live_data"
    refresh_cadence = "telemetry ingest"
    stale_threshold = "15 minutes"
    fallback_behavior = "downgrade to advisory status and avoid disruption claims when telemetry is stale or absent"

    trucks = load_truck_positions(conn)
    routes = load_active_routes(conn)
    freshest_at = _latest_timestamp(trucks, routes)
    freshness_state = _freshness_state(freshest_at, stale_after=timedelta(minutes=15))

    if trucks.empty and routes.empty:
        return ShellSignalBundle(
            scope_label=scope_label,
            metrics=[
                _contract_metric(
                    signal_id="network_active_nodes",
                    label="Active Nodes",
                    value="0",
                    owner=owner,
                    source=source,
                    refresh_cadence=refresh_cadence,
                    stale_threshold=stale_threshold,
                    freshness_state="unknown",
                    decision_grade="advisory",
                    fallback_behavior=fallback_behavior,
                ),
                _contract_metric(
                    signal_id="network_live_trucks",
                    label="Live Trucks",
                    value="0",
                    owner=owner,
                    source=source,
                    refresh_cadence=refresh_cadence,
                    stale_threshold=stale_threshold,
                    freshness_state="unknown",
                    decision_grade="advisory",
                    fallback_behavior=fallback_behavior,
                ),
                _contract_metric(
                    signal_id="network_congestion_level",
                    label="Congestion Level",
                    value="Unknown",
                    owner=owner,
                    source=source,
                    refresh_cadence=refresh_cadence,
                    stale_threshold=stale_threshold,
                    freshness_state="unknown",
                    decision_grade="advisory",
                    fallback_behavior=fallback_behavior,
                ),
            ],
            alert=ShellAlertSignal(
                signal_id="network_status",
                title="Telemetry Unavailable",
                message="No live telemetry is currently available. Network status is advisory only until route and truck updates resume.",
                severity="warning",
                source=source,
                owner=owner,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state="unknown",
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
            owner=owner,
            source=source,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state="unknown",
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )

    active_nodes = _count_active_nodes(routes)
    live_trucks = len(trucks.index)
    delayed_routes = _count_delayed_routes(routes)
    route_count = len(routes.index)

    if route_count == 0:
        congestion_value = "Low"
        congestion_delta = None
        congestion_direction = "neutral"
    else:
        delayed_share = delayed_routes / route_count
        if delayed_share >= 0.5:
            congestion_value = "High"
            congestion_delta = f"{delayed_routes} delayed"
            congestion_direction = "down"
        elif delayed_share > 0:
            congestion_value = "Moderate"
            congestion_delta = f"{delayed_routes} delayed"
            congestion_direction = "down"
        else:
            congestion_value = "Low"
            congestion_delta = "No delayed routes"
            congestion_direction = "up"

    if freshness_state == "stale":
        alert = ShellAlertSignal(
            signal_id="network_status",
            title="Telemetry Stale",
            message="Live telemetry is older than the network freshness threshold. Treat ETA and route-state outputs as advisory until updates resume.",
            severity="warning",
            source=source,
            owner=owner,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state=freshness_state,
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )
    elif delayed_routes > 0:
        alert = ShellAlertSignal(
            signal_id="network_status",
            title="Delayed Routes Detected",
            message=f"{delayed_routes} active route{'s are' if delayed_routes != 1 else ' is'} currently marked delayed.",
            severity="warning",
            source=source,
            owner=owner,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state=freshness_state,
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )
    else:
        alert = ShellAlertSignal(
            signal_id="network_status",
            title="Network Status",
            message="No delayed active routes are currently reported by telemetry.",
            severity="info",
            source=source,
            owner=owner,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state=freshness_state,
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )

    return ShellSignalBundle(
        scope_label=scope_label,
        metrics=[
            _contract_metric(
                signal_id="network_active_nodes",
                label="Active Nodes",
                value=str(active_nodes),
                delta=f"{route_count} active routes" if route_count else None,
                direction="up" if route_count else "neutral",
                owner=owner,
                source=source,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state=freshness_state,
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
            _contract_metric(
                signal_id="network_live_trucks",
                label="Live Trucks",
                value=str(live_trucks),
                delta="Telemetry current" if freshness_state == "fresh" else "Telemetry stale",
                direction="up" if freshness_state == "fresh" else "down",
                owner=owner,
                source=source,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state=freshness_state,
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
            _contract_metric(
                signal_id="network_congestion_level",
                label="Congestion Level",
                value=congestion_value,
                delta=congestion_delta,
                direction=congestion_direction,
                owner=owner,
                source=source,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state=freshness_state,
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
        ],
        alert=alert,
        owner=owner,
        source=source,
        refresh_cadence=refresh_cadence,
        stale_threshold=stale_threshold,
        freshness_state=freshness_state,
        decision_grade="advisory",
        fallback_behavior=fallback_behavior,
    )


def build_operations_shell_signal_bundle(conn: sqlite3.Connection) -> ShellSignalBundle:
    scope_label = "Operations"
    owner = "Operations control"
    source = "analytics.operations_assignment + analytics.operations_diary"
    refresh_cadence = "query-time operational state"
    stale_threshold = "review on page load"
    fallback_behavior = "downgrade to advisory or unknown state when board/diary summaries are unavailable"

    try:
        board_rows = list_job_operations_board(conn)
        anchor_date = _operations_anchor_date(board_rows)
        diary = build_operations_diary(conn, anchor_date=anchor_date, view_mode="day")
    except Exception:
        return ShellSignalBundle(
            scope_label=scope_label,
            metrics=[
                _contract_metric(
                    signal_id="ops_dispatch_fulfillment",
                    label="Dispatch Fulfillment",
                    value="Unknown",
                    owner=owner,
                    source=source,
                    refresh_cadence=refresh_cadence,
                    stale_threshold=stale_threshold,
                    freshness_state="unknown",
                    decision_grade="advisory",
                    fallback_behavior=fallback_behavior,
                ),
                _contract_metric(
                    signal_id="ops_open_tasks",
                    label="Open Tasks",
                    value="Unknown",
                    owner=owner,
                    source=source,
                    refresh_cadence=refresh_cadence,
                    stale_threshold=stale_threshold,
                    freshness_state="unknown",
                    decision_grade="advisory",
                    fallback_behavior=fallback_behavior,
                ),
                _contract_metric(
                    signal_id="ops_planned_labor",
                    label="Planned Labor",
                    value="Unknown",
                    owner=owner,
                    source=source,
                    refresh_cadence=refresh_cadence,
                    stale_threshold=stale_threshold,
                    freshness_state="unknown",
                    decision_grade="advisory",
                    fallback_behavior=fallback_behavior,
                ),
            ],
            alert=ShellAlertSignal(
                signal_id="ops_operational_state",
                title="Operational State Unavailable",
                message="Operations board and diary summaries could not be resolved. Treat this shell as advisory until review data is available again.",
                severity="warning",
                source=source,
                owner=owner,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state="unknown",
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
            owner=owner,
            source=source,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state="unknown",
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )

    summary = diary["summary"]
    total_jobs = len(board_rows)
    clear_jobs = sum(1 for row in board_rows if int(row.get("blockingCount") or 0) == 0)
    blocked_jobs = sum(1 for row in board_rows if int(row.get("blockingCount") or 0) > 0)
    fulfillment = 100.0 if total_jobs == 0 else (clear_jobs / total_jobs) * 100.0
    open_tasks = int(summary.get("openTaskCount") or 0)
    planned_labor = int(summary.get("plannedLaborCount") or 0)
    invoice_issues = int(summary.get("invoiceExceptionCount") or 0)
    bill_issues = int(summary.get("billExceptionCount") or 0)

    if blocked_jobs > 0:
        alert = ShellAlertSignal(
            signal_id="ops_operational_state",
            title="Blocking Jobs Require Review",
            message=f"{blocked_jobs} job{'s have' if blocked_jobs != 1 else ' has'} active blocking flags in the operations board.",
            severity="critical",
            source=source,
            owner=owner,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state="fresh",
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )
    elif invoice_issues > 0 or bill_issues > 0 or open_tasks > 0:
        alert = ShellAlertSignal(
            signal_id="ops_operational_state",
            title="Operational Follow-Through Pending",
            message=(
                f"{open_tasks} open task(s), {invoice_issues} invoice exception(s), "
                f"and {bill_issues} supplier-bill exception(s) are currently active."
            ),
            severity="warning",
            source=source,
            owner=owner,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state="fresh",
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )
    else:
        alert = ShellAlertSignal(
            signal_id="ops_operational_state",
            title="Operations Status",
            message="No blocking jobs or active reconciliation exceptions are currently reported in the operational summaries.",
            severity="info",
            source=source,
            owner=owner,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state="fresh",
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )

    return ShellSignalBundle(
        scope_label=scope_label,
        metrics=[
            _contract_metric(
                signal_id="ops_dispatch_fulfillment",
                label="Dispatch Fulfillment",
                value=f"{fulfillment:.0f}%",
                delta=f"{clear_jobs}/{total_jobs} jobs clear" if total_jobs else "No active jobs",
                direction="up" if blocked_jobs == 0 else "down",
                owner=owner,
                source=source,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state="fresh",
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
            _contract_metric(
                signal_id="ops_open_tasks",
                label="Open Tasks",
                value=str(open_tasks),
                delta="Action queue" if open_tasks else "No open tasks",
                direction="down" if open_tasks else "up",
                owner=owner,
                source=source,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state="fresh",
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
            _contract_metric(
                signal_id="ops_planned_labor",
                label="Planned Labor",
                value=str(planned_labor),
                delta=f"{summary.get('jobCount', 0)} jobs in diary",
                direction="neutral",
                owner=owner,
                source=source,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state="fresh",
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
        ],
        alert=alert,
        owner=owner,
        source=source,
        refresh_cadence=refresh_cadence,
        stale_threshold=stale_threshold,
        freshness_state="fresh",
        decision_grade="advisory",
        fallback_behavior=fallback_behavior,
    )


def build_quote_shell_signal_bundle(conn: sqlite3.Connection) -> ShellSignalBundle:
    scope_label = "Quote"
    owner = "Commercial operations"
    source = "quotes table"
    refresh_cadence = "query-time persisted quote history"
    stale_threshold = "review on page load"
    fallback_behavior = "downgrade to advisory or unknown state when persisted quote history is unavailable"

    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS quote_count,
                AVG(COALESCE(margin_percent, 0.0)) AS avg_margin_percent,
                COUNT(DISTINCT COALESCE(NULLIF(client_display, ''), 'Quote builder')) AS client_count,
                SUM(
                    CASE
                        WHEN quote_date >= date('now', '-7 day') THEN 1
                        ELSE 0
                    END
                ) AS recent_quote_count
            FROM quotes
            """
        ).fetchone()
    except Exception:
        return ShellSignalBundle(
            scope_label=scope_label,
            metrics=[
                _contract_metric(
                    signal_id="quote_saved_quotes",
                    label="Saved Quotes",
                    value="Unknown",
                    owner=owner,
                    source=source,
                    refresh_cadence=refresh_cadence,
                    stale_threshold=stale_threshold,
                    freshness_state="unknown",
                    decision_grade="advisory",
                    fallback_behavior=fallback_behavior,
                ),
                _contract_metric(
                    signal_id="quote_avg_margin_built",
                    label="Avg Margin Built",
                    value="Unknown",
                    owner=owner,
                    source=source,
                    refresh_cadence=refresh_cadence,
                    stale_threshold=stale_threshold,
                    freshness_state="unknown",
                    decision_grade="advisory",
                    fallback_behavior=fallback_behavior,
                ),
                _contract_metric(
                    signal_id="quote_recent_saved",
                    label="Saved Last 7 Days",
                    value="Unknown",
                    owner=owner,
                    source=source,
                    refresh_cadence=refresh_cadence,
                    stale_threshold=stale_threshold,
                    freshness_state="unknown",
                    decision_grade="advisory",
                    fallback_behavior=fallback_behavior,
                ),
            ],
            alert=ShellAlertSignal(
                signal_id="quote_commercial_state",
                title="Quote History Unavailable",
                message="Persisted quote history could not be read. Commercial shell signals are advisory only until saved quote data is available again.",
                severity="warning",
                source=source,
                owner=owner,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state="unknown",
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
            owner=owner,
            source=source,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state="unknown",
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )

    quote_count = int(row["quote_count"] or 0)
    avg_margin_percent = float(row["avg_margin_percent"] or 0.0)
    client_count = int(row["client_count"] or 0)
    recent_quote_count = int(row["recent_quote_count"] or 0)

    if quote_count == 0:
        alert = ShellAlertSignal(
            signal_id="quote_commercial_state",
            title="No Saved Quotes Yet",
            message="Persisted quote history is empty. Commercial shell signals remain advisory until quotes are saved from the builder.",
            severity="info",
            source=source,
            owner=owner,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state="fresh",
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )
    elif recent_quote_count == 0:
        alert = ShellAlertSignal(
            signal_id="quote_commercial_state",
            title="Commercial Activity Quiet",
            message="No saved quotes were recorded in the last 7 days. Review whether current pipeline activity is happening outside the persisted quote flow.",
            severity="warning",
            source=source,
            owner=owner,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state="fresh",
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )
    else:
        alert = ShellAlertSignal(
            signal_id="quote_commercial_state",
            title="Commercial Snapshot",
            message=f"{recent_quote_count} saved quote(s) were recorded in the last 7 days across {client_count} client record(s).",
            severity="info",
            source=source,
            owner=owner,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state="fresh",
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )

    return ShellSignalBundle(
        scope_label=scope_label,
        metrics=[
            _contract_metric(
                signal_id="quote_saved_quotes",
                label="Saved Quotes",
                value=str(quote_count),
                delta="Persisted history",
                direction="up" if quote_count else "neutral",
                owner=owner,
                source=source,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state="fresh",
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
            _contract_metric(
                signal_id="quote_avg_margin_built",
                label="Avg Margin Built",
                value=f"{avg_margin_percent:.1f}%",
                delta="Saved quotes only",
                direction="up" if avg_margin_percent > 0 else "neutral",
                owner=owner,
                source=source,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state="fresh",
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
            _contract_metric(
                signal_id="quote_recent_saved",
                label="Saved Last 7 Days",
                value=str(recent_quote_count),
                delta=f"{client_count} client record(s)",
                direction="up" if recent_quote_count else "neutral",
                owner=owner,
                source=source,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state="fresh",
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
        ],
        alert=alert,
        owner=owner,
        source=source,
        refresh_cadence=refresh_cadence,
        stale_threshold=stale_threshold,
        freshness_state="fresh",
        decision_grade="advisory",
        fallback_behavior=fallback_behavior,
    )


def build_pricing_shell_signal_bundle(
    filtered_df: pd.DataFrame,
    break_even_value: float,
    dataset_error: str | None = None,
) -> ShellSignalBundle:
    scope_label = "Pricing Intelligence"
    owner = "Pricing intelligence"
    source = "filtered pricing dataset"
    refresh_cadence = "query-time filtered pricing state"
    stale_threshold = "review on filter change"
    fallback_behavior = (
        "downgrade to advisory or unknown state when pricing inputs are unavailable "
        "or the current selection has no priced jobs"
    )

    if dataset_error:
        return ShellSignalBundle(
            scope_label=scope_label,
            metrics=[
                _contract_metric(
                    signal_id="pricing_median_margin",
                    label="Median Margin $/m³",
                    value="Unknown",
                    owner=owner,
                    source=source,
                    refresh_cadence=refresh_cadence,
                    stale_threshold=stale_threshold,
                    freshness_state="unknown",
                    decision_grade="advisory",
                    fallback_behavior=fallback_behavior,
                ),
                _contract_metric(
                    signal_id="pricing_below_break_even",
                    label="Below Break-Even",
                    value="Unknown",
                    owner=owner,
                    source=source,
                    refresh_cadence=refresh_cadence,
                    stale_threshold=stale_threshold,
                    freshness_state="unknown",
                    decision_grade="advisory",
                    fallback_behavior=fallback_behavior,
                ),
                _contract_metric(
                    signal_id="pricing_median_price",
                    label="Median Price / m³",
                    value="Unknown",
                    owner=owner,
                    source=source,
                    refresh_cadence=refresh_cadence,
                    stale_threshold=stale_threshold,
                    freshness_state="unknown",
                    decision_grade="advisory",
                    fallback_behavior=fallback_behavior,
                ),
            ],
            alert=ShellAlertSignal(
                signal_id="pricing_dataset_state",
                title="Pricing Dataset Unavailable",
                message="Pricing analytics inputs could not be resolved for the current selection. Treat margin and break-even signals as advisory until the dataset is available again.",
                severity="warning",
                source=source,
                owner=owner,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state="unknown",
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
            owner=owner,
            source=source,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state="unknown",
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )

    try:
        distribution = summarise_distribution(filtered_df, break_even_value)
        profitability = summarise_profitability(filtered_df)
    except (KeyError, TypeError, ValueError):
        return ShellSignalBundle(
            scope_label=scope_label,
            metrics=[
                _contract_metric(
                    signal_id="pricing_median_margin",
                    label="Median Margin $/m³",
                    value="Unknown",
                    owner=owner,
                    source=source,
                    refresh_cadence=refresh_cadence,
                    stale_threshold=stale_threshold,
                    freshness_state="unknown",
                    decision_grade="advisory",
                    fallback_behavior=fallback_behavior,
                ),
                _contract_metric(
                    signal_id="pricing_below_break_even",
                    label="Below Break-Even",
                    value="Unknown",
                    owner=owner,
                    source=source,
                    refresh_cadence=refresh_cadence,
                    stale_threshold=stale_threshold,
                    freshness_state="unknown",
                    decision_grade="advisory",
                    fallback_behavior=fallback_behavior,
                ),
                _contract_metric(
                    signal_id="pricing_median_price",
                    label="Median Price / m³",
                    value="Unknown",
                    owner=owner,
                    source=source,
                    refresh_cadence=refresh_cadence,
                    stale_threshold=stale_threshold,
                    freshness_state="unknown",
                    decision_grade="advisory",
                    fallback_behavior=fallback_behavior,
                ),
            ],
            alert=ShellAlertSignal(
                signal_id="pricing_dataset_state",
                title="Pricing Dataset Unavailable",
                message="Pricing analytics inputs could not be normalized for the current selection. Treat margin and break-even signals as advisory until the dataset is available again.",
                severity="warning",
                source=source,
                owner=owner,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state="unknown",
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
            owner=owner,
            source=source,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state="unknown",
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )

    median_margin = _format_currency_number(profitability.margin_per_m3_median)
    below_break_even_ratio = distribution.below_break_even_ratio * 100.0
    median_price = _format_currency_number(distribution.median)

    if distribution.priced_job_count == 0:
        alert = ShellAlertSignal(
            signal_id="pricing_dataset_state",
            title="No Priced Jobs In Scope",
            message="The current pricing selection has no priced jobs. Pricing shell signals stay advisory until priced history is available for the active filters.",
            severity="info",
            source=source,
            owner=owner,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state="fresh",
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )
    elif distribution.below_break_even_count > 0:
        severity = "critical" if distribution.below_break_even_ratio >= 0.3 else "warning"
        alert = ShellAlertSignal(
            signal_id="pricing_dataset_state",
            title="Break-Even Pressure Detected",
            message=(
                f"{distribution.below_break_even_count} priced job(s) in the current selection are below break-even, "
                f"representing {below_break_even_ratio:.0f}% of priced history."
            ),
            severity=severity,
            source=source,
            owner=owner,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state="fresh",
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )
    else:
        alert = ShellAlertSignal(
            signal_id="pricing_dataset_state",
            title="Pricing Snapshot",
            message="No priced jobs in the current selection are below break-even. Use the optimizer as an advisory review aid, not an automatic pricing decision.",
            severity="info",
            source=source,
            owner=owner,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state="fresh",
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )

    return ShellSignalBundle(
        scope_label=scope_label,
        metrics=[
            _contract_metric(
                signal_id="pricing_median_margin",
                label="Median Margin $/m³",
                value=median_margin,
                delta="Current filtered scope",
                direction="up" if _nonnegative(profitability.margin_per_m3_median) else "down",
                owner=owner,
                source=source,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state="fresh",
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
            _contract_metric(
                signal_id="pricing_below_break_even",
                label="Below Break-Even",
                value=str(distribution.below_break_even_count),
                delta=(
                    f"{below_break_even_ratio:.0f}% of priced jobs"
                    if distribution.priced_job_count
                    else "No priced jobs"
                ),
                direction="down" if distribution.below_break_even_count else "up",
                owner=owner,
                source=source,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state="fresh",
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
            _contract_metric(
                signal_id="pricing_median_price",
                label="Median Price / m³",
                value=median_price,
                delta=f"Break-even ${break_even_value:,.0f}/m³",
                direction="up" if _safe_ge(distribution.median, break_even_value) else "down",
                owner=owner,
                source=source,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state="fresh",
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
        ],
        alert=alert,
        owner=owner,
        source=source,
        refresh_cadence=refresh_cadence,
        stale_threshold=stale_threshold,
        freshness_state="fresh",
        decision_grade="advisory",
        fallback_behavior=fallback_behavior,
    )


def build_admin_shell_signal_bundle(conn: sqlite3.Connection) -> ShellSignalBundle:
    scope_label = "Admin"
    owner = "System administration"
    source = "dashboard_users + Kent governance + adaptive policy proposals"
    refresh_cadence = "query-time governance state"
    stale_threshold = "review on page load"
    fallback_behavior = (
        "downgrade unavailable governance surfaces to unknown without bootstrapping tables or seed state during a normal page render"
    )

    user_summary = _read_only_admin_user_summary(conn)
    kent_summary = _read_only_kent_governance_summary(conn)
    adaptive_summary = _read_only_adaptive_policy_summary(conn)

    active_users = user_summary["active_users"] if user_summary else None
    active_admins = user_summary["active_admins"] if user_summary else None
    active_reasons = kent_summary["active_reason_count"] if kent_summary else None
    tender_review_count = kent_summary["tender_review_count"] if kent_summary else None
    proposal_review_count = adaptive_summary["pending_review_count"] if adaptive_summary else None
    policy_mode = kent_summary["policy_mode"] if kent_summary else None

    user_governance_available = (
        user_summary is not None
        and active_users is not None
        and active_admins is not None
    )
    open_review_count = (
        int(tender_review_count or 0) + int(proposal_review_count or 0)
        if tender_review_count is not None or proposal_review_count is not None
        else None
    )
    governance_available = (
        user_governance_available
        and kent_summary is not None
        and adaptive_summary is not None
        and open_review_count is not None
    )
    bundle_freshness: FreshnessState = "fresh" if governance_available else "unknown"

    if active_admins == 0:
        alert = ShellAlertSignal(
            signal_id="admin_governance_state",
            title="No Active Admin Coverage",
            message=(
                "No active system rollout admin is currently configured. Governance changes and user-management recovery may be blocked until admin coverage is restored."
                if governance_available
                else "No active system rollout admin is currently configured, and some governance control surfaces are unavailable for read-only review."
            ),
            severity="critical",
            source=source,
            owner=owner,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state="fresh",
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )
    elif open_review_count and open_review_count > 0:
        alert = ShellAlertSignal(
            signal_id="admin_governance_state",
            title="Governance Review Pending",
            message=(
                f"{tender_review_count} Kent tender review(s) and {proposal_review_count} adaptive-policy proposal(s) "
                "currently require governance attention."
            ),
            severity="warning",
            source=source,
            owner=owner,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state="fresh",
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )
    elif not governance_available:
        alert = ShellAlertSignal(
            signal_id="admin_governance_state",
            title="Governance State Unavailable",
            message="User, Kent, or adaptive-policy governance summaries could not be resolved in read-only mode. Treat the admin shell as advisory until those control surfaces are available again.",
            severity="warning",
            source=source,
            owner=owner,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state="unknown",
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )
    else:
        alert = ShellAlertSignal(
            signal_id="admin_governance_state",
            title="Governance Snapshot",
            message="Admin coverage, Kent policy defaults, and adaptive-policy review queues are currently in a reviewable state.",
            severity="info",
            source=source,
            owner=owner,
            refresh_cadence=refresh_cadence,
            stale_threshold=stale_threshold,
            freshness_state="fresh",
            decision_grade="advisory",
            fallback_behavior=fallback_behavior,
        )

    return ShellSignalBundle(
        scope_label=scope_label,
        metrics=[
            _contract_metric(
                signal_id="admin_active_users",
                label="Active Users",
                value=str(active_users) if active_users is not None else "Unknown",
                delta=(
                    f"{user_summary['total_users']} local user record(s)"
                    if user_summary is not None
                    else "User records unavailable"
                ),
                direction="up" if active_users else ("neutral" if active_users == 0 else "neutral"),
                owner=owner,
                source=source,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state="fresh" if active_users is not None else "unknown",
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
            _contract_metric(
                signal_id="admin_active_admins",
                label="Active Admins",
                value=str(active_admins) if active_admins is not None else "Unknown",
                delta=(
                    f"{active_reasons} active override reason(s)"
                    if active_reasons is not None
                    else "Kent override reasons unavailable"
                ),
                direction="up" if active_admins else ("down" if active_admins == 0 else "neutral"),
                owner=owner,
                source=source,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state="fresh" if active_admins is not None else "unknown",
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
            _contract_metric(
                signal_id="admin_open_reviews",
                label="Open Reviews",
                value=str(open_review_count) if open_review_count is not None else "Unknown",
                delta=(
                    f"Kent {policy_mode} mode"
                    if policy_mode
                    else "Kent policy unavailable"
                ),
                direction=(
                    "down"
                    if open_review_count
                    else ("up" if open_review_count == 0 else "neutral")
                ),
                owner=owner,
                source=source,
                refresh_cadence=refresh_cadence,
                stale_threshold=stale_threshold,
                freshness_state="fresh" if open_review_count is not None else "unknown",
                decision_grade="advisory",
                fallback_behavior=fallback_behavior,
            ),
        ],
        alert=alert,
        owner=owner,
        source=source,
        refresh_cadence=refresh_cadence,
        stale_threshold=stale_threshold,
        freshness_state=bundle_freshness,
        decision_grade="advisory",
        fallback_behavior=fallback_behavior,
    )


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _read_only_admin_user_summary(conn: sqlite3.Connection) -> dict[str, int] | None:
    if not _table_exists(conn, "dashboard_users"):
        return None
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total_users,
            SUM(CASE WHEN active = 1 THEN 1 ELSE 0 END) AS active_users,
            SUM(
                CASE
                    WHEN active = 1 AND role_key = 'system_rollout_admin' THEN 1
                    ELSE 0
                END
            ) AS active_admins
        FROM dashboard_users
        """
    ).fetchone()
    if row is None:
        return None
    return {
        "total_users": int(row["total_users"] or 0),
        "active_users": int(row["active_users"] or 0),
        "active_admins": int(row["active_admins"] or 0),
    }


def _read_only_kent_governance_summary(conn: sqlite3.Connection) -> dict[str, int | str | None] | None:
    global_parameters_available = _table_exists(conn, "global_parameters")
    tender_table_available = _table_exists(conn, "kent_job_tenders")
    reason_table_available = _table_exists(conn, "kent_tender_override_reason_codes")

    if not any((global_parameters_available, tender_table_available, reason_table_available)):
        return None

    policy_mode: str | None = None
    if global_parameters_available:
        policy_row = conn.execute(
            "SELECT value_text FROM global_parameters WHERE key = ?",
            (KENT_TENDER_RULE_MODE_KEY,),
        ).fetchone()
        if policy_row is not None and policy_row["value_text"]:
            policy_mode = str(policy_row["value_text"])

    active_reason_count: int | None = None
    if reason_table_available:
        reason_row = conn.execute(
            "SELECT COUNT(*) AS active_reason_count FROM kent_tender_override_reason_codes WHERE active = 1"
        ).fetchone()
        active_reason_count = int(reason_row["active_reason_count"] or 0) if reason_row else 0

    tender_review_count: int | None = None
    if tender_table_available:
        review_row = conn.execute(
            """
            SELECT COUNT(*) AS tender_review_count
            FROM kent_job_tenders
            WHERE tender_status = 'open'
              AND recommended_action IN (
                  'hard_blocked',
                  'review_with_override',
                  'review_if_strategic',
                  'review_today',
                  'review_if_capacity',
                  'pursue_with_flags'
              )
            """
        ).fetchone()
        tender_review_count = int(review_row["tender_review_count"] or 0) if review_row else 0

    if policy_mode is None and active_reason_count is None and tender_review_count is None:
        return None
    return {
        "policy_mode": policy_mode,
        "active_reason_count": active_reason_count,
        "tender_review_count": tender_review_count,
    }


def _read_only_adaptive_policy_summary(conn: sqlite3.Connection) -> dict[str, int] | None:
    if not _table_exists(conn, "adaptive_policy_proposals"):
        return None
    row = conn.execute(
        """
        SELECT COUNT(*) AS pending_review_count
        FROM adaptive_policy_proposals
        WHERE status = ?
        """,
        (PROPOSAL_STATUS_PENDING_REVIEW,),
    ).fetchone()
    if row is None:
        return None
    return {"pending_review_count": int(row["pending_review_count"] or 0)}


def _scaffold_metric(
    *,
    signal_id: str,
    label: str,
    value: str,
    delta: str | None = None,
    direction: str = "neutral",
    owner: str,
    source: str,
) -> ShellMetricSignal:
    return ShellMetricSignal(
        signal_id=signal_id,
        label=label,
        value=value,
        delta=delta,
        direction=direction,
        source=source,
        owner=owner,
        refresh_cadence="manual placeholder",
        stale_threshold="not applicable until sourced",
        freshness_state="scaffold",
        decision_grade="placeholder",
        fallback_behavior="render explicit placeholder notice instead of implying live truth",
    )


def _contract_metric(
    *,
    signal_id: str,
    label: str,
    value: str,
    delta: str | None = None,
    direction: str = "neutral",
    owner: str,
    source: str,
    refresh_cadence: str,
    stale_threshold: str,
    freshness_state: FreshnessState,
    decision_grade: DecisionGrade,
    fallback_behavior: str,
) -> ShellMetricSignal:
    return ShellMetricSignal(
        signal_id=signal_id,
        label=label,
        value=value,
        delta=delta,
        direction=direction,
        source=source,
        owner=owner,
        refresh_cadence=refresh_cadence,
        stale_threshold=stale_threshold,
        freshness_state=freshness_state,
        decision_grade=decision_grade,
        fallback_behavior=fallback_behavior,
    )


def _scaffold_alert(
    *,
    signal_id: str,
    title: str,
    message: str,
    severity: str,
    owner: str,
    source: str,
) -> ShellAlertSignal:
    return ShellAlertSignal(
        signal_id=signal_id,
        title=title,
        message=message,
        severity=severity,
        source=source,
        owner=owner,
        refresh_cadence="manual placeholder",
        stale_threshold="not applicable until sourced",
        freshness_state="scaffold",
        decision_grade="placeholder",
        fallback_behavior="downgrade to explicit placeholder governance notice",
    )


def _bundle(
    *,
    scope_label: str,
    owner: str,
    source: str,
    metrics: Sequence[ShellMetricSignal],
    alert: ShellAlertSignal,
) -> ShellSignalBundle:
    return ShellSignalBundle(
        scope_label=scope_label,
        metrics=metrics,
        alert=alert,
        owner=owner,
        source=source,
        refresh_cadence="manual placeholder",
        stale_threshold="not applicable until sourced",
        freshness_state="scaffold",
        decision_grade="placeholder",
        fallback_behavior="render governance notice and avoid decision-grade claims",
    )


def _build_quote_bundle(scope_label: str) -> ShellSignalBundle:
    return _bundle(
        scope_label=scope_label,
        owner="Commercial operations",
        source="Quote shell scaffold",
        metrics=[
            _scaffold_metric(signal_id="quote_win_rate", label="Quote Win Rate", value="34%", delta="+2%", direction="up", owner="Commercial operations", source="Quote shell scaffold"),
            _scaffold_metric(signal_id="quote_margin_built", label="Avg Margin Built", value="21%", delta="-1%", direction="down", owner="Commercial operations", source="Quote shell scaffold"),
            _scaffold_metric(signal_id="quote_active_pending", label="Active Pending", value="12", owner="Commercial operations", source="Quote shell scaffold"),
        ],
        alert=_scaffold_alert(
            signal_id="quote_pending_review",
            title="Quotes Pending Review",
            message="3 quotes are awaiting urgent review. Client Acme Inc credit check delayed.",
            severity="warning",
            owner="Commercial operations",
            source="Quote shell scaffold",
        ),
    )


def _build_pricing_bundle(scope_label: str) -> ShellSignalBundle:
    return _bundle(
        scope_label=scope_label,
        owner="Pricing intelligence",
        source="Pricing shell scaffold",
        metrics=[
            _scaffold_metric(signal_id="pricing_network_margin", label="Network Margin", value="18.5%", delta="-0.5%", direction="down", owner="Pricing intelligence", source="Pricing shell scaffold"),
            _scaffold_metric(signal_id="pricing_yield_target", label="Yield vs Target", value="-2%", delta="Below", direction="down", owner="Pricing intelligence", source="Pricing shell scaffold"),
            _scaffold_metric(signal_id="pricing_loss_leading_corridors", label="Loss Leading Corridors", value="4", owner="Pricing intelligence", source="Pricing shell scaffold"),
        ],
        alert=_scaffold_alert(
            signal_id="pricing_break_even_alert",
            title="Corridor Below Break-Even",
            message="SYD-MEL is operating below break-even margin for the last 5 days.",
            severity="critical",
            owner="Pricing intelligence",
            source="Pricing shell scaffold",
        ),
    )


def _build_network_bundle(scope_label: str) -> ShellSignalBundle:
    return _bundle(
        scope_label=scope_label,
        owner="Network control",
        source="Network shell scaffold",
        metrics=[
            _scaffold_metric(signal_id="network_active_nodes", label="Active Nodes", value="24", delta="+1", direction="up", owner="Network control", source="Network shell scaffold"),
            _scaffold_metric(signal_id="network_live_trucks", label="Live Trucks", value="18", delta="Optimal", direction="up", owner="Network control", source="Network shell scaffold"),
            _scaffold_metric(signal_id="network_congestion_level", label="Congestion Level", value="Low", owner="Network control", source="Network shell scaffold"),
        ],
        alert=_scaffold_alert(
            signal_id="network_status",
            title="Network Status",
            message="No significant network disruptions. SYD terminal operating at 85% capacity.",
            severity="info",
            owner="Network control",
            source="Network shell scaffold",
        ),
    )


def _build_operations_bundle(scope_label: str) -> ShellSignalBundle:
    return _bundle(
        scope_label=scope_label,
        owner="Operations control",
        source="Operations shell scaffold",
        metrics=[
            _scaffold_metric(signal_id="ops_dispatch_fulfillment", label="Dispatch Fulfillment", value="98%", delta="+1%", direction="up", owner="Operations control", source="Operations shell scaffold"),
            _scaffold_metric(signal_id="ops_active_driver_shifts", label="Active Driver Shifts", value="42", owner="Operations control", source="Operations shell scaffold"),
            _scaffold_metric(signal_id="ops_fleet_availability", label="Fleet Availability", value="92%", delta="-2%", direction="down", owner="Operations control", source="Operations shell scaffold"),
        ],
        alert=_scaffold_alert(
            signal_id="ops_maintenance_shift_gap",
            title="Operations Alert",
            message="Maintenance due on 2 long-haul vehicles. 1 Shift gap in Metro deliveries.",
            severity="critical",
            owner="Operations control",
            source="Operations shell scaffold",
        ),
    )


def _build_admin_bundle(scope_label: str) -> ShellSignalBundle:
    return _bundle(
        scope_label=scope_label,
        owner="System administration",
        source="Admin shell scaffold",
        metrics=[
            _scaffold_metric(signal_id="admin_system_health", label="System Health", value="99.9%", delta="+0.01%", direction="up", owner="System administration", source="Admin shell scaffold"),
            _scaffold_metric(signal_id="admin_active_users", label="Active Users", value="14", owner="System administration", source="Admin shell scaffold"),
            _scaffold_metric(signal_id="admin_pending_roles", label="Pending Roles", value="0", owner="System administration", source="Admin shell scaffold"),
        ],
        alert=_scaffold_alert(
            signal_id="admin_integration_status",
            title="Integration Status",
            message="Kent AMS integration running smoothly.",
            severity="info",
            owner="System administration",
            source="Admin shell scaffold",
        ),
    )


def _build_unknown_bundle(scope_label: str) -> ShellSignalBundle:
    return _bundle(
        scope_label=scope_label,
        owner="Unassigned",
        source="Unknown shell scaffold",
        metrics=[],
        alert=_scaffold_alert(
            signal_id="unknown_scope",
            title="Signal State Unknown",
            message="No reviewed shell signal contract is registered for this scope yet.",
            severity="warning",
            owner="Unassigned",
            source="Unknown shell scaffold",
        ),
    )


_SHELL_SIGNAL_BUILDERS = {
    "quote": _build_quote_bundle,
    "pricing intelligence": _build_pricing_bundle,
    "network": _build_network_bundle,
    "operations": _build_operations_bundle,
    "admin": _build_admin_bundle,
}


def _latest_timestamp(*frames) -> datetime | None:
    latest: datetime | None = None
    for frame in frames:
        if frame.empty or "updated_at" not in frame.columns:
            continue
        parsed = [
            _parse_timestamp(value)
            for value in frame["updated_at"].dropna().tolist()
        ]
        parsed = [value for value in parsed if value is not None]
        if not parsed:
            continue
        candidate = max(parsed)
        if latest is None or candidate > latest:
            latest = candidate
    return latest


def _parse_timestamp(value: object) -> datetime | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _freshness_state(updated_at: datetime | None, *, stale_after: timedelta) -> FreshnessState:
    if updated_at is None:
        return "unknown"
    if datetime.now(UTC) - updated_at > stale_after:
        return "stale"
    return "fresh"


def _count_active_nodes(routes) -> int:
    if routes.empty:
        return 0
    coords: set[tuple[float, float]] = set()
    for lat_column, lon_column in (("origin_lat", "origin_lon"), ("dest_lat", "dest_lon")):
        if lat_column not in routes.columns or lon_column not in routes.columns:
            continue
        for lat, lon in zip(routes[lat_column], routes[lon_column]):
            try:
                coords.add((round(float(lat), 5), round(float(lon), 5)))
            except (TypeError, ValueError):
                continue
    return len(coords)


def _count_delayed_routes(routes) -> int:
    if routes.empty or "status" not in routes.columns:
        return 0
    return sum(1 for status in routes["status"].fillna("").tolist() if str(status) == "delayed")


def _operations_anchor_date(board_rows: list[dict]) -> str:
    candidates: list[str] = []
    for row in board_rows:
        planned_start = str(row.get("plannedStart") or "")
        if len(planned_start) >= 10:
            candidates.append(planned_start[:10])
    return min(candidates) if candidates else date.today().isoformat()


def _format_currency_number(value: float | None) -> str:
    if value is None:
        return "Unknown"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "Unknown"
    if math.isnan(numeric):
        return "Unknown"
    return f"${numeric:,.0f}"


def _nonnegative(value: float | None) -> bool:
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(numeric) and numeric >= 0


def _safe_ge(left: float | None, right: float) -> bool:
    if left is None:
        return False
    try:
        numeric = float(left)
    except (TypeError, ValueError):
        return False
    return not math.isnan(numeric) and numeric >= right
