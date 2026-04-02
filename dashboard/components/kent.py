from __future__ import annotations

import sqlite3
from typing import Any, Callable

import pandas as pd
import streamlit as st

from analytics.kent_ams_import import (
    get_kent_tender_policy_config,
    list_kent_override_reason_codes,
    list_kent_tender_override_history,
    list_prioritized_tenders,
    record_kent_tender_override,
    update_kent_tender_policy_config,
    upsert_kent_override_reason_code,
)
from corkysoft.quote_service import format_currency


KENT_ADMIN_WRITE_ROLES: frozenset[str] = frozenset({"system_rollout_admin"})


def _kent_admin_write_enabled(current_role_key: str | None) -> bool:
    return current_role_key in KENT_ADMIN_WRITE_ROLES


def render_kent_tenders_tab(
    conn: sqlite3.Connection,
    *,
    rerun_app: Callable[[], None],
) -> None:
    st.subheader("Kent AMS tender queue")
    st.caption(
        "Profitability rule mode drives priority, not hard blocking. Only safety/legal/compliance flags should hard-block."
    )

    policy = get_kent_tender_policy_config(conn)
    queue_cols = st.columns([1, 1, 1, 1])
    status_filter = queue_cols[0].selectbox(
        "Status",
        options=["open", "awarded", "closed", "all"],
        index=0,
        key="kent_tender_status_filter",
    )
    limit_value = int(
        queue_cols[1].number_input(
            "Rows",
            min_value=5,
            max_value=250,
            value=25,
            step=5,
            key="kent_tender_limit",
        )
    )
    operator_id = queue_cols[2].text_input(
        "Operator ID",
        value=st.session_state.get("kent_tender_operator_id", ""),
        key="kent_tender_operator_id",
    )
    queue_cols[3].metric("Rule mode", policy["ruleMode"])
    triage_filters = st.columns(3)
    hard_block_scope = triage_filters[0].selectbox(
        "Hard-block scope",
        options=["all", "hide_hard_blocked", "only_hard_blocked"],
        index=0,
        key="kent_tender_hard_block_scope",
    )
    policy_scope = triage_filters[1].selectbox(
        "Policy scope",
        options=["all", "policy_fail_only", "policy_matched_only"],
        index=0,
        key="kent_tender_policy_scope",
    )
    loss_scope = triage_filters[2].selectbox(
        "Loss scope",
        options=["all", "loss_alert_only", "hide_loss_alerts"],
        index=0,
        key="kent_tender_loss_scope",
    )

    rows = list_prioritized_tenders(conn, status=status_filter, limit=limit_value)
    if hard_block_scope == "hide_hard_blocked":
        rows = [row for row in rows if not row["hardBlockFlags"]]
    elif hard_block_scope == "only_hard_blocked":
        rows = [row for row in rows if row["hardBlockFlags"]]
    if policy_scope == "policy_fail_only":
        rows = [row for row in rows if not row["policyMatched"]]
    elif policy_scope == "policy_matched_only":
        rows = [row for row in rows if row["policyMatched"]]
    if loss_scope == "loss_alert_only":
        rows = [row for row in rows if row["lossAlert"]]
    elif loss_scope == "hide_loss_alerts":
        rows = [row for row in rows if not row["lossAlert"]]
    if not rows:
        st.info("No Kent tenders found for the selected filter.")
        return

    reason_options = {
        row["code"]: row["label"]
        for row in list_kent_override_reason_codes(conn)
        if row["active"]
    }
    if not reason_options:
        st.warning(
            "No active override reasons are configured. Operators can review the queue, but overrides are disabled until an admin activates at least one reason."
        )

    summary_rows = [
        {
            "Tender": row["tenderExternalId"],
            "Job": row["jobNumber"],
            "Client": row["clientName"],
            "Origin": row["origin"],
            "Destination": row["destination"],
            "Action": row["recommendedAction"],
            "Policy": "PASS" if row["policyMatched"] else "FAIL",
            "Margin": row["estimatedMargin"],
            "Margin %": row["estimatedMarginPct"],
            "Score": row["scoreTotal"],
            "Loss": "ALERT" if row["lossAlert"] else "",
            "Freshness": row["freshnessState"],
        }
        for row in rows
    ]
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    for row in rows:
        badge_parts = []
        if row["hardBlockFlags"]:
            badge_parts.append("HARD BLOCK")
        if row["lossAlert"]:
            badge_parts.append("LOSS ALERT")
        if not row["policyMatched"]:
            badge_parts.append("POLICY FAIL")
        header = " | ".join(
            part for part in [row["tenderExternalId"], row["jobNumber"], ", ".join(badge_parts)] if part
        )
        with st.expander(header or row["tenderExternalId"], expanded=False):
            detail_cols = st.columns(4)
            detail_cols[0].metric("Expected revenue", format_currency(row["expectedRevenue"] or 0.0))
            detail_cols[1].metric("Est. margin", format_currency(row["estimatedMargin"] or 0.0))
            detail_cols[2].metric(
                "Est. margin %",
                "n/a" if row["estimatedMarginPct"] is None else f"{row['estimatedMarginPct']:.1f}%",
            )
            detail_cols[3].metric("Priority score", f"{row['scoreTotal']:.1f}")
            st.caption(
                f"Rule mode `{row['profitRuleMode']}` | thresholds: ${row['absoluteMarginThreshold']:,.0f} and {row['marginPercentThreshold']:.1f}% | freshness `{row['freshnessState']}` ({row['confidenceScore']:.1f})"
            )
            if row["policyFailReasons"]:
                st.warning("Policy fail reasons: " + ", ".join(row["policyFailReasons"]))
            if row["overrideableFlags"]:
                st.info("Overrideable flags: " + ", ".join(row["overrideableFlags"]))
            if row["hardBlockFlags"]:
                st.error("Hard-block flags: " + ", ".join(row["hardBlockFlags"]))

            with st.form(f"kent_override_form_{row['tenderExternalId']}"):
                action = st.selectbox(
                    "Action",
                    options=["pursue", "review", "defer", "award_override"],
                    key=f"kent_override_action_{row['tenderExternalId']}",
                )
                reason_code = st.selectbox(
                    "Reason code",
                    options=list(reason_options.keys()) or ["<no-active-reasons>"],
                    format_func=lambda code: reason_options.get(code, code),
                    key=f"kent_override_reason_{row['tenderExternalId']}",
                )
                note = st.text_area(
                    "Optional note",
                    key=f"kent_override_note_{row['tenderExternalId']}",
                    height=80,
                )
                submit_disabled = (
                    bool(row["hardBlockFlags"])
                    or not operator_id.strip()
                    or not reason_options
                )
                if st.form_submit_button("Record override", disabled=submit_disabled):
                    try:
                        record_kent_tender_override(
                            conn,
                            tender_external_id=row["tenderExternalId"],
                            action=action,
                            operator_id=operator_id.strip(),
                            reason_code=reason_code,
                            note=note,
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.success("Override recorded.")
                        rerun_app()
                if not operator_id.strip():
                    st.caption("Enter an operator ID to record override actions.")


def render_kent_admin_tab(
    conn: sqlite3.Connection,
    *,
    current_role_key: str | None = None,
    rerun_app: Callable[[], None],
    render_dashboard_user_admin: Callable[[sqlite3.Connection, str | None], None] | None = None,
) -> None:
    st.subheader("Kent AMS admin")
    st.caption(
        "Use this surface for policy defaults, override reason governance, and review. Operators should work from the Kent tenders tab."
    )

    if current_role_key and render_dashboard_user_admin is not None:
        render_dashboard_user_admin(conn, current_role_key=current_role_key)

    write_enabled = _kent_admin_write_enabled(current_role_key)

    policy = get_kent_tender_policy_config(conn)
    if not write_enabled:
        st.info(
            "Kent governance controls are admin-only. Operators should review tenders in the Kent tenders tab."
        )

    with st.form("kent_tender_policy_form"):
        config_cols = st.columns(4)
        rule_mode = config_cols[0].selectbox(
            "Rule mode",
            options=["ABS_ONLY", "PCT_ONLY", "EITHER", "BOTH"],
            index=["ABS_ONLY", "PCT_ONLY", "EITHER", "BOTH"].index(policy["ruleMode"]),
            disabled=not write_enabled,
        )
        abs_threshold = config_cols[1].number_input(
            "Abs margin threshold",
            value=float(policy["absoluteMarginThreshold"]),
            step=100.0,
            disabled=not write_enabled,
        )
        pct_threshold = config_cols[2].number_input(
            "Margin % threshold",
            value=float(policy["marginPercentThreshold"]),
            step=1.0,
            disabled=not write_enabled,
        )
        loss_floor = config_cols[3].number_input(
            "Loss alert floor",
            value=float(policy["lossAlertFloor"]),
            step=100.0,
            disabled=not write_enabled,
        )
        if st.form_submit_button("Save policy defaults", disabled=not write_enabled):
            try:
                update_kent_tender_policy_config(
                    conn,
                    rule_mode=rule_mode,
                    absolute_margin_threshold=float(abs_threshold),
                    margin_percent_threshold=float(pct_threshold),
                    loss_alert_floor=float(loss_floor),
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success("Kent tender policy defaults updated.")
                rerun_app()

    reasons = list_kent_override_reason_codes(conn)
    if reasons:
        st.dataframe(pd.DataFrame(reasons), width="stretch", hide_index=True)

    review_rows = list_prioritized_tenders(conn, status="all", limit=100)
    review_summary = {
        "hardBlocked": sum(1 for row in review_rows if row["hardBlockFlags"]),
        "policyFail": sum(1 for row in review_rows if not row["policyMatched"]),
        "lossAlert": sum(1 for row in review_rows if row["lossAlert"]),
        "overrideable": sum(1 for row in review_rows if row["overrideableFlags"]),
    }
    review_cols = st.columns(4)
    review_cols[0].metric("Hard blocked", review_summary["hardBlocked"])
    review_cols[1].metric("Policy fail", review_summary["policyFail"])
    review_cols[2].metric("Loss alerts", review_summary["lossAlert"])
    review_cols[3].metric("Overrideable", review_summary["overrideable"])

    with st.form("kent_override_reason_form"):
        reason_cols = st.columns(4)
        new_code = reason_cols[0].text_input("Code", disabled=not write_enabled)
        new_label = reason_cols[1].text_input("Label", disabled=not write_enabled)
        new_description = reason_cols[2].text_input("Description", disabled=not write_enabled)
        new_active = reason_cols[3].checkbox("Active", value=True, disabled=not write_enabled)
        if st.form_submit_button("Save reason", disabled=not write_enabled):
            try:
                upsert_kent_override_reason_code(
                    conn,
                    code=new_code,
                    label=new_label,
                    description=new_description,
                    active=new_active,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success("Override reason saved.")
                rerun_app()
    recent_tenders = list_prioritized_tenders(conn, status="all", limit=10)
    recent_override_rows: list[dict[str, Any]] = []
    for tender_row in recent_tenders:
        history = list_kent_tender_override_history(
            conn, tender_external_id=tender_row["tenderExternalId"]
        )
        for item in history[:3]:
            recent_override_rows.append(
                {
                    "Tender": tender_row["tenderExternalId"],
                    "At": item["createdAt"],
                    "Action": item["action"],
                    "Operator": item["operatorId"],
                    "Reason": item["reasonCode"],
                    "Note": item["note"],
                    "Policy matched": item["policyMatched"],
                    "Loss alert": item["lossAlert"],
                }
            )
    if recent_override_rows:
        st.markdown("#### Recent override history")
        st.dataframe(
            pd.DataFrame(recent_override_rows).sort_values("At", ascending=False),
            width="stretch",
            hide_index=True,
        )
