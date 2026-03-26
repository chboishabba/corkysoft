from __future__ import annotations

import os
import sqlite3
from typing import Any, Callable

import pandas as pd
import streamlit as st

from analytics.db import (
    INVENTORY_ARCHITECTURES,
    INVENTORY_CUSTODY_TYPES,
    INVENTORY_STATES,
    INVENTORY_SUBSTITUTION_APPROVER_ROLES,
    allocate_inventory_to_segment,
    decide_inventory_substitution,
    get_allowed_inventory_execution_stages,
    import_inventory_items_from_dataframe,
    import_inventory_movements_from_dataframe,
    import_suppliers_from_google_sheet,
    list_inventory,
    list_inventory_balances,
    list_inventory_execution_events,
    list_inventory_exceptions,
    list_inventory_movements,
    list_inventory_requirements,
    list_inventory_substitution_reason_codes,
    list_inventory_substitutions,
    list_segment_inventory_coordination,
    record_inventory_execution_event,
    record_inventory_movement,
    request_inventory_substitution,
    resolve_inventory_exception,
    upsert_inventory_requirement,
    upsert_inventory_substitution_reason_code,
)


def _read_uploaded_inventory_file(uploaded_file: Any | None) -> pd.DataFrame:
    """Parse a CSV or Excel upload into a dataframe."""

    if uploaded_file is None:
        return pd.DataFrame()

    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if filename.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload CSV or Excel.")


def _default_operations_sheet_reference() -> str:
    return (
        os.environ.get("OPERATIONS_WORKBOOK_URL")
        or os.environ.get("OPERATIONS_WORKBOOK_SHEET_ID")
        or ""
    )


def render_inventory_tab(
    conn: sqlite3.Connection,
    *,
    rerun_app: Callable[[], None],
) -> None:
    st.subheader("Inventory and movements")
    st.caption(
        "Execution stages are constrained warehouse actions layered above the lower-level logistics states."
    )

    segment_coordination = list_segment_inventory_coordination(conn)
    st.markdown("#### Segment-linked inventory coordination")
    if segment_coordination:
        coordination_df = pd.DataFrame(segment_coordination)
        coordination_df["inventoryNames"] = coordination_df["inventoryNames"].apply(
            lambda values: ", ".join(values) if isinstance(values, list) else ""
        )
        coordination_df["supplierNames"] = coordination_df["supplierNames"].apply(
            lambda values: ", ".join(values) if isinstance(values, list) else ""
        )
        coordination_df["architectures"] = coordination_df["architectures"].apply(
            lambda values: ", ".join(values) if isinstance(values, list) else ""
        )
        coordination_df["requirementNames"] = coordination_df["requirementNames"].apply(
            lambda values: ", ".join(values) if isinstance(values, list) else ""
        )
        st.dataframe(
            coordination_df[
                [
                    "jobId",
                    "segmentSequence",
                    "fromLocation",
                    "toLocation",
                    "plannedStart",
                    "assignmentStatus",
                    "requirementCount",
                    "requiredQuantity",
                    "shipmentCount",
                    "allocatedQuantity",
                    "shortageQuantity",
                    "inventoryNames",
                    "requirementNames",
                    "architectures",
                    "supplierNames",
                ]
            ].rename(
                columns={
                    "jobId": "Job",
                    "segmentSequence": "Segment",
                    "fromLocation": "From",
                    "toLocation": "To",
                    "plannedStart": "Planned start",
                    "assignmentStatus": "Status",
                    "requirementCount": "Requirements",
                    "requiredQuantity": "Required qty",
                    "shipmentCount": "Shipments",
                    "allocatedQuantity": "Allocated qty",
                    "shortageQuantity": "Shortage qty",
                    "inventoryNames": "Inventory",
                    "requirementNames": "Requirement lines",
                    "architectures": "Architectures",
                    "supplierNames": "Suppliers",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No segment-linked inventory allocations recorded yet.")

    state_filter = st.multiselect(
        "Filter by state",
        INVENTORY_STATES,
        default=list(INVENTORY_STATES),
        help="States are derived from movement events and item imports.",
    )

    job_filter_raw = st.text_input(
        "Filter by job (numeric id)",
        value="",
        help="Leave blank to show all jobs.",
    )
    job_filter: int | None = None
    if job_filter_raw.strip():
        try:
            job_filter = int(job_filter_raw)
        except ValueError:
            st.warning("Job filter must be a number if provided.")

    requirements = list_inventory_requirements(conn, job_id=job_filter)
    st.markdown("#### Requirement planning")
    if requirements:
        requirements_df = pd.DataFrame(requirements)
        st.dataframe(
            requirements_df[
                [
                    "jobId",
                    "segmentSequence",
                    "requirementName",
                    "inventoryName",
                    "architecture",
                    "requiredQuantity",
                    "allocatedQuantity",
                    "approvedSubstitutionQuantity",
                    "requestedSubstitutionQuantity",
                    "effectiveFulfilledQuantity",
                    "shortageQuantity",
                    "substitutionAllowed",
                    "hasPendingSubstitution",
                    "executionStage",
                    "executionActor",
                    "unit",
                    "notes",
                ]
            ].rename(
                columns={
                    "jobId": "Job",
                    "segmentSequence": "Segment",
                    "requirementName": "Requirement",
                    "inventoryName": "Inventory item",
                    "architecture": "Architecture",
                    "requiredQuantity": "Required qty",
                    "allocatedQuantity": "Allocated qty",
                    "approvedSubstitutionQuantity": "Approved substitution qty",
                    "requestedSubstitutionQuantity": "Requested substitution qty",
                    "effectiveFulfilledQuantity": "Effective fulfilled qty",
                    "shortageQuantity": "Shortage qty",
                    "substitutionAllowed": "Substitutable",
                    "hasPendingSubstitution": "Pending substitution",
                    "executionStage": "Execution stage",
                    "executionActor": "Last actor",
                    "unit": "Unit",
                    "notes": "Notes",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption(
            "No requirement lines recorded yet. Add segment requirements to surface shortages before execution."
        )

    active_substitution_reasons = list_inventory_substitution_reason_codes(
        conn, active_only=True
    )

    with st.expander("Warehouse execution updates", expanded=False):
        if not requirements:
            st.caption("Add requirement lines before recording pick / pack / load activity.")
        else:
            requirement_options = {
                (
                    f"Job {row['jobId']} / Segment {row['segmentSequence']} / "
                    f"{row['requirementName']} ({row['requiredQuantity']} required)"
                ): row
                for row in requirements
            }
            event_label = st.selectbox(
                "Requirement line",
                options=list(requirement_options.keys()),
                key="inventory_execution_requirement",
            )
            selected_requirement = requirement_options[event_label]
            allowed_execution_stages = get_allowed_inventory_execution_stages(
                selected_requirement.get("executionStage"),
                architecture=str(selected_requirement.get("architecture") or "general"),
            )
            st.caption(
                f"Current stage: `{selected_requirement.get('executionStage') or 'required'}`"
            )
            if allowed_execution_stages:
                st.caption("Allowed next actions: " + ", ".join(allowed_execution_stages))
            else:
                st.caption(
                    "No further routine execution actions are available for this requirement."
                )
            execution_cols = st.columns(4)
            execution_stage = execution_cols[0].selectbox(
                "Next action",
                options=allowed_execution_stages
                or [selected_requirement.get("executionStage") or "required"],
                key="inventory_execution_stage",
                disabled=not allowed_execution_stages,
            )
            execution_quantity = execution_cols[1].number_input(
                "Quantity",
                min_value=0.1,
                value=float(
                    selected_requirement.get("shortageQuantity")
                    or selected_requirement.get("requiredQuantity")
                    or 1.0
                ),
                step=0.5,
                key="inventory_execution_quantity",
            )
            execution_actor = execution_cols[2].text_input(
                "Actor",
                value="",
                key="inventory_execution_actor",
            )
            execution_truck = execution_cols[3].text_input(
                "Truck (optional)",
                value=str(selected_requirement.get("executionTruckId") or ""),
                key="inventory_execution_truck",
            )
            execution_aux_cols = st.columns(4)
            execution_container = execution_aux_cols[0].text_input(
                "Container ref",
                value=str(selected_requirement.get("executionContainerRef") or ""),
                key="inventory_execution_container",
            )
            execution_location_type = execution_aux_cols[1].selectbox(
                "Location type",
                options=[""] + list(INVENTORY_CUSTODY_TYPES),
                key="inventory_execution_location_type",
            )
            execution_location_ref = execution_aux_cols[2].text_input(
                "Location ref",
                value="",
                key="inventory_execution_location_ref",
            )
            execution_location_label = execution_aux_cols[3].text_input(
                "Location label",
                value="",
                key="inventory_execution_location_label",
            )
            execution_note = st.text_input(
                "Note",
                value="",
                key="inventory_execution_note",
            )
            if st.button(
                "Record execution update",
                type="primary",
                key="inventory_execution_save",
                disabled=not allowed_execution_stages,
            ):
                try:
                    record_inventory_execution_event(
                        conn,
                        job_id=int(selected_requirement["jobId"]),
                        segment_id=int(selected_requirement["segmentId"]),
                        requirement_id=int(selected_requirement["requirementId"]),
                        inventory_item_id=selected_requirement.get("inventoryItemId"),
                        stage=execution_stage,
                        quantity=float(execution_quantity),
                        actor=execution_actor or None,
                        note=execution_note or None,
                        container_ref=execution_container or None,
                        truck_id=execution_truck or None,
                        location_type=execution_location_type or None,
                        location_ref=execution_location_ref or None,
                        location_label=execution_location_label or None,
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to record execution update: {exc}")
                else:
                    st.success("Execution update recorded.")
                    rerun_app()

    with st.expander("Substitutions", expanded=False):
        if not requirements:
            st.caption("Add requirement lines before requesting or approving substitutions.")
        else:
            requirement_options = {
                (
                    f"Job {row['jobId']} / Segment {row['segmentSequence']} / "
                    f"{row['requirementName']} ({row['shortageQuantity']} shortage)"
                ): row
                for row in requirements
            }
            inventory_items = list_inventory(conn)
            substitution_item_options = {
                "<no substitute item selected>": None,
                **{str(row["name"]): int(row["id"]) for row in inventory_items},
            }

            request_cols = st.columns(4)
            request_requirement_label = request_cols[0].selectbox(
                "Requirement for request",
                options=list(requirement_options.keys()),
                key="inventory_substitution_requirement",
            )
            selected_requirement = requirement_options[request_requirement_label]
            request_quantity = request_cols[1].number_input(
                "Requested quantity",
                min_value=0.1,
                value=max(float(selected_requirement.get("shortageQuantity") or 0.0), 0.1),
                step=0.5,
                key="inventory_substitution_quantity",
            )
            request_actor = request_cols[2].text_input(
                "Requested by",
                value="",
                key="inventory_substitution_requested_by",
            )
            request_reason = request_cols[3].selectbox(
                "Reason code",
                options=[row["code"] for row in active_substitution_reasons]
                if active_substitution_reasons
                else ["<no active reasons>"],
                format_func=lambda code: next(
                    (
                        f"{row['label']} ({row['code']})"
                        for row in active_substitution_reasons
                        if row["code"] == code
                    ),
                    code,
                ),
                key="inventory_substitution_reason_code",
                disabled=not active_substitution_reasons,
            )
            request_substitute_label = st.selectbox(
                "Proposed substitute item",
                options=list(substitution_item_options.keys()),
                key="inventory_substitution_item",
            )
            request_note = st.text_input(
                "Request note",
                value="",
                key="inventory_substitution_note",
            )
            if st.button(
                "Request substitution",
                type="primary",
                key="inventory_substitution_request_save",
                disabled=not active_substitution_reasons,
            ):
                try:
                    request_inventory_substitution(
                        conn,
                        requirement_id=int(selected_requirement["requirementId"]),
                        requested_quantity=float(request_quantity),
                        requested_by=request_actor or None,
                        reason_code=request_reason.strip(),
                        note=request_note or None,
                        substitute_inventory_item_id=substitution_item_options[
                            request_substitute_label
                        ],
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to request substitution: {exc}")
                else:
                    st.success("Substitution request recorded.")
                    rerun_app()

            substitutions = list_inventory_substitutions(conn, job_id=job_filter)
            if substitutions:
                substitutions_df = pd.DataFrame(substitutions)
                st.dataframe(
                    substitutions_df[
                        [
                            "substitutionId",
                            "jobId",
                            "segmentId",
                            "requirementName",
                            "inventoryName",
                            "substituteInventoryName",
                            "requestedQuantity",
                            "approvedQuantity",
                            "status",
                            "requestedBy",
                            "approvedBy",
                            "approvedRole",
                            "reasonCode",
                            "note",
                            "createdAt",
                            "decidedAt",
                        ]
                    ].rename(
                        columns={
                            "substitutionId": "ID",
                            "jobId": "Job",
                            "segmentId": "Segment",
                            "requirementName": "Requirement",
                            "inventoryName": "Original item",
                            "substituteInventoryName": "Substitute item",
                            "requestedQuantity": "Requested qty",
                            "approvedQuantity": "Approved qty",
                            "status": "Status",
                            "requestedBy": "Requested by",
                            "approvedBy": "Approved by",
                            "approvedRole": "Approved role",
                            "reasonCode": "Reason code",
                            "note": "Note",
                            "createdAt": "Created",
                            "decidedAt": "Decided",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
                pending = [row for row in substitutions if row["status"] == "requested"]
                if pending:
                    pending_options = {
                        (
                            f"#{row['substitutionId']} / Job {row['jobId']} / Segment {row['segmentId']} / "
                            f"{row['requirementName']}"
                        ): row
                        for row in pending
                    }
                    decision_cols = st.columns(5)
                    pending_label = decision_cols[0].selectbox(
                        "Pending request",
                        options=list(pending_options.keys()),
                        key="inventory_substitution_pending",
                    )
                    pending_row = pending_options[pending_label]
                    decision_status = decision_cols[1].selectbox(
                        "Decision",
                        options=["approved", "rejected"],
                        key="inventory_substitution_decision",
                    )
                    approved_quantity = decision_cols[2].number_input(
                        "Approved qty",
                        min_value=0.0,
                        value=float(pending_row.get("requestedQuantity") or 0.0),
                        step=0.5,
                        key="inventory_substitution_approved_qty",
                    )
                    approved_by = decision_cols[3].text_input(
                        "Approved by",
                        value="",
                        key="inventory_substitution_approved_by",
                    )
                    approved_role = decision_cols[4].selectbox(
                        "Approval role",
                        options=list(INVENTORY_SUBSTITUTION_APPROVER_ROLES),
                        key="inventory_substitution_approved_role",
                    )
                    decision_aux_cols = st.columns(2)
                    decision_substitute_label = decision_aux_cols[0].selectbox(
                        "Decision substitute item",
                        options=list(substitution_item_options.keys()),
                        index=list(substitution_item_options.keys()).index(
                            pending_row.get("substituteInventoryName")
                            if pending_row.get("substituteInventoryName")
                            in substitution_item_options
                            else "<no substitute item selected>"
                        ),
                        key="inventory_substitution_decision_item",
                    )
                    decision_note = decision_aux_cols[1].text_input(
                        "Decision note",
                        value="",
                        key="inventory_substitution_decision_note",
                    )
                    if st.button(
                        "Apply substitution decision",
                        type="primary",
                        key="inventory_substitution_decision_save",
                    ):
                        try:
                            decide_inventory_substitution(
                                conn,
                                substitution_id=int(pending_row["substitutionId"]),
                                status=decision_status,
                                approved_by=approved_by or None,
                                approved_role=approved_role,
                                approved_quantity=(
                                    float(approved_quantity)
                                    if decision_status == "approved"
                                    else None
                                ),
                                note=decision_note or None,
                                substitute_inventory_item_id=substitution_item_options[
                                    decision_substitute_label
                                ],
                            )
                        except Exception as exc:  # pragma: no cover
                            st.error(f"Failed to apply substitution decision: {exc}")
                        else:
                            st.success("Substitution decision recorded.")
                            rerun_app()
            else:
                st.caption("No substitution requests recorded yet.")

        with st.expander("Substitution reason governance", expanded=False):
            all_reasons = list_inventory_substitution_reason_codes(conn, active_only=False)
            if all_reasons:
                st.dataframe(
                    pd.DataFrame(all_reasons)[
                        ["code", "label", "description", "active", "systemSeeded"]
                    ].rename(
                        columns={
                            "code": "Code",
                            "label": "Label",
                            "description": "Description",
                            "active": "Active",
                            "systemSeeded": "Seeded",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
            reason_cols = st.columns(4)
            reason_code = reason_cols[0].text_input("Code", key="inventory_reason_code")
            reason_label = reason_cols[1].text_input("Label", key="inventory_reason_label")
            reason_description = reason_cols[2].text_input(
                "Description",
                key="inventory_reason_description",
            )
            reason_active = reason_cols[3].checkbox(
                "Active",
                value=True,
                key="inventory_reason_active",
            )
            if st.button("Save reason code", key="inventory_reason_save"):
                try:
                    upsert_inventory_substitution_reason_code(
                        conn,
                        code=reason_code,
                        label=reason_label,
                        description=reason_description or None,
                        active=bool(reason_active),
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to save substitution reason code: {exc}")
                else:
                    st.success("Substitution reason code saved.")
                    rerun_app()

    execution_events = list_inventory_execution_events(conn, job_id=job_filter, limit=50)
    if execution_events:
        st.markdown("#### Recent execution events")
        events_df = pd.DataFrame(execution_events)
        st.dataframe(
            events_df[
                [
                    "jobId",
                    "segmentId",
                    "requirementName",
                    "inventoryName",
                    "stage",
                    "quantity",
                    "actor",
                    "containerRef",
                    "truckId",
                    "locationType",
                    "locationLabel",
                    "note",
                    "createdAt",
                ]
            ].rename(
                columns={
                    "jobId": "Job",
                    "segmentId": "Segment",
                    "requirementName": "Requirement",
                    "inventoryName": "Inventory item",
                    "stage": "Stage",
                    "quantity": "Qty",
                    "actor": "Actor",
                    "containerRef": "Container",
                    "truckId": "Truck",
                    "locationType": "Location type",
                    "locationLabel": "Location",
                    "note": "Note",
                    "createdAt": "Recorded",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    balances = list_inventory_balances(conn, job_id=job_filter, states=state_filter or None)
    balances_df = pd.DataFrame(balances)
    if balances_df.empty:
        st.info("No inventory items found. Import items to begin tracking balances.")
    else:
        display_columns = [
            "name",
            "state",
            "job_id",
            "on_hand_quantity",
            "allocated_quantity",
            "available_quantity",
            "architecture",
            "custody_location_type",
            "custody_location_label",
            "unit",
            "updated_at",
        ]
        present_columns = [col for col in display_columns if col in balances_df.columns]
        st.dataframe(balances_df[present_columns], width="stretch")

    with st.expander("Import inventory items", expanded=False):
        items_file = st.file_uploader(
            "Upload CSV or Excel for inventory items",
            type=["csv", "xlsx", "xls"],
            key="inventory_items_upload",
        )
        if st.button(
            "Import items",
            type="primary",
            disabled=items_file is None,
            key="inventory_items_import_button",
        ):
            try:
                df = _read_uploaded_inventory_file(items_file)
                imported = import_inventory_items_from_dataframe(conn, df)
            except Exception as exc:  # pragma: no cover - surfaced in UI
                st.error(f"Failed to import inventory items: {exc}")
            else:
                st.success(f"Imported or refreshed {imported} inventory rows.")
                rerun_app()

    with st.expander("Import suppliers from Google Sheets", expanded=False):
        suppliers_sheet_reference = st.text_input(
            "Operations workbook ID or URL",
            value=_default_operations_sheet_reference(),
            help="Shared operations workbook containing the SUPPLIERS tab.",
            key="suppliers_sheet_reference",
        )
        suppliers_sheet_name = st.text_input(
            "Supplier tab name",
            value="SUPPLIERS",
            key="suppliers_sheet_name",
        )
        if st.button(
            "Import suppliers",
            type="primary",
            disabled=not suppliers_sheet_reference.strip(),
            key="suppliers_import_button",
        ):
            try:
                imported = import_suppliers_from_google_sheet(
                    conn,
                    sheet_id=suppliers_sheet_reference.strip(),
                    sheet_name=suppliers_sheet_name.strip() or "SUPPLIERS",
                )
            except Exception as exc:  # pragma: no cover - surfaced in UI
                st.error(f"Failed to import suppliers: {exc}")
            else:
                st.success(f"Imported or refreshed {imported} suppliers.")
                rerun_app()

    with st.expander("Import movement events", expanded=False):
        movements_file = st.file_uploader(
            "Upload CSV or Excel for movement events",
            type=["csv", "xlsx", "xls"],
            key="inventory_movements_upload",
        )
        default_reason = st.text_input(
            "Default reason (optional)",
            value="",
            help="Applied when the upload does not specify a reason column.",
        )
        if st.button(
            "Import movements",
            type="primary",
            disabled=movements_file is None,
            key="inventory_movements_import_button",
        ):
            try:
                df = _read_uploaded_inventory_file(movements_file)
                imported = import_inventory_movements_from_dataframe(
                    conn, df, default_reason=default_reason or None
                )
            except Exception as exc:  # pragma: no cover - surfaced in UI
                st.error(f"Failed to import movement events: {exc}")
            else:
                st.success(f"Recorded {imported} movement events.")
                rerun_app()

    with st.expander("Plan inventory requirements", expanded=False):
        segment_options = {
            f"Job {row['jobId']} / Segment {row['segmentSequence']}": int(row["segmentId"])
            for row in segment_coordination
        }
        item_rows = list_inventory(conn)
        item_lookup = {f"{row['name']}": int(row["id"]) for row in item_rows}
        if not segment_options:
            st.caption(
                "Need at least one planned segment before defining inventory requirements."
            )
        else:
            req_cols = st.columns(4)
            segment_label = req_cols[0].selectbox(
                "Target segment",
                options=list(segment_options.keys()),
                key="inventory_requirement_segment",
            )
            selected_segment_id = segment_options[segment_label]
            selected_segment = next(
                row
                for row in segment_coordination
                if int(row["segmentId"]) == int(selected_segment_id)
            )
            item_label = req_cols[1].selectbox(
                "Inventory item (optional)",
                options=["<generic requirement>"] + list(item_lookup.keys()),
                key="inventory_requirement_item",
            )
            architecture = req_cols[2].selectbox(
                "Architecture",
                options=list(INVENTORY_ARCHITECTURES),
                index=list(INVENTORY_ARCHITECTURES).index("container"),
                key="inventory_requirement_architecture",
            )
            substitution_allowed = req_cols[3].checkbox(
                "Substitution allowed",
                value=False,
                key="inventory_requirement_substitution",
            )
            req_name_default = item_label if item_label != "<generic requirement>" else ""
            requirement_name = st.text_input(
                "Requirement name",
                value=req_name_default,
                key="inventory_requirement_name",
            )
            qty_cols = st.columns(2)
            required_quantity = qty_cols[0].number_input(
                "Required quantity",
                min_value=0.1,
                value=1.0,
                step=0.5,
                key="inventory_requirement_quantity",
            )
            requirement_notes = qty_cols[1].text_input(
                "Notes",
                value="",
                key="inventory_requirement_notes",
            )
            if st.button("Save requirement", type="primary", key="inventory_requirement_save"):
                try:
                    upsert_inventory_requirement(
                        conn,
                        job_id=int(selected_segment["jobId"]),
                        segment_id=int(selected_segment_id),
                        inventory_item_id=(
                            item_lookup.get(item_label)
                            if item_label != "<generic requirement>"
                            else None
                        ),
                        requirement_name=requirement_name,
                        required_quantity=float(required_quantity),
                        substitution_allowed=bool(substitution_allowed),
                        architecture=architecture,
                        notes=requirement_notes or None,
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to save inventory requirement: {exc}")
                else:
                    st.success("Inventory requirement saved.")
                    rerun_app()

    with st.expander("Reserve or release stock", expanded=False):
        if balances_df.empty:
            st.caption("Add inventory items to enable reservations and releases.")
        else:
            option_labels = {
                f"{row['name']} ({row['available_quantity']} available)": int(row["id"])
                for row in balances
            }
            selected_label = st.selectbox(
                "Inventory item",
                options=list(option_labels.keys()),
                key="inventory_reservation_item",
            )
            quantity = st.number_input(
                "Quantity", min_value=1, step=1, value=1, key="inventory_reservation_qty"
            )
            target_state = st.selectbox(
                "Set state",
                INVENTORY_STATES,
                index=INVENTORY_STATES.index("staged"),
                key="inventory_reservation_state",
            )

            item_id = option_labels.get(selected_label)
            cols = st.columns(2)
            with cols[0]:
                if st.button("Reserve allocation", type="primary"):
                    record_inventory_movement(
                        conn,
                        inventory_item_id=int(item_id),
                        change_allocated=int(quantity),
                        state=target_state,
                        job_id=job_filter,
                    )
                    st.success("Reserved stock and updated state.")
                    rerun_app()
            with cols[1]:
                if st.button("Release allocation"):
                    record_inventory_movement(
                        conn,
                        inventory_item_id=int(item_id),
                        change_allocated=-int(quantity),
                        state=target_state,
                        job_id=job_filter,
                    )
                    st.success("Released stock and updated state.")
                    rerun_app()

    with st.expander("Update custody / location", expanded=False):
        if balances_df.empty:
            st.caption("Add inventory items before recording custody/location changes.")
        else:
            custody_options = {
                f"{row['name']} ({row['available_quantity']} available)": int(row["id"])
                for row in balances
            }
            custody_label = st.selectbox(
                "Inventory item for custody update",
                options=list(custody_options.keys()),
                key="inventory_custody_item",
            )
            custody_cols = st.columns(3)
            location_type = custody_cols[0].selectbox(
                "Location type",
                options=list(INVENTORY_CUSTODY_TYPES),
                key="inventory_custody_type",
            )
            location_ref = custody_cols[1].text_input(
                "Location reference",
                value="",
                key="inventory_custody_ref",
            )
            location_label = custody_cols[2].text_input(
                "Location label",
                value="",
                key="inventory_custody_label",
            )
            custody_state = st.selectbox(
                "State",
                options=list(INVENTORY_STATES),
                index=list(INVENTORY_STATES).index("staged"),
                key="inventory_custody_state",
            )
            if st.button("Record custody update", type="primary", key="inventory_custody_save"):
                try:
                    record_inventory_movement(
                        conn,
                        inventory_item_id=int(custody_options[custody_label]),
                        reason="custody_update",
                        state=custody_state,
                        job_id=job_filter,
                        location_type=location_type,
                        location_ref=location_ref or None,
                        location_label=location_label or None,
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to record custody update: {exc}")
                else:
                    st.success("Custody/location updated.")
                    rerun_app()

    with st.expander("Allocate inventory to planned segment", expanded=False):
        segment_options = {
            f"Job {row['jobId']} / Segment {row['segmentSequence']}": int(row["segmentId"])
            for row in segment_coordination
        }
        item_options = (
            {
                f"{row['name']} ({row['available_quantity']} available)": int(row["id"])
                for row in balances
            }
            if balances
            else {f"{row['name']}": int(row["id"]) for row in list_inventory(conn)}
        )
        if not segment_options or not item_options:
            st.caption(
                "Need at least one planned segment and one inventory item before allocating stock."
            )
        else:
            segment_label = st.selectbox(
                "Target segment",
                options=list(segment_options.keys()),
                key="inventory_segment_target",
            )
            item_label = st.selectbox(
                "Inventory item for segment",
                options=list(item_options.keys()),
                key="inventory_segment_item",
            )
            alloc_quantity = st.number_input(
                "Allocation quantity",
                min_value=0.1,
                value=1.0,
                step=0.5,
                key="inventory_segment_quantity",
            )
            alloc_status = st.selectbox(
                "Shipment status",
                options=["planned", "staged", "loaded", "in_transit"],
                index=0,
                key="inventory_segment_status",
            )
            if st.button(
                "Allocate to segment",
                type="primary",
                key="inventory_segment_allocate_button",
            ):
                try:
                    allocate_inventory_to_segment(
                        conn,
                        segment_id=segment_options[segment_label],
                        inventory_item_id=item_options[item_label],
                        quantity=float(alloc_quantity),
                        status=alloc_status,
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to allocate inventory to segment: {exc}")
                else:
                    st.success("Inventory allocated to segment.")
                    rerun_app()

    with st.expander("Recent movements", expanded=True):
        movements = list_inventory_movements(
            conn, limit=100, job_id=job_filter, states=state_filter or None
        )
        movements_df = pd.DataFrame(movements)
        if movements_df.empty:
            st.caption("No movement history available for the current filters.")
        else:
            display_columns = [
                "inventory_name",
                "movement_state",
                "job_id",
                "change_on_hand",
                "change_allocated",
                "reason",
                "location_type_value",
                "location_label_value",
                "sequence_no",
                "created_at",
            ]
            present_columns = [col for col in display_columns if col in movements_df.columns]
            st.dataframe(movements_df[present_columns], width="stretch")

    with st.expander("Inventory exceptions", expanded=True):
        exceptions = list_inventory_exceptions(conn, resolved=False)
        if not exceptions:
            st.caption("No outstanding exceptions detected by reconciliation jobs.")
        else:
            for exception in exceptions:
                cols = st.columns([4, 1])
                with cols[0]:
                    st.markdown(
                        f"**Item:** {exception.get('inventory_name') or 'Unknown'}  \\\n"
                        f"**State:** {exception.get('state') or 'n/a'}  \\\n"
                        f"**Job:** {exception.get('job_id') or exception.get('inventory_job_id') or 'n/a'}"
                    )
                    st.caption(exception.get("notes") or "No notes recorded.")
                with cols[1]:
                    if st.button(
                        "Reconcile",
                        key=f"inventory_exception_{exception['id']}",
                    ):
                        resolve_inventory_exception(
                            conn,
                            exception_id=int(exception["id"]),
                            note="Reconciled via dashboard",
                        )
                        st.success("Exception marked as reconciled.")
                        rerun_app()
