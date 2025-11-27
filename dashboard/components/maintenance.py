from __future__ import annotations

import os
from typing import Iterable, Optional

import pandas as pd
import streamlit as st

from analytics.db import ensure_dashboard_tables, upsert_truck, upsert_vehicle_details
from analytics.vehicle_repairs import (
    import_vehicle_repairs_from_sheet,
    load_vehicle_repairs,
)
from analytics.vehicle_workbook import (
    import_vehicle_details_from_dataframe,
    import_vehicle_details_from_google_sheet,
)


def _trigger_rerun() -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return

    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()


def _format_currency(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "$0"
    return f"${value:,.0f}"


def load_vehicle_overview(conn) -> pd.DataFrame:
    ensure_dashboard_tables(conn)
    query = """
        SELECT
            t.truck_id,
            t.name,
            t.capacity_m3,
            t.active,
            t.notes,
            vd.state,
            vd.rego,
            vd.rego_expiry,
            vd.make,
            vd.model,
            vd.year,
            vd.body_type,
            vd.description,
            vd.nhv_code,
            vd.insurance,
            vd.odometer,
            vd.last_service,
            vd.next_service,
            vd.coi_number,
            vd.coi_due,
            vd.present_driver,
            vd.daily_check_complete
        FROM trucks AS t
        LEFT JOIN vehicle_details AS vd ON vd.truck_id = t.truck_id
        ORDER BY t.truck_id
    """
    return pd.read_sql_query(query, conn)


def render_vehicle_maintenance_tab(conn) -> None:
    """Render the maintenance history tab."""

    st.markdown("### Vehicle maintenance history")
    st.caption("Import workshop visits from Google Sheets and keep a running log of spend per truck.")

    vehicle_overview = load_vehicle_overview(conn)
    if vehicle_overview.empty:
        st.info("No vehicles found. Use the Fleet tab to add trucks or import VEHICLE_DETAILS data.")
    else:
        metadata_columns: Iterable[str] = (
            "truck_id",
            "rego",
            "rego_expiry",
            "insurance",
            "odometer",
            "last_service",
            "next_service",
            "present_driver",
            "daily_check_complete",
        )
        st.markdown("#### Vehicle register")
        vehicle_preview = vehicle_overview.loc[:, metadata_columns].copy()
        vehicle_preview = vehicle_preview.rename(
            columns={
                "truck_id": "Vehicle",
                "rego": "Rego",
                "rego_expiry": "Rego expiry",
                "insurance": "Insurance",
                "odometer": "Odometer",
                "last_service": "Last service",
                "next_service": "Next service",
                "present_driver": "Assigned driver",
                "daily_check_complete": "Daily check complete",
            }
        )
        vehicle_preview["Daily check complete"] = vehicle_preview["Daily check complete"].map(
            {1: "Yes", 0: "No"}
        )
        st.dataframe(vehicle_preview, use_container_width=True)

    default_sheet = os.environ.get("VEHICLE_REPAIRS_SHEET_URL") or os.environ.get(
        "VEHICLE_REPAIRS_SHEET"
    )

    with st.expander("Import VEHICLE_REPAIRS sheet", expanded=False):
        sheet_url = st.text_input(
            "Google Sheets CSV/Excel URL",
            value=default_sheet or "",
            help=(
                "Paste the CSV export link for the VEHICLE_REPAIRS sheet. "
                "Both CSV and XLSX feeds are supported."
            ),
            key="vehicle_repairs_sheet_url",
        )
        if st.button("Import vehicle repairs", key="vehicle_repairs_import_button"):
            if not sheet_url:
                st.error("Provide a Google Sheets URL before importing.")
            else:
                try:
                    inserted, updated = import_vehicle_repairs_from_sheet(
                        conn, sheet_url=sheet_url
                    )
                except Exception as exc:  # pragma: no cover - surfaced in UI only
                    st.error(f"Failed to import repairs: {exc}")
                else:
                    st.success(
                        f"Imported {inserted} new record{'s' if inserted != 1 else ''}. "
                        f"Updated {updated} existing row{'s' if updated != 1 else ''}."
                    )

    repairs_df = load_vehicle_repairs(conn)
    if repairs_df.empty:
        st.info(
            "No vehicle repairs captured yet. Import the VEHICLE_REPAIRS sheet to populate this view."
        )
        return

    display_df = repairs_df.copy()
    for column in ("service_date", "next_service_date", "created_at", "updated_at"):
        if column in display_df.columns:
            display_df[column] = pd.to_datetime(display_df[column], errors="coerce")

    display_df = display_df.sort_values(
        by=["service_date", "created_at"], ascending=[False, False], na_position="last"
    )

    st.markdown("#### Spend by vehicle")
    spend_summary = (
        display_df.assign(price_numeric=pd.to_numeric(display_df["price"], errors="coerce"))
        .groupby("truck_id")
        .agg(
            total_spend=pd.NamedAgg(column="price_numeric", aggfunc="sum"),
            jobs=pd.NamedAgg(column="job_item", aggfunc="count"),
            last_service=pd.NamedAgg(column="service_date", aggfunc="max"),
        )
        .reset_index()
        .sort_values("total_spend", ascending=False)
    )

    summary_cols = st.columns(max(1, min(3, len(spend_summary))))
    for idx, row in spend_summary.iterrows():
        column = summary_cols[idx % len(summary_cols)]
        column.metric(
            f"{row['truck_id']} spend",
            _format_currency(row["total_spend"]),
            help="Total repair spend captured for this vehicle.",
        )
        column.metric(
            f"{row['truck_id']} jobs",
            f"{int(row['jobs'])}",
            help="Count of repair log entries.",
        )
        if pd.notna(row["last_service"]):
            column.caption(f"Last service: {pd.to_datetime(row['last_service']).date()}")

    st.dataframe(
        spend_summary.rename(
            columns={
                "truck_id": "Vehicle",
                "total_spend": "Spend",
                "jobs": "Repairs logged",
                "last_service": "Most recent service",
            }
        ),
        use_container_width=True,
    )

    st.markdown("#### Repair log")
    if not vehicle_overview.empty:
        metadata_fields = [
            "rego",
            "insurance",
            "odometer",
            "last_service",
            "next_service",
            "present_driver",
            "daily_check_complete",
        ]
        merged = display_df.merge(
            vehicle_overview[["truck_id", *metadata_fields]],
            on="truck_id",
            how="left",
        )
        merged = merged.rename(
            columns={
                "rego": "Rego",
                "insurance": "Insurance",
                "odometer": "Odometer",
                "last_service": "Last service marker",
                "next_service": "Next service marker",
                "present_driver": "Assigned driver",
                "daily_check_complete": "Daily check complete",
            }
        )
        merged["Daily check complete"] = merged["Daily check complete"].map({1: "Yes", 0: "No"})
        st.dataframe(merged, use_container_width=True)
    else:
        st.dataframe(display_df, use_container_width=True)


def _parse_sheet_id(sheet_reference: str) -> str:
    sheet_reference = sheet_reference.strip()
    if "/d/" in sheet_reference:
        parts = sheet_reference.split("/d/")
        if len(parts) > 1:
            remainder = parts[1]
            return remainder.split("/")[0]
    return sheet_reference


def render_fleet_tab(conn) -> None:
    st.markdown("### Fleet register")
    st.caption("Manage trucks, vehicle metadata, and VEHICLE_DETAILS imports.")

    ensure_dashboard_tables(conn)
    vehicle_df = load_vehicle_overview(conn)

    with st.expander("Import/Export VEHICLE_DETAILS", expanded=False):
        sheet_input = st.text_input(
            "Google Sheets ID or URL",
            value="",
            help="Paste the spreadsheet ID or full link for the VEHICLE_DETAILS workbook.",
        )
        upload = st.file_uploader(
            "Upload VEHICLE_DETAILS workbook (XLSX or CSV)",
            type=["xlsx", "xls", "csv"],
        )
        import_cols = st.columns(2)
        if import_cols[0].button("Import from Google Sheet"):
            try:
                sheet_id = _parse_sheet_id(sheet_input)
                imported = import_vehicle_details_from_google_sheet(conn, sheet_id=sheet_id)
            except Exception as exc:  # pragma: no cover - UI feedback only
                st.error(f"Failed to import VEHICLE_DETAILS: {exc}")
            else:
                st.success(f"Imported {imported} vehicle{'s' if imported != 1 else ''} from Google Sheets.")
                _trigger_rerun()

        if import_cols[1].button("Import uploaded workbook"):
            if not upload:
                st.error("Upload a workbook before importing.")
            else:
                try:
                    if upload.name.endswith(".csv"):
                        frame = pd.read_csv(upload)
                    else:
                        frame = pd.read_excel(upload)
                    imported = import_vehicle_details_from_dataframe(conn, frame)
                except Exception as exc:  # pragma: no cover - UI feedback only
                    st.error(f"Failed to import uploaded workbook: {exc}")
                else:
                    st.success(f"Imported {imported} vehicle{'s' if imported != 1 else ''} from the upload.")
                    _trigger_rerun()

        if not vehicle_df.empty:
            csv_data = vehicle_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download VEHICLE_DETAILS CSV",
                data=csv_data,
                file_name="vehicle_details.csv",
                mime="text/csv",
            )
        else:
            st.caption("Export will be available after vehicles are added.")

    filter_cols = st.columns(2)
    truck_filter = filter_cols[0].multiselect(
        "Filter by vehicle",
        options=sorted(vehicle_df["truck_id"].dropna().unique()) if not vehicle_df.empty else [],
    )
    active_filter = filter_cols[1].selectbox(
        "Active status",
        options=["All", "Active", "Inactive"],
        help="Quick filter to show only active or inactive trucks.",
    )

    filtered_df = vehicle_df
    if truck_filter:
        filtered_df = filtered_df[filtered_df["truck_id"].isin(truck_filter)]
    if active_filter != "All":
        desired = 1 if active_filter == "Active" else 0
        filtered_df = filtered_df[filtered_df["active"] == desired]

    st.markdown("#### Vehicles")
    if filtered_df.empty:
        st.info("No vehicles match the selected filters.")
    else:
        preview = filtered_df.copy()
        preview["active"] = preview["active"].map({1: "Yes", 0: "No"})
        st.dataframe(preview, use_container_width=True)

    st.markdown("#### Add or update vehicle")
    existing_ids = list(vehicle_df["truck_id"].dropna().unique()) if not vehicle_df.empty else []
    selection_label = "Select vehicle" if existing_ids else "New vehicle"
    selection_options = ["New vehicle", *existing_ids]
    selected_vehicle = st.selectbox(selection_label, options=selection_options)

    defaults: dict[str, object] = {}
    if selected_vehicle != "New vehicle" and not vehicle_df.empty:
        defaults = (
            vehicle_df.loc[vehicle_df["truck_id"] == selected_vehicle]
            .iloc[0]
            .to_dict()
        )

    with st.form("vehicle_editor"):
        truck_id_value = st.text_input("Truck ID (rego)", value=str(defaults.get("truck_id", "")))
        name_value = st.text_input("Name/label", value=str(defaults.get("name", "") or ""))
        capacity_value = st.number_input(
            "Capacity (m³)",
            min_value=0.0,
            value=float(defaults.get("capacity_m3")) if defaults.get("capacity_m3") is not None else 0.0,
            step=1.0,
        )
        active_value = st.checkbox("Active", value=bool(defaults.get("active", True)))
        notes_value = st.text_area("Notes", value=str(defaults.get("notes", "") or ""))

        st.markdown("##### Vehicle details")
        detail_cols1 = st.columns(3)
        state_value = detail_cols1[0].text_input("State", value=str(defaults.get("state", "") or ""))
        rego_expiry_default = pd.to_datetime(
            defaults.get("rego_expiry"), errors="coerce"
        )
        rego_expiry_value = detail_cols1[1].date_input(
            "Rego expiry",
            value=rego_expiry_default.date() if pd.notna(rego_expiry_default) else None,
        )
        insurance_value = detail_cols1[2].text_input("Insurance", value=str(defaults.get("insurance", "") or ""))

        detail_cols2 = st.columns(3)
        make_value = detail_cols2[0].text_input("Make", value=str(defaults.get("make", "") or ""))
        model_value = detail_cols2[1].text_input("Model", value=str(defaults.get("model", "") or ""))
        year_value = detail_cols2[2].number_input(
            "Year", min_value=0, max_value=9999, value=int(defaults.get("year")) if defaults.get("year") else 0, step=1
        )

        body_type_value = st.text_input("Body type", value=str(defaults.get("body_type", "") or ""))
        description_value = st.text_area("Description", value=str(defaults.get("description", "") or ""))
        nhv_code_value = st.text_input("NHV code", value=str(defaults.get("nhv_code", "") or ""))
        odometer_value = st.number_input(
            "Odometer", min_value=0, value=int(defaults.get("odometer")) if defaults.get("odometer") else 0, step=100
        )
        detail_cols3 = st.columns(2)
        last_service_default = pd.to_datetime(defaults.get("last_service"), errors="coerce")
        next_service_default = pd.to_datetime(defaults.get("next_service"), errors="coerce")
        last_service_value = detail_cols3[0].date_input(
            "Last service",
            value=last_service_default.date() if pd.notna(last_service_default) else None,
        )
        next_service_value = detail_cols3[1].date_input(
            "Next service",
            value=next_service_default.date() if pd.notna(next_service_default) else None,
        )
        detail_cols4 = st.columns(2)
        coi_number_value = detail_cols4[0].text_input("COI number", value=str(defaults.get("coi_number", "") or ""))
        coi_due_default = pd.to_datetime(defaults.get("coi_due"), errors="coerce")
        coi_due_value = detail_cols4[1].date_input(
            "COI due", value=coi_due_default.date() if pd.notna(coi_due_default) else None
        )
        present_driver_value = st.text_input("Assigned driver", value=str(defaults.get("present_driver", "") or ""))
        daily_check_value = st.checkbox(
            "Daily check complete", value=bool(defaults.get("daily_check_complete", False))
        )

        submitted = st.form_submit_button("Save vehicle")

    if submitted:
        if not truck_id_value.strip():
            st.error("Truck ID/rego is required.")
        else:
            upsert_truck(
                conn,
                truck_id=truck_id_value.strip(),
                name=name_value or None,
                capacity_m3=capacity_value if capacity_value else None,
                active=active_value,
                notes=notes_value or None,
            )
            upsert_vehicle_details(
                conn,
                truck_id=truck_id_value.strip(),
                state=state_value or None,
                rego=truck_id_value.strip(),
                rego_expiry=rego_expiry_value.isoformat() if rego_expiry_value else None,
                make=make_value or None,
                model=model_value or None,
                year=int(year_value) if year_value else None,
                body_type=body_type_value or None,
                description=description_value or None,
                nhv_code=nhv_code_value or None,
                insurance=insurance_value or None,
                odometer=int(odometer_value) if odometer_value else None,
                last_service=last_service_value.isoformat() if last_service_value else None,
                next_service=next_service_value.isoformat() if next_service_value else None,
                coi_number=coi_number_value or None,
                coi_due=coi_due_value.isoformat() if coi_due_value else None,
                present_driver=present_driver_value or None,
                daily_check_complete=daily_check_value,
            )
            st.success("Vehicle saved.")
            _trigger_rerun()

    if selected_vehicle != "New vehicle" and selected_vehicle in existing_ids:
        if st.button("Delete vehicle", type="secondary"):
            conn.execute("DELETE FROM trucks WHERE truck_id = ?", (selected_vehicle,))
            conn.commit()
            st.success(f"Deleted vehicle {selected_vehicle}.")
            _trigger_rerun()
