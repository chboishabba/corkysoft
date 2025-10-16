"""Data preparation helpers for the Streamlit dashboard."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import streamlit as st

from analytics.db import connection_scope, ensure_dashboard_tables
from analytics.price_distribution import (
    ColumnMapping,
    ensure_break_even_parameter,
    import_historical_jobs_from_dataframe,
    load_historical_jobs,
    load_live_jobs,
    load_quotes,
    update_break_even,
)
from corkysoft.repo import ensure_schema as ensure_quote_schema
from corkysoft.schema import ensure_schema as ensure_core_schema

DatasetLoader = Callable[..., Tuple[pd.DataFrame, ColumnMapping]]


@dataclass
class PreparedDashboardData:
    """Snapshot of the dataset preparation phase."""

    conn: sqlite3.Connection
    break_even_value: float
    dataset_error: Optional[str]
    empty_dataset_message: Optional[str]
    data_available: bool
    mapping: ColumnMapping
    filtered_mapping: ColumnMapping
    filtered_df: pd.DataFrame
    has_filtered_data: bool


def _blank_column_mapping() -> ColumnMapping:
    return ColumnMapping(
        date=None,
        client=None,
        price=None,
        revenue=None,
        volume=None,
        origin=None,
        destination=None,
        corridor=None,
        distance=None,
        final_cost=None,
    )


def _initial_dataset_loader() -> Tuple[str, DatasetLoader]:
    return "historical", load_historical_jobs


def _resolve_dataset_choice() -> Dict[str, Tuple[str, DatasetLoader]]:
    return {
        "Historical quotes": ("historical", load_historical_jobs),
        "Saved quick quotes": ("quotes", load_quotes),
        "Live jobs": ("live", load_live_jobs),
    }


def _handle_historical_import(
    conn: sqlite3.Connection,
) -> Optional[Tuple[str, str]]:
    with st.expander("Import historical jobs from CSV", expanded=False):
        import_form = st.form(key="historical_import_form")
        uploaded_file = import_form.file_uploader(
            "Select CSV file",
            type=["csv"],
            help="Requires headers such as date, origin, destination and m3.",
        )
        submit_import = import_form.form_submit_button("Import jobs")
        if not submit_import:
            return None
        if uploaded_file is None:
            return ("warning", "Choose a CSV file before importing.")
        try:
            imported_df = pd.read_csv(uploaded_file)
        except Exception as exc:  # pragma: no cover - streamlit UI feedback only
            return ("error", f"Failed to read CSV: {exc}")
        try:
            inserted, skipped_rows = import_historical_jobs_from_dataframe(
                conn, imported_df
            )
        except ValueError as exc:
            return ("error", str(exc))
        except Exception as exc:  # pragma: no cover - surfaced to the UI
            return ("error", f"Failed to import historical jobs: {exc}")
        if inserted:
            message = (
                f"Imported {inserted} historical job"
                f"{'s' if inserted != 1 else ''}."
            )
            if skipped_rows:
                message += (
                    f" Skipped {skipped_rows} row"
                    f"{'s' if skipped_rows != 1 else ''} with missing or duplicate data."
                )
            return ("success", message)
        if skipped_rows:
            message = (
                "No new rows imported. Skipped "
                f"{skipped_rows} row{'s' if skipped_rows != 1 else ''} due to validation or duplicates."
            )
        else:
            message = "No rows imported from the provided file."
        return ("warning", message)


def _apply_import_feedback(feedback: Optional[Tuple[str, str]]) -> None:
    if not feedback:
        return
    level, message = feedback
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.info(message)
    else:
        st.error(message)


def _load_dataset(
    conn: sqlite3.Connection,
    loader: DatasetLoader,
    dataset_label: str,
) -> Tuple[pd.DataFrame, ColumnMapping, Optional[str]]:
    try:
        df, mapping = loader(conn)
    except RuntimeError as exc:
        return pd.DataFrame(), _blank_column_mapping(), str(exc)
    except Exception as exc:
        return (
            pd.DataFrame(),
            _blank_column_mapping(),
            f"Failed to load {dataset_label.lower()} data: {exc}",
        )
    return df, mapping, None


def _sidebar_filters(
    conn: sqlite3.Connection,
    break_even_value: float,
) -> Tuple[
    float,
    pd.DataFrame,
    ColumnMapping,
    DatasetLoader,
    Optional[str],
    Optional[date],
    Optional[date],
    Optional[str],
    List[str],
    Optional[str],
    Optional[str],
]:
    dataset_key, dataset_loader = _initial_dataset_loader()
    dataset_label = "Historical quotes"
    dataset_error: Optional[str] = None
    empty_dataset_message: Optional[str] = None
    df_all: pd.DataFrame = pd.DataFrame()
    mapping: ColumnMapping = _blank_column_mapping()

    start_date: Optional[date] = None
    end_date: Optional[date] = None
    selected_corridor: Optional[str] = None
    selected_clients: List[str] = []
    postcode_prefix: Optional[str] = None

    with st.sidebar:
        st.header("Filters")
        if st.button(
            "Initialise database tables",
            help=(
                "Create empty historical and live job tables so the dashboard can run "
                "before data imports."
            ),
        ):
            ensure_core_schema(conn)
            ensure_dashboard_tables(conn)
            ensure_quote_schema(conn)
            st.success(
                "Database tables initialised. Import data or start building quotes below."
            )

        dataset_options = _resolve_dataset_choice()
        dataset_label = st.radio(
            "Dataset",
            options=list(dataset_options.keys()),
            format_func=lambda label: label,
        )
        dataset_key, dataset_loader = dataset_options[dataset_label]

        import_feedback: Optional[Tuple[str, str]] = None
        if dataset_key == "historical":
            import_feedback = _handle_historical_import(conn)

        df_all, mapping, dataset_error = _load_dataset(
            conn, dataset_loader, dataset_label
        )

        _apply_import_feedback(import_feedback)

        data_available = dataset_error is None and not df_all.empty

        today_value = date.today()
        date_column = "job_date" if "job_date" in df_all.columns else mapping.date
        if data_available and date_column and date_column in df_all.columns:
            df_all[date_column] = pd.to_datetime(df_all[date_column], errors="coerce")
            min_date = df_all[date_column].min()
            max_date = df_all[date_column].max()
            default_start = (
                min_date.date() if isinstance(min_date, pd.Timestamp) else today_value
            )
            default_end = (
                max_date.date() if isinstance(max_date, pd.Timestamp) else today_value
            )
            date_range = st.date_input(
                "Date range",
                value=(default_start, default_end),
                min_value=default_start,
                max_value=default_end,
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = default_start
                end_date = default_end
        else:
            st.date_input(
                "Date range",
                value=(today_value, today_value),
                disabled=True,
            )
            start_date = None
            end_date = None

        corridor_options: List[str] = []
        if data_available:
            corridor_series = df_all.get("corridor_display")
            if corridor_series is not None:
                corridor_options = sorted(
                    pd.Series(corridor_series).dropna().astype(str).unique().tolist()
                )
        corridor_selection = st.selectbox(
            "Corridor",
            options=["All corridors"] + corridor_options,
            index=0,
            disabled=not data_available,
        )
        selected_corridor = (
            None if corridor_selection == "All corridors" else corridor_selection
        )

        client_options: List[str] = []
        if data_available:
            client_series = df_all.get("client_display")
            if client_series is not None:
                client_options = sorted(
                    pd.Series(client_series).dropna().astype(str).unique().tolist()
                )
        selected_clients = st.multiselect(
            "Client",
            options=client_options,
            default=client_options if client_options else [],
            disabled=not data_available,
        )

        postcode_prefix = st.text_input(
            "Corridor contains postcode prefix",
            value=postcode_prefix or "",
            disabled=not data_available,
            help="Match origin or destination postcode prefixes (e.g. 40 to match 4000-4099).",
        ) or None

        if dataset_error:
            st.error(dataset_error)
        elif not data_available:
            empty_messages = {
                "historical": (
                    "historical_jobs table has no rows yet. Import historical jobs to populate the view."
                ),
                "quotes": (
                    "quotes table has no rows yet. Save a quick quote to populate the view."
                ),
                "live": "jobs table has no rows yet. Add live jobs to populate the view.",
            }
            empty_dataset_message = empty_messages.get(
                dataset_key, "No rows available for the selected dataset."
            )
            st.info(empty_dataset_message)

        st.subheader("Break-even model")
        new_break_even = st.number_input(
            "Break-even $/m³",
            min_value=0.0,
            value=float(break_even_value),
            step=5.0,
            help="Used to draw break-even bands on the histogram.",
        )
        if st.button("Update break-even"):
            update_break_even(conn, new_break_even)
            st.success(f"Break-even updated to ${new_break_even:,.2f}")
            break_even_value = new_break_even

    return (
        break_even_value,
        df_all,
        mapping,
        dataset_loader,
        dataset_error,
        start_date,
        end_date,
        selected_corridor,
        selected_clients,
        postcode_prefix,
        empty_dataset_message,
    )


def _prepare_dashboard_data(conn: sqlite3.Connection) -> PreparedDashboardData:
    break_even_value = ensure_break_even_parameter(conn)
    ensure_quote_schema(conn)

    (
        break_even_value,
        df_all,
        mapping,
        dataset_loader,
        dataset_error,
        start_date,
        end_date,
        selected_corridor,
        selected_clients,
        postcode_prefix,
        empty_dataset_message,
    ) = _sidebar_filters(conn, break_even_value)

    data_available = dataset_error is None and not df_all.empty

    filtered_df = pd.DataFrame()
    filtered_mapping = mapping
    has_filtered_data = False
    if data_available:
        try:
            filtered_df, filtered_mapping = dataset_loader(
                conn,
                start_date,
                end_date,
                selected_clients or None,
                selected_corridor,
                postcode_prefix,
            )
            has_filtered_data = not filtered_df.empty
        except RuntimeError as exc:
            dataset_error = str(exc)
        except Exception as exc:  # pragma: no cover - defensive UI feedback
            dataset_error = f"Failed to apply filters: {exc}"

    if dataset_error:
        st.error(dataset_error)
    elif not data_available:
        st.info(
            empty_dataset_message
            or (
                "No rows available for the selected dataset. Use the initialise "
                "button to create empty tables."
            )
        )
    elif not has_filtered_data:
        st.warning(
            "No jobs match the selected filters. Quote builder remains available below."
        )

    return PreparedDashboardData(
        conn=conn,
        break_even_value=float(break_even_value),
        dataset_error=dataset_error,
        empty_dataset_message=empty_dataset_message,
        data_available=data_available,
        mapping=mapping,
        filtered_mapping=filtered_mapping,
        filtered_df=filtered_df,
        has_filtered_data=has_filtered_data,
    )


@contextmanager
def prepare_dashboard_data() -> Iterator[PreparedDashboardData]:
    """Yield prepared dashboard data while keeping the DB connection open."""

    with connection_scope() as conn:
        state = _prepare_dashboard_data(conn)
        try:
            yield state
        finally:
            # connection_scope handles closing; no additional cleanup required here.
            pass
