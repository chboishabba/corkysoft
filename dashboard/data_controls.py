from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import os
from typing import Callable, List, Optional

import pandas as pd
import streamlit as st

from analytics.price_distribution import (
    ColumnMapping,
    import_historical_jobs_from_dataframe,
    latest_historical_ingest_summary,
    load_historical_jobs,
    load_live_jobs,
    load_quotes,
    update_break_even,
)
from corkysoft.quote_service import format_currency
from dashboard.data import blank_column_mapping
from dashboard.map_provider import google_maps_api_key


DatasetLoader = Callable[..., tuple[pd.DataFrame, ColumnMapping]]
ImportFeedback = tuple[str, str]

DATASET_OPTIONS: dict[str, tuple[str, DatasetLoader]] = {
    "Historical quotes": ("historical", load_historical_jobs),
    "Saved quick quotes": ("quotes", load_quotes),
    "Live jobs": ("live", load_live_jobs),
}

PROVIDER_OPTIONS = {
    "OpenRouteService": "ors",
    "Google Maps": "google",
}


@dataclass
class DatasetControlResult:
    dataset_key: str
    dataset_label: str
    dataset_loader: DatasetLoader
    df_all: pd.DataFrame
    mapping: ColumnMapping
    dataset_error: Optional[str]
    data_available: bool
    start_date: Optional[date]
    end_date: Optional[date]
    selected_corridor: Optional[str]
    selected_clients: List[str]
    postcode_prefix: Optional[str]
    break_even_value: float
    empty_dataset_message: Optional[str]


@dataclass
class DatasetFilterState:
    start_date: Optional[date]
    end_date: Optional[date]
    selected_corridor: Optional[str]
    selected_clients: List[str]
    postcode_prefix: Optional[str]


def _initialise_database_tables(
    conn: "sqlite3.Connection",
    rerun_app: Callable[[], None],
) -> None:
    from analytics.db import ensure_dashboard_tables
    from corkysoft.repo import ensure_quote_schema
    from corkysoft.schema import ensure_schema as ensure_core_schema

    ensure_core_schema(conn)
    ensure_dashboard_tables(conn)
    ensure_quote_schema(conn)
    st.success(
        "Database tables initialised. Import data or start building quotes below."
    )
    rerun_app()


def _render_dataset_selector(current_label: str) -> tuple[str, str, DatasetLoader]:
    dataset_label = st.radio(
        "Dataset",
        options=list(DATASET_OPTIONS.keys()),
        format_func=lambda label: label,
        key="dashboard_dataset_selector",
        index=list(DATASET_OPTIONS.keys()).index(current_label)
        if current_label in DATASET_OPTIONS
        else 0,
    )
    dataset_key, dataset_loader = DATASET_OPTIONS[dataset_label]
    return dataset_label, dataset_key, dataset_loader


def _current_provider_state() -> tuple[List[str], str, str]:
    provider_labels = list(PROVIDER_OPTIONS.keys())
    current_provider_env = os.environ.get("ROUTING_PROVIDER", "ors").strip().lower()
    default_provider_label = next(
        (
            label
            for label, value in PROVIDER_OPTIONS.items()
            if value == current_provider_env
        ),
        provider_labels[0],
    )
    return provider_labels, current_provider_env, default_provider_label


def _render_routing_provider_selector(rerun_app: Callable[[], None]) -> None:
    provider_labels, current_provider_env, default_provider_label = _current_provider_state()
    provider_choice_label = st.radio(
        "Routing provider",
        options=provider_labels,
        index=provider_labels.index(default_provider_label),
        key="dashboard_routing_provider_selector",
        help="Select which routing provider to use for map tiles and route geometry.",
    )
    resolved_provider = PROVIDER_OPTIONS[provider_choice_label]
    if resolved_provider != current_provider_env:
        os.environ["ROUTING_PROVIDER"] = resolved_provider
        rerun_app()

    if resolved_provider == "google" and not google_maps_api_key():
        st.warning("Google Maps selected but GOOGLE_MAPS_API_KEY is not configured.")


def _render_historical_ingest_summary(conn: "sqlite3.Connection") -> None:
    ingest_summary = latest_historical_ingest_summary(conn)
    if ingest_summary is None:
        return

    ingest_cols = st.columns(4)
    ingest_cols[0].metric(
        "Readiness",
        str(ingest_summary.get("readiness_status") or "unknown"),
    )
    ingest_cols[1].metric(
        "Inserted",
        int(ingest_summary.get("inserted_rows") or 0),
    )
    ingest_cols[2].metric(
        "Skipped",
        int(ingest_summary.get("skipped_rows") or 0),
    )
    ingest_cols[3].metric(
        "Issues",
        int(ingest_summary.get("issue_count") or 0),
    )
    st.caption(
        "Latest ingest: "
        + str(ingest_summary.get("source_name") or "unknown source")
        + " at "
        + str(ingest_summary.get("completed_at") or "unknown time")
    )
    coverage = ingest_summary.get("coverage_summary") or {}
    top_issues = coverage.get("topIssueCodes") or []
    if top_issues:
        st.caption(
            "Top issues: "
            + ", ".join(
                f"{item['issueCode']} ({item['count']})"
                for item in top_issues
            )
        )


def _render_historical_import_controls(
    conn: "sqlite3.Connection",
) -> Optional[ImportFeedback]:
    with st.expander("Import historical jobs from CSV", expanded=False):
        _render_historical_ingest_summary(conn)
        import_form = st.form(key="dashboard_sidebar_historical_import_form")
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
        except Exception as exc:  # pragma: no cover - streamlit upload
            return ("error", f"Failed to read CSV: {exc}")
        try:
            inserted, skipped_rows = import_historical_jobs_from_dataframe(
                conn, imported_df
            )
        except ValueError as exc:
            return ("error", str(exc))
        except Exception as exc:
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
            return (
                "warning",
                "No new rows imported. Skipped "
                f"{skipped_rows} row{'s' if skipped_rows != 1 else ''} due to validation or duplicates.",
            )
        return ("warning", "No rows imported from the provided file.")


def _load_dataset_snapshot(
    conn: "sqlite3.Connection",
    dataset_loader: DatasetLoader,
    dataset_label: str,
) -> tuple[pd.DataFrame, ColumnMapping, Optional[str], bool]:
    try:
        df_all, mapping = dataset_loader(conn)
    except RuntimeError as exc:
        return pd.DataFrame(), blank_column_mapping(), str(exc), False
    except Exception as exc:
        return (
            pd.DataFrame(),
            blank_column_mapping(),
            f"Failed to load {dataset_label.lower()} data: {exc}",
            False,
        )
    return df_all, mapping, None, not df_all.empty


def _render_import_feedback(import_feedback: Optional[ImportFeedback]) -> None:
    if not import_feedback:
        return
    level, message = import_feedback
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.info(message)
    else:
        st.error(message)


def _date_bounds_for_dataset(
    df_all: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    today_value: date,
) -> tuple[Optional[date], Optional[date]]:
    date_column = "job_date" if "job_date" in df_all.columns else mapping.date
    if not date_column or date_column not in df_all.columns or df_all.empty:
        return None, None

    df_all[date_column] = pd.to_datetime(df_all[date_column], errors="coerce")
    min_date = df_all[date_column].min()
    max_date = df_all[date_column].max()
    default_start = min_date.date() if isinstance(min_date, pd.Timestamp) else today_value
    default_end = max_date.date() if isinstance(max_date, pd.Timestamp) else today_value
    return default_start, default_end


def _corridor_options(df_all: pd.DataFrame) -> List[str]:
    corridor_series = df_all.get("corridor_display")
    if corridor_series is None:
        return []
    return sorted(
        pd.Series(corridor_series).dropna().astype(str).unique().tolist()
    )


def _client_options(df_all: pd.DataFrame) -> List[str]:
    client_series = df_all.get("client_display")
    if client_series is None:
        return []
    return sorted(
        pd.Series(client_series).dropna().astype(str).unique().tolist()
    )


def _render_dataset_filters(
    df_all: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    data_available: bool,
) -> DatasetFilterState:
    today_value = date.today()
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    bounds = _date_bounds_for_dataset(df_all, mapping, today_value=today_value)
    if data_available and bounds != (None, None):
        default_start, default_end = bounds
        date_range = st.date_input(
            "Date range",
            value=(default_start, default_end),
            min_value=default_start,
            max_value=default_end,
            key="date_range_active",
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
            key="date_range_disabled",
        )

    corridor_selection = st.selectbox(
        "Corridor",
        options=["All corridors"] + (_corridor_options(df_all) if data_available else []),
        index=0,
        disabled=not data_available,
    )
    selected_corridor = None if corridor_selection == "All corridors" else corridor_selection

    client_options = _client_options(df_all) if data_available else []
    selected_clients = st.multiselect(
        "Client",
        options=client_options,
        default=client_options if client_options else [],
        disabled=not data_available,
        key="client_filter_multiselect",
    )

    postcode_prefix = st.text_input(
        "Corridor contains postcode prefix",
        value="",
        disabled=not data_available,
        help="Match origin or destination postcode prefixes (e.g. 40 to match 4000-4099).",
        key="postcode_prefix_filter",
    ) or None

    return DatasetFilterState(
        start_date=start_date,
        end_date=end_date,
        selected_corridor=selected_corridor,
        selected_clients=selected_clients,
        postcode_prefix=postcode_prefix,
    )


def _empty_dataset_message_for(dataset_key: str) -> str:
    empty_messages = {
        "historical": (
            "historical_jobs table has no rows yet. Import historical jobs to populate the view."
        ),
        "quotes": (
            "quotes table has no rows yet. Save a quick quote to populate the view."
        ),
        "live": "jobs table has no rows yet. Add live jobs to populate the view.",
    }
    return empty_messages.get(
        dataset_key, "No rows available for the selected dataset."
    )


def _render_break_even_controls(
    conn: "sqlite3.Connection",
    break_even_value: float,
) -> float:
    st.subheader("Break-even model")
    new_break_even = st.number_input(
        "Break-even $/m³",
        min_value=0.0,
        value=float(break_even_value),
        step=5.0,
        help="Used to draw break-even bands on the histogram.",
        key="break_even_input",
    )
    if st.button("Update break-even", key="break_even_update_button"):
        update_break_even(conn, new_break_even)
        st.success(f"Break-even updated to {format_currency(new_break_even)}")
        return new_break_even
    return break_even_value


def render_dataset_sidebar(
    conn: "sqlite3.Connection",
    *,
    sidebar_heading: str,
    sidebar_caption: Optional[str],
    collapse_analytics_sidebar: bool,
    dataset_loader: DatasetLoader,
    dataset_key: str,
    dataset_label: str,
    df_all: pd.DataFrame,
    mapping: ColumnMapping,
    break_even_value: float,
    rerun_app: Callable[[], None],
) -> DatasetControlResult:
    dataset_error: Optional[str] = None
    empty_dataset_message: Optional[str] = None
    data_available = False
    filters = DatasetFilterState(
        start_date=None,
        end_date=None,
        selected_corridor=None,
        selected_clients=[],
        postcode_prefix=None,
    )

    with st.sidebar:
        st.header(sidebar_heading)
        if sidebar_caption:
            st.caption(sidebar_caption)
        analytics_sidebar = (
            st.expander("Analytics filters and pricing controls", expanded=False)
            if collapse_analytics_sidebar
            else st.container()
        )
        with analytics_sidebar:
            if st.button(
                "Initialise database tables",
                help=(
                    "Create empty historical and live job tables so the dashboard can run "
                    "before data imports."
                ),
                key="dashboard_sidebar_init_db",
            ):
                _initialise_database_tables(conn, rerun_app)

            dataset_label, dataset_key, dataset_loader = _render_dataset_selector(
                dataset_label
            )
            _render_routing_provider_selector(rerun_app)

            import_feedback: Optional[ImportFeedback] = None
            if dataset_key == "historical":
                import_feedback = _render_historical_import_controls(conn)

            df_all, mapping, dataset_error, data_available = _load_dataset_snapshot(
                conn,
                dataset_loader,
                dataset_label,
            )
            _render_import_feedback(import_feedback)
            filters = _render_dataset_filters(
                df_all,
                mapping,
                data_available=data_available,
            )

            if dataset_error:
                st.error(dataset_error)
            elif not data_available:
                empty_dataset_message = _empty_dataset_message_for(dataset_key)
                st.info(empty_dataset_message)

            break_even_value = _render_break_even_controls(conn, break_even_value)

    return DatasetControlResult(
        dataset_key=dataset_key,
        dataset_label=dataset_label,
        dataset_loader=dataset_loader,
        df_all=df_all,
        mapping=mapping,
        dataset_error=dataset_error,
        data_available=data_available,
        start_date=filters.start_date,
        end_date=filters.end_date,
        selected_corridor=filters.selected_corridor,
        selected_clients=filters.selected_clients,
        postcode_prefix=filters.postcode_prefix,
        break_even_value=break_even_value,
        empty_dataset_message=empty_dataset_message,
    )
