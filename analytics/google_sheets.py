"""Helpers for resolving Google Sheets IDs and export URLs."""
from __future__ import annotations

import os
import re
from typing import Sequence
from urllib.parse import quote_plus

GOOGLE_SHEET_XLSX_EXPORT_TEMPLATE = (
    "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
)
GOOGLE_SHEET_CSV_EXPORT_TEMPLATE = (
    "https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?"
    "tqx=out:csv&sheet={sheet_name}"
)


def extract_google_sheet_id(sheet_id_or_url: str | None) -> str | None:
    """Return the Google Sheets ID from either an ID or sharing URL."""

    if not sheet_id_or_url:
        return None
    candidate = sheet_id_or_url.strip()
    if not candidate:
        return None
    if candidate.startswith("http"):
        match = re.search(r"/spreadsheets/d/([\w-]+)", candidate)
        return match.group(1) if match else None
    return candidate


def resolve_google_sheet_reference(
    explicit: str | None, *, env_keys: Sequence[str]
) -> str | None:
    """Return the first non-empty explicit/env Google Sheet reference."""

    if explicit and explicit.strip():
        return explicit.strip()
    for key in env_keys:
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return None


def build_google_sheet_xlsx_url(sheet_id_or_url: str) -> str:
    """Return the XLSX export URL for a Google Sheet."""

    sheet_id = extract_google_sheet_id(sheet_id_or_url)
    if not sheet_id:
        raise ValueError("Could not resolve Google Sheet ID")
    return GOOGLE_SHEET_XLSX_EXPORT_TEMPLATE.format(sheet_id=sheet_id)


def build_google_sheet_csv_url(sheet_id_or_url: str, sheet_name: str) -> str:
    """Return the CSV export URL for a Google Sheet tab."""

    sheet_id = extract_google_sheet_id(sheet_id_or_url)
    if not sheet_id:
        raise ValueError("Could not resolve Google Sheet ID")
    return GOOGLE_SHEET_CSV_EXPORT_TEMPLATE.format(
        sheet_id=sheet_id,
        sheet_name=quote_plus(sheet_name),
    )


__all__ = [
    "build_google_sheet_csv_url",
    "build_google_sheet_xlsx_url",
    "extract_google_sheet_id",
    "resolve_google_sheet_reference",
]
