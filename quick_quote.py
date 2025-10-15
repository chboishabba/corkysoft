#!/usr/bin/env python3
"""Quick quote entry CLI for Corkysoft.

Feature 2 implementation: collects quote inputs, queries ORS for distance,
calculates pricing using the Feature 1 model, applies modifiers/seasonal uplift
and optional margin, persists to SQLite and prints a copy-paste summary.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import textwrap
from datetime import date, datetime
from typing import List, Optional, Sequence

from analytics.db import ensure_dashboard_tables
from corkysoft.pricing import DEFAULT_MODIFIERS
from corkysoft.quote_service import (
    COUNTRY_DEFAULT,
    QuoteInput,
    calculate_quote,
    format_currency,
)
from corkysoft.repo import ensure_schema, persist_quote

DB_PATH = os.environ.get("ROUTES_DB", "routes.db")


def prompt_input(prompt: str, default: Optional[str] = None) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        value = input(f"{prompt}{suffix}: ").strip()
        if not value and default is not None:
            return default
        if value:
            return value
        print("This field is required. Please enter a value.")


def prompt_float(
    prompt: str, default: Optional[float] = None, minimum: Optional[float] = None
) -> float:
    while True:
        suffix = ""
        if default is not None:
            suffix = f" [{default}]"
        value = input(f"{prompt}{suffix}: ").strip()
        if not value and default is not None:
            value = str(default)
        try:
            num = float(value)
        except ValueError:
            print("Enter a numeric value.")
            continue
        if minimum is not None and num < minimum:
            print(f"Value must be ≥ {minimum}.")
            continue
        return num


def prompt_date(prompt: str, default: Optional[date] = None) -> date:
    default_str = default.isoformat() if default else None
    while True:
        raw = prompt_input(prompt, default=default_str)
        try:
            return datetime.fromisoformat(raw).date()
        except ValueError:
            print("Enter a date in ISO format (YYYY-MM-DD).")


def prompt_modifiers() -> List[str]:
    print("\nSelect modifiers (tick boxes). Enter comma-separated numbers, blank for none.")
    for idx, mod in enumerate(DEFAULT_MODIFIERS, start=1):
        desc = textwrap.fill(mod.description, width=68)
        desc_lines = desc.splitlines() or [""]
        label_line = f"  [{idx}] {mod.label}"
        if mod.calc_type == "flat":
            value_txt = format_currency(mod.value)
        else:
            value_txt = f"{mod.value*100:.0f}% of base"
        label_line += f" ({value_txt})"
        print(label_line)
        for line in desc_lines:
            print(f"       {line}")
    raw = input("Selection: ").strip()
    if not raw:
        return []
    selected: List[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            print(f"Ignoring invalid selection: {part}")
            continue
        idx = int(part)
        if idx < 1 or idx > len(DEFAULT_MODIFIERS):
            print(f"Ignoring invalid selection: {part}")
            continue
        selected.append(DEFAULT_MODIFIERS[idx - 1].id)
    return selected


def prompt_margin() -> Optional[float]:
    raw = input("Target margin % (blank for none): ").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print("Invalid margin. Skipping margin application.")
        return None


def gather_inputs(args: argparse.Namespace) -> QuoteInput:
    if args.origin and args.destination and args.cubic_m is not None:
        quote_dt = args.date or date.today()
        modifiers = args.modifiers or []
        margin = args.margin
        return QuoteInput(
            origin=args.origin,
            destination=args.destination,
            cubic_m=args.cubic_m,
            quote_date=quote_dt,
            modifiers=modifiers,
            target_margin_percent=margin,
            country=args.country or COUNTRY_DEFAULT,
        )

    print("\n--- Quick Quote Entry ---")
    origin = prompt_input("Origin address / postcode")
    destination = prompt_input("Destination address / postcode")
    cubic_m = prompt_float("Volume (m³)", minimum=1.0)
    quote_dt = prompt_date("Move date", default=date.today())
    modifiers = prompt_modifiers()
    margin = prompt_margin()
    return QuoteInput(
        origin=origin,
        destination=destination,
        cubic_m=cubic_m,
        quote_date=quote_dt,
        modifiers=modifiers,
        target_margin_percent=margin,
        country=args.country or COUNTRY_DEFAULT,
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick quote entry CLI")
    parser.add_argument("--origin")
    parser.add_argument("--destination")
    parser.add_argument("--cubic-m", type=float, dest="cubic_m")
    parser.add_argument("--date", type=lambda s: datetime.fromisoformat(s).date())
    parser.add_argument("--modifier", action="append", dest="modifiers")
    parser.add_argument("--margin", type=float)
    parser.add_argument("--country", default=COUNTRY_DEFAULT)
    parser.add_argument(
        "--no-save", action="store_true", help="Do not persist quote to the database"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)
    ensure_dashboard_tables(conn)

    inputs = gather_inputs(args)

    try:
        result = calculate_quote(conn, inputs)
    except RuntimeError as exc:
        print(str(exc))
        conn.close()
        return 1

    rowid: Optional[int] = None
    if not args.no_save:
        rowid = persist_quote(conn, inputs, result)

    print("\n--- Quote Summary ---")
    print(result.summary_text)

    if rowid is not None:
        print(f"\nSaved quote #{rowid} to {DB_PATH}.")
    conn.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
