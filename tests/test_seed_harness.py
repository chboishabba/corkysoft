from __future__ import annotations

import sqlite3

from analytics.db.inventory import list_inventory_requirements
from analytics.db.schema import ensure_dashboard_tables
from analytics.seed_harness import seed_mainland_jobs


def test_seed_mainland_jobs_creates_jobs_segments_and_container_requirements() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    seeded = seed_mainland_jobs(conn, count=10, seed=1234, baseline_containers=30)

    assert len(seeded) == 10
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 10
    assert conn.execute("SELECT COUNT(*) FROM historical_jobs").fetchone()[0] == 10
    assert conn.execute("SELECT COUNT(*) FROM historical_job_routes").fetchone()[0] == 10
    assert conn.execute("SELECT COUNT(*) FROM job_segments").fetchone()[0] == 10
    assert conn.execute("SELECT COUNT(*) FROM inventory_requirements").fetchone()[0] == 10

    stock = conn.execute(
        "SELECT quantity, architecture FROM inventory_items WHERE name = ?",
        ("Standard Container Pod",),
    ).fetchone()
    assert stock is not None
    assert stock["quantity"] == 30
    assert stock["architecture"] == "container"

    requirements = list_inventory_requirements(conn)
    assert len(requirements) == 10
    assert all(item["architecture"] == "container" for item in requirements)
    assert all(item["requiredQuantity"] > 0 for item in requirements)
    assert any(item["shortageQuantity"] >= 0 for item in requirements)


def test_seed_mainland_jobs_is_idempotent_for_deterministic_seed() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    first = seed_mainland_jobs(conn, count=10, seed=20260314, baseline_containers=30)
    second = seed_mainland_jobs(conn, count=10, seed=20260314, baseline_containers=30)

    assert len(first) == 10
    assert len(second) == 10
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 10
    assert conn.execute("SELECT COUNT(*) FROM historical_jobs").fetchone()[0] == 10
    assert conn.execute("SELECT COUNT(*) FROM historical_job_routes").fetchone()[0] == 10
