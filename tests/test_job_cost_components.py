import os
import sqlite3

import pytest

os.environ.setdefault("ORS_API_KEY", "dummy")

routes = pytest.importorskip("routes_to_sqlite")


def build_conn():
    conn = sqlite3.connect(":memory:")
    routes.ensure_schema(conn)
    routes.migrate_schema(conn)
    return conn


def fetch_job_id(conn):
    row = conn.execute("SELECT id FROM jobs LIMIT 1").fetchone()
    assert row is not None
    return row[0]


def test_add_cost_component_updates_internal_total():
    conn = build_conn()
    try:
        routes.add_job(conn, "Melbourne", "Sydney")
        job_id = fetch_job_id(conn)

        component_id = routes.add_cost_component(
            conn,
            job_id,
            "crew",
            quantity=12,
            rate=45,
            unit="hr",
            description="Crew wages",
        )
        assert isinstance(component_id, int)

        row = conn.execute(
            "SELECT category, quantity, rate, total FROM job_cost_components WHERE id=?",
            (component_id,),
        ).fetchone()
        assert row == ("crew", 12.0, 45.0, pytest.approx(540.0))

        job_row = conn.execute(
            "SELECT internal_cost_total FROM jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        assert job_row[0] == pytest.approx(540.0)

        routes.add_cost_component(
            conn,
            job_id,
            "truck",
            total_override=950,
            description="Prime mover",
        )

        job_row = conn.execute(
            "SELECT internal_cost_total FROM jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        assert job_row[0] == pytest.approx(1490.0)

        per_category, total = routes.summarise_cost_components(conn, job_id)
        assert total == pytest.approx(1490.0)
        assert per_category == {"crew": pytest.approx(540.0), "truck": pytest.approx(950.0)}
    finally:
        conn.close()


def test_delete_cost_component_recalculates_total():
    conn = build_conn()
    try:
        routes.add_job(conn, "Brisbane", "Cairns")
        job_id = fetch_job_id(conn)

        fuel_id = routes.add_cost_component(
            conn,
            job_id,
            "fuel",
            quantity=300,
            rate=1.8,
            unit="L",
        )
        labour_id = routes.add_cost_component(
            conn,
            job_id,
            "labour",
            quantity=10,
            rate=60,
            unit="hr",
        )

        job_row = conn.execute(
            "SELECT internal_cost_total FROM jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        assert job_row[0] == pytest.approx(1140.0)

        routes.delete_cost_component(conn, fuel_id)

        job_row = conn.execute(
            "SELECT internal_cost_total FROM jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        assert job_row[0] == pytest.approx(600.0)

        remaining = conn.execute(
            "SELECT id FROM job_cost_components WHERE job_id=?",
            (job_id,),
        ).fetchall()
        assert remaining == [(labour_id,)]
    finally:
        conn.close()

