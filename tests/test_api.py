from __future__ import annotations

import sqlite3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient

import corkysoft.api as api


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """Provision an isolated SQLite database for API tests."""

    db_path = tmp_path / "api.db"
    monkeypatch.setenv("CORKYSOFT_DB", str(db_path))
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_date TEXT,
                client TEXT,
                origin TEXT,
                destination TEXT,
                revenue_total REAL,
                revenue REAL,
                volume_m3 REAL,
                volume REAL,
                distance_km REAL,
                final_cost REAL,
                updated_at TEXT,
                billing_name TEXT,
                billing_email TEXT,
                service_code TEXT,
                service_text TEXT
            );
            """
        )
        conn.commit()
    yield db_path


def _create_job(conn: sqlite3.Connection) -> int:
    cursor = conn.execute(
        """
        INSERT INTO jobs (
            job_date,
            client,
            origin,
            destination,
            revenue_total,
            revenue,
            volume_m3,
            volume,
            distance_km,
            final_cost,
            updated_at,
            billing_name,
            billing_email,
            service_code,
            service_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2018-08-30T00:00+10:00",
            "SYD Customer",
            "1p, 2p",
            "3p 4p",
            1250.0,
            1250.0,
            50.0,
            50.0,
            900.5,
            975.0,
            "2018-09-01T12:30:00+10:00",
            "SYD Customer",
            "luke.pitcher@moveconnect.com",
            "LTL",
            "Less than truck load",
        ),
    )
    return int(cursor.lastrowid)


def test_get_job_by_id_returns_payload(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        job_id = _create_job(conn)
        conn.commit()

    client = TestClient(api.app)
    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == str(job_id)
    assert payload["billing"]["name"] == "SYD Customer"
    assert payload["billing"]["email"] == "luke.pitcher@moveconnect.com"
    assert payload["service"]["code"] == "LTL"


def test_get_job_by_id_missing_returns_404(isolated_db):
    client = TestClient(api.app)
    response = client.get("/jobs/9999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"


def test_moveware_importer_returns_summary(isolated_db):
    client = TestClient(api.app)
    response = client.post(
        "/importers/moveware/jobs",
        json={
            "records": [
                {"id": "100006", "externalId": "X001-D"},
                {"id": "100007", "externalId": "X002-D"},
            ],
            "dry_run": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "resource": "jobs",
        "imported": 2,
        "dry_run": True,
    }
