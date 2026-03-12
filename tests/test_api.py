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


def test_kent_ams_importer_returns_summary(isolated_db):
    client = TestClient(api.app)
    response = client.post(
        "/importers/kent-ams/jobs",
        json={
            "records": [
                {"moveId": "KENT-1001", "origin": "Brisbane", "destination": "Cairns"},
                {"moveId": "KENT-1002", "origin": "Sydney", "destination": "Melbourne"},
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


def test_kent_ams_tender_importer_returns_summary(isolated_db):
    client = TestClient(api.app)
    response = client.post(
        "/importers/kent-ams/tenders",
        json={
            "records": [
                {
                    "tenderId": "T-1001",
                    "moveId": "KENT-2001",
                    "expectedRevenue": 18000,
                    "estimatedDistanceKm": 1200,
                    "estimatedVolumeM3": 34,
                    "requiredTrucks": 2,
                    "requiredWorkers": 4,
                },
                {
                    "tenderId": "T-1002",
                    "moveId": "KENT-2002",
                    "expectedRevenue": 6200,
                    "estimatedDistanceKm": 180,
                    "estimatedVolumeM3": 18,
                    "requiredTrucks": 1,
                    "requiredWorkers": 2,
                },
            ],
            "dry_run": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "resource": "tenders",
        "imported": 2,
        "dry_run": True,
    }


def test_get_prioritized_kent_tenders_returns_ranked_rows(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kent_job_tenders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tender_external_id TEXT NOT NULL UNIQUE,
                job_number TEXT NOT NULL,
                client_name TEXT,
                origin TEXT,
                destination TEXT,
                expected_revenue REAL,
                estimated_cost REAL,
                required_trucks INTEGER,
                required_workers INTEGER,
                due_at TEXT,
                move_date TEXT,
                tender_status TEXT NOT NULL DEFAULT 'open',
                score_total REAL NOT NULL DEFAULT 0,
                score_profitability REAL NOT NULL DEFAULT 0,
                score_capacity REAL NOT NULL DEFAULT 0,
                score_urgency REAL NOT NULL DEFAULT 0,
                score_seasonality REAL NOT NULL DEFAULT 0,
                score_route_location REAL NOT NULL DEFAULT 0,
                score_spare_capacity REAL NOT NULL DEFAULT 0,
                overrideable_flags TEXT NOT NULL DEFAULT '[]',
                hard_block_flags TEXT NOT NULL DEFAULT '[]',
                recommended_action TEXT NOT NULL DEFAULT 'review',
                updated_at TEXT
            );
            """
        )
        conn.execute(
            """
            INSERT INTO kent_job_tenders (
                tender_external_id, job_number, client_name, tender_status,
                score_total, score_profitability, score_capacity, score_urgency,
                score_seasonality, score_route_location, score_spare_capacity,
                overrideable_flags, hard_block_flags, recommended_action, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("T-A", "J-A", "Client A", "open", 91.5, 88.0, 75.0, 95.0, 90.0, 85.0, 92.0, '[]', '[]', "pursue_now", "2026-03-12T00:00:00+00:00"),
        )
        conn.execute(
            """
            INSERT INTO kent_job_tenders (
                tender_external_id, job_number, client_name, tender_status,
                score_total, score_profitability, score_capacity, score_urgency,
                score_seasonality, score_route_location, score_spare_capacity,
                overrideable_flags, hard_block_flags, recommended_action, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("T-B", "J-B", "Client B", "open", 65.0, 62.0, 60.0, 70.0, 55.0, 40.0, 35.0, '["sla_risk_increased"]', '[]', "review_today", "2026-03-12T00:00:00+00:00"),
        )
        conn.commit()

    client = TestClient(api.app)
    response = client.get("/kent-ams/tenders/prioritized?status=open&limit=10")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    assert payload[0]["tenderExternalId"] == "T-A"
    assert payload[1]["tenderExternalId"] == "T-B"
    assert payload[0]["scoreSpareCapacity"] == 92.0
    assert payload[0]["policyMatched"] is False
    assert payload[0]["profitRuleMode"] == "EITHER"
    assert payload[1]["overrideableFlags"] == ["sla_risk_increased"]


def test_get_and_update_kent_tender_config(isolated_db):
    client = TestClient(api.app)

    response = client.get("/kent-ams/config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ruleMode"] == "EITHER"
    assert payload["absoluteMarginThreshold"] == 750.0

    response = client.put(
        "/kent-ams/config",
        json={
            "ruleMode": "BOTH",
            "absoluteMarginThreshold": 1200.0,
            "marginPercentThreshold": 18.0,
            "lossAlertFloor": -250.0,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ruleMode"] == "BOTH"
    assert payload["absoluteMarginThreshold"] == 1200.0
    assert payload["marginPercentThreshold"] == 18.0
    assert payload["lossAlertFloor"] == -250.0


def test_kent_tender_override_reasons_and_history(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kent_job_tenders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tender_external_id TEXT NOT NULL UNIQUE,
                job_number TEXT NOT NULL,
                client_name TEXT,
                expected_revenue REAL,
                estimated_cost REAL,
                tender_status TEXT NOT NULL DEFAULT 'open',
                score_total REAL NOT NULL DEFAULT 0,
                score_profitability REAL NOT NULL DEFAULT 0,
                score_capacity REAL NOT NULL DEFAULT 0,
                score_urgency REAL NOT NULL DEFAULT 0,
                score_seasonality REAL NOT NULL DEFAULT 0,
                score_route_location REAL NOT NULL DEFAULT 0,
                score_spare_capacity REAL NOT NULL DEFAULT 0,
                overrideable_flags TEXT NOT NULL DEFAULT '[]',
                hard_block_flags TEXT NOT NULL DEFAULT '[]',
                recommended_action TEXT NOT NULL DEFAULT 'review',
                created_at TEXT,
                updated_at TEXT
            );
            """
        )
        conn.execute(
            """
            INSERT INTO kent_job_tenders (
                tender_external_id,
                job_number,
                client_name,
                expected_revenue,
                estimated_cost,
                tender_status,
                score_total,
                score_profitability,
                score_capacity,
                score_urgency,
                score_seasonality,
                score_route_location,
                score_spare_capacity,
                overrideable_flags,
                hard_block_flags,
                recommended_action,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "T-OVR",
                "JOB-OVR",
                "Kent Client",
                5000.0,
                5200.0,
                "open",
                44.0,
                40.0,
                45.0,
                50.0,
                55.0,
                60.0,
                65.0,
                '["beyond_transfer_rule"]',
                "[]",
                "review_if_strategic",
                "2026-03-12T00:00:00+00:00",
                "2026-03-12T00:00:00+00:00",
            ),
        )
        conn.commit()

    client = TestClient(api.app)

    response = client.get("/kent-ams/override-reasons")
    assert response.status_code == 200
    reasons = response.json()
    assert any(row["code"] == "retention" for row in reasons)

    response = client.put(
        "/kent-ams/override-reasons/manual_dispatch",
        json={
            "code": "manual_dispatch",
            "label": "Manual dispatch",
            "description": "Planner-led exception",
            "active": True,
        },
    )
    assert response.status_code == 200
    assert response.json()["code"] == "manual_dispatch"

    response = client.post(
        "/kent-ams/tenders/T-OVR/override",
        json={
            "action": "pursue",
            "operatorId": "ops-1",
            "reasonCode": "retention",
            "note": "Needed for customer retention",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["tenderExternalId"] == "T-OVR"
    assert payload["reasonCode"] == "retention"
    assert payload["policyMatched"] is False
    assert payload["lossAlert"] is True

    response = client.get("/kent-ams/tenders/T-OVR/overrides")
    assert response.status_code == 200
    history = response.json()
    assert len(history) == 1
    assert history[0]["operatorId"] == "ops-1"
    assert history[0]["reasonCode"] == "retention"


def test_get_kent_tender_calibration_returns_band_metrics(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.executescript(
            """
            ALTER TABLE jobs ADD COLUMN job_number TEXT;
            CREATE TABLE IF NOT EXISTS kent_job_tenders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tender_external_id TEXT NOT NULL UNIQUE,
                job_number TEXT NOT NULL,
                expected_revenue REAL,
                estimated_cost REAL,
                tender_status TEXT NOT NULL DEFAULT 'open',
                score_total REAL NOT NULL DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS kent_job_awards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                award_external_id TEXT NOT NULL UNIQUE,
                job_number TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO jobs (job_number, revenue_total, final_cost, origin, destination)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("J-A", 15000.0, 11000.0, "Brisbane", "Cairns"),
        )
        conn.execute(
            """
            INSERT INTO jobs (job_number, revenue_total, final_cost, origin, destination)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("J-B", 9000.0, 7600.0, "Sydney", "Melbourne"),
        )
        conn.execute(
            """
            INSERT INTO kent_job_tenders (
                tender_external_id, job_number, expected_revenue, estimated_cost,
                tender_status, score_total, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("T-A", "J-A", 14000.0, 10000.0, "awarded", 92.0, "2026-03-11T00:00:00+00:00", "2026-03-11T00:00:00+00:00"),
        )
        conn.execute(
            """
            INSERT INTO kent_job_tenders (
                tender_external_id, job_number, expected_revenue, estimated_cost,
                tender_status, score_total, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("T-B", "J-B", 8000.0, 7000.0, "open", 68.0, "2026-03-11T00:00:00+00:00", "2026-03-11T00:00:00+00:00"),
        )
        conn.execute(
            """
            INSERT INTO kent_job_awards (award_external_id, job_number)
            VALUES (?, ?)
            """,
            ("A-100", "J-A"),
        )
        conn.commit()

    client = TestClient(api.app)
    response = client.get("/kent-ams/tenders/calibration?lookback_days=365")
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["tenders"] == 2
    assert payload["summary"]["wins"] == 1
    assert payload["summary"]["overallWinRate"] == 0.5
    top_band = next(row for row in payload["bands"] if row["scoreBand"] == "90-100")
    assert top_band["tenders"] == 1
    assert top_band["wins"] == 1
