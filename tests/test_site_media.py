from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from analytics.db import ensure_dashboard_tables
from analytics.db.site_media import (
    accepted_site_context,
    create_media_inference_result,
    create_site_media_asset,
    list_media_inference_results,
    list_site_assessments,
    list_site_media_assets,
    persist_uploaded_site_media,
    review_media_inference_result,
    upsert_site_assessment,
)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    conn.execute(
        """
        INSERT INTO jobs (
            client, origin, destination, origin_resolved, destination_resolved,
            job_date, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("Client", "Brisbane", "Sydney", "Brisbane", "Sydney", "2026-03-20", "2026-03-20T00:00:00+00:00"),
    )
    conn.commit()
    return conn


def test_site_media_and_assessment_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CORKYSOFT_SITE_MEDIA_DIR", str(tmp_path))
    conn = _conn()
    job_id = int(conn.execute("SELECT id FROM jobs").fetchone()[0])

    asset_id = create_site_media_asset(
        conn,
        job_id=job_id,
        site_kind="origin",
        media_type="street_view_static",
        source="google",
        media_url="https://example.com/street",
        title="Origin street",
    )
    persisted_id = persist_uploaded_site_media(
        conn,
        job_id=job_id,
        site_kind="destination",
        media_type="walkaround_image",
        source="uploaded",
        uploaded_name="site.jpg",
        mime_type="image/jpeg",
        file_bytes=b"test-bytes",
    )
    assessment_id = upsert_site_assessment(
        conn,
        job_id=job_id,
        site_kind="origin",
        loading_access_risk="high",
        parking_risk="medium",
        narrow_street_risk="low",
        stairs_risk="unknown",
        clearance_risk="medium",
        large_vehicle_suitability="review",
        note="Check loading access",
        reviewed_by="planner",
    )

    assets = list_site_media_assets(conn, job_id=job_id)
    assessments = list_site_assessments(conn, job_id=job_id, accepted_only=True)

    assert asset_id > 0
    assert persisted_id > 0
    assert assessment_id > 0
    assert len(assets) == 2
    assert len(assessments) == 1
    uploaded = next(item for item in assets if item["id"] == persisted_id)
    assert uploaded["localPath"] is not None
    assert Path(uploaded["localPath"]).exists()


def test_media_inference_review_and_context() -> None:
    conn = _conn()
    job_id = int(conn.execute("SELECT id FROM jobs").fetchone()[0])
    asset_id = create_site_media_asset(
        conn,
        job_id=job_id,
        site_kind="origin",
        media_type="walkaround_video",
        source="uploaded",
        title="walkthrough",
    )
    inference_id = create_media_inference_result(
        conn,
        media_asset_id=asset_id,
        job_id=job_id,
        result_type="volume_estimate",
        payload={"estimated_m3": 38.5},
        confidence=0.81,
        status="pending_review",
    )
    reviewed = review_media_inference_result(
        conn,
        inference_id,
        decision="corrected",
        reviewed_by="planner",
        corrected_payload={"estimated_m3": 40.0},
    )
    rows = list_media_inference_results(conn, job_id=job_id, statuses=("accepted", "corrected"))
    context = accepted_site_context(conn, job_id=job_id)

    assert reviewed["status"] == "corrected"
    assert reviewed["payload"] == {"estimated_m3": 40.0}
    assert rows[0]["id"] == inference_id
    assert context["acceptedVolumeEstimate"]["payload"] == {"estimated_m3": 40.0}
