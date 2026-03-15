from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from .schema import ensure_dashboard_tables

SITE_MEDIA_TYPES: Sequence[str] = (
    "street_view_static",
    "street_view_360",
    "walkaround_video",
    "walkaround_image",
)
SITE_ASSESSMENT_RISK_LEVELS: Sequence[str] = (
    "unknown",
    "low",
    "medium",
    "high",
)
SITE_KIND_VALUES: Sequence[str] = (
    "origin",
    "destination",
    "general",
)
SITE_TRUCK_SUITABILITY: Sequence[str] = (
    "unknown",
    "suitable",
    "review",
    "unsuitable",
)
MEDIA_INFERENCE_TYPES: Sequence[str] = (
    "object_detection",
    "volume_estimate",
    "site_features",
)
MEDIA_INFERENCE_STATUSES: Sequence[str] = (
    "pending_review",
    "accepted",
    "rejected",
    "corrected",
)


def create_site_media_asset(
    conn: sqlite3.Connection,
    *,
    quote_id: int | None = None,
    job_id: int | None = None,
    site_kind: str = "general",
    media_type: str,
    source: str,
    title: str | None = None,
    media_url: str | None = None,
    local_path: str | None = None,
    mime_type: str | None = None,
    heading_degrees: float | None = None,
    captured_by: str | None = None,
    status: str = "available",
    metadata: dict[str, Any] | None = None,
) -> int:
    ensure_dashboard_tables(conn)
    timestamp = datetime.now(UTC).isoformat()
    site_kind_value = site_kind if site_kind in SITE_KIND_VALUES else "general"
    media_type_value = media_type if media_type in SITE_MEDIA_TYPES else "walkaround_image"
    payload_json = json.dumps(metadata or {}, sort_keys=True)
    cursor = conn.execute(
        """
        INSERT INTO site_media_assets (
            quote_id, job_id, site_kind, media_type, source, title,
            media_url, local_path, mime_type, heading_degrees,
            captured_by, status, metadata_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            quote_id,
            job_id,
            site_kind_value,
            media_type_value,
            source.strip(),
            _clean_optional_str(title),
            _clean_optional_str(media_url),
            _clean_optional_str(local_path),
            _clean_optional_str(mime_type),
            None if heading_degrees is None else float(heading_degrees),
            _clean_optional_str(captured_by),
            status.strip() or "available",
            payload_json,
            timestamp,
            timestamp,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def list_site_media_assets(
    conn: sqlite3.Connection,
    *,
    quote_id: int | None = None,
    job_id: int | None = None,
) -> list[dict[str, Any]]:
    ensure_dashboard_tables(conn)
    clauses: list[str] = []
    params: list[object] = []
    if quote_id is not None:
        clauses.append("quote_id = ?")
        params.append(int(quote_id))
    if job_id is not None:
        clauses.append("job_id = ?")
        params.append(int(job_id))
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT *
        FROM site_media_assets
        {where_sql}
        ORDER BY created_at DESC, id DESC
        """,
        tuple(params),
    ).fetchall()
    return [_site_media_row(row) for row in rows]


def upsert_site_assessment(
    conn: sqlite3.Connection,
    *,
    quote_id: int | None = None,
    job_id: int | None = None,
    site_kind: str,
    source: str = "manual",
    loading_access_risk: str = "unknown",
    parking_risk: str = "unknown",
    narrow_street_risk: str = "unknown",
    stairs_risk: str = "unknown",
    clearance_risk: str = "unknown",
    large_vehicle_suitability: str = "unknown",
    uncertainty_flag: bool = False,
    note: str | None = None,
    reviewed_by: str | None = None,
    accepted: bool = True,
) -> int:
    ensure_dashboard_tables(conn)
    timestamp = datetime.now(UTC).isoformat()
    site_kind_value = site_kind if site_kind in SITE_KIND_VALUES else "general"
    cursor = conn.execute(
        """
        INSERT INTO site_assessments (
            quote_id, job_id, site_kind, source,
            loading_access_risk, parking_risk, narrow_street_risk,
            stairs_risk, clearance_risk, large_vehicle_suitability,
            uncertainty_flag, note, reviewed_by, accepted,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(quote_id, job_id, site_kind, source) DO UPDATE SET
            loading_access_risk = excluded.loading_access_risk,
            parking_risk = excluded.parking_risk,
            narrow_street_risk = excluded.narrow_street_risk,
            stairs_risk = excluded.stairs_risk,
            clearance_risk = excluded.clearance_risk,
            large_vehicle_suitability = excluded.large_vehicle_suitability,
            uncertainty_flag = excluded.uncertainty_flag,
            note = excluded.note,
            reviewed_by = excluded.reviewed_by,
            accepted = excluded.accepted,
            updated_at = excluded.updated_at
        """,
        (
            quote_id,
            job_id,
            site_kind_value,
            source.strip() or "manual",
            _risk_level(loading_access_risk),
            _risk_level(parking_risk),
            _risk_level(narrow_street_risk),
            _risk_level(stairs_risk),
            _risk_level(clearance_risk),
            _truck_suitability(large_vehicle_suitability),
            int(bool(uncertainty_flag)),
            _clean_optional_str(note),
            _clean_optional_str(reviewed_by),
            int(bool(accepted)),
            timestamp,
            timestamp,
        ),
    )
    conn.commit()
    row = conn.execute(
        """
        SELECT id FROM site_assessments
        WHERE COALESCE(quote_id, -1) = COALESCE(?, -1)
          AND COALESCE(job_id, -1) = COALESCE(?, -1)
          AND site_kind = ?
          AND source = ?
        """,
        (quote_id, job_id, site_kind_value, source.strip() or "manual"),
    ).fetchone()
    return int(row[0])


def list_site_assessments(
    conn: sqlite3.Connection,
    *,
    quote_id: int | None = None,
    job_id: int | None = None,
    accepted_only: bool = False,
) -> list[dict[str, Any]]:
    ensure_dashboard_tables(conn)
    clauses: list[str] = []
    params: list[object] = []
    if quote_id is not None:
        clauses.append("quote_id = ?")
        params.append(int(quote_id))
    if job_id is not None:
        clauses.append("job_id = ?")
        params.append(int(job_id))
    if accepted_only:
        clauses.append("accepted = 1")
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT *
        FROM site_assessments
        {where_sql}
        ORDER BY updated_at DESC, id DESC
        """,
        tuple(params),
    ).fetchall()
    return [_site_assessment_row(row) for row in rows]


def create_media_inference_result(
    conn: sqlite3.Connection,
    *,
    media_asset_id: int | None = None,
    quote_id: int | None = None,
    job_id: int | None = None,
    result_type: str,
    payload: dict[str, Any],
    confidence: float | None = None,
    source: str = "manual",
    model_name: str | None = None,
    model_version: str | None = None,
    status: str = "pending_review",
) -> int:
    ensure_dashboard_tables(conn)
    timestamp = datetime.now(UTC).isoformat()
    result_type_value = result_type if result_type in MEDIA_INFERENCE_TYPES else "site_features"
    status_value = status if status in MEDIA_INFERENCE_STATUSES else "pending_review"
    cursor = conn.execute(
        """
        INSERT INTO media_inference_results (
            media_asset_id, quote_id, job_id, result_type, source,
            confidence, payload_json, model_name, model_version,
            status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            media_asset_id,
            quote_id,
            job_id,
            result_type_value,
            source.strip() or "manual",
            None if confidence is None else float(confidence),
            json.dumps(payload, sort_keys=True),
            _clean_optional_str(model_name),
            _clean_optional_str(model_version),
            status_value,
            timestamp,
            timestamp,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def review_media_inference_result(
    conn: sqlite3.Connection,
    inference_id: int,
    *,
    decision: str,
    reviewed_by: str,
    corrected_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_dashboard_tables(conn)
    current = conn.execute(
        "SELECT * FROM media_inference_results WHERE id = ?",
        (int(inference_id),),
    ).fetchone()
    if current is None:
        raise ValueError(f"Inference result {inference_id} does not exist.")
    decision_value = decision if decision in MEDIA_INFERENCE_STATUSES else "pending_review"
    timestamp = datetime.now(UTC).isoformat()
    corrected_payload_json = (
        json.dumps(corrected_payload, sort_keys=True) if corrected_payload is not None else None
    )
    conn.execute(
        """
        UPDATE media_inference_results
        SET status = ?, reviewed_by = ?, reviewed_at = ?,
            corrected_payload_json = COALESCE(?, corrected_payload_json),
            updated_at = ?
        WHERE id = ?
        """,
        (
            decision_value,
            reviewed_by.strip(),
            timestamp,
            corrected_payload_json,
            timestamp,
            int(inference_id),
        ),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM media_inference_results WHERE id = ?", (int(inference_id),)).fetchone()
    return _inference_row(row)


def list_media_inference_results(
    conn: sqlite3.Connection,
    *,
    media_asset_id: int | None = None,
    quote_id: int | None = None,
    job_id: int | None = None,
    statuses: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    ensure_dashboard_tables(conn)
    clauses: list[str] = []
    params: list[object] = []
    if media_asset_id is not None:
        clauses.append("media_asset_id = ?")
        params.append(int(media_asset_id))
    if quote_id is not None:
        clauses.append("quote_id = ?")
        params.append(int(quote_id))
    if job_id is not None:
        clauses.append("job_id = ?")
        params.append(int(job_id))
    if statuses:
        clean = [status for status in statuses if status in MEDIA_INFERENCE_STATUSES]
        if clean:
            clauses.append(f"status IN ({','.join('?' for _ in clean)})")
            params.extend(clean)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT *
        FROM media_inference_results
        {where_sql}
        ORDER BY created_at DESC, id DESC
        """,
        tuple(params),
    ).fetchall()
    return [_inference_row(row) for row in rows]


def persist_uploaded_site_media(
    conn: sqlite3.Connection,
    *,
    quote_id: int | None = None,
    job_id: int | None = None,
    site_kind: str,
    media_type: str,
    source: str,
    uploaded_name: str,
    mime_type: str | None,
    file_bytes: bytes,
    heading_degrees: float | None = None,
    captured_by: str | None = None,
) -> int:
    directory = Path(os.environ.get("CORKYSOFT_SITE_MEDIA_DIR", "/tmp/corkysoft_site_media"))
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    safe_name = _safe_filename(uploaded_name)
    target = directory / f"{timestamp}_{safe_name}"
    target.write_bytes(file_bytes)
    return create_site_media_asset(
        conn,
        quote_id=quote_id,
        job_id=job_id,
        site_kind=site_kind,
        media_type=media_type,
        source=source,
        title=uploaded_name,
        local_path=str(target),
        mime_type=mime_type,
        heading_degrees=heading_degrees,
        captured_by=captured_by,
        metadata={"uploadedName": uploaded_name},
    )


def accepted_site_context(
    conn: sqlite3.Connection,
    *,
    quote_id: int | None = None,
    job_id: int | None = None,
) -> dict[str, Any]:
    assessments = list_site_assessments(conn, quote_id=quote_id, job_id=job_id, accepted_only=True)
    media_assets = list_site_media_assets(conn, quote_id=quote_id, job_id=job_id)
    inference_rows = list_media_inference_results(
        conn,
        quote_id=quote_id,
        job_id=job_id,
        statuses=("accepted", "corrected"),
    )
    latest_volume = next((row for row in inference_rows if row["resultType"] == "volume_estimate"), None)
    latest_site_features = [row for row in inference_rows if row["resultType"] == "site_features"]
    detections = [row for row in inference_rows if row["resultType"] == "object_detection"]
    return {
        "assessments": assessments,
        "mediaAssets": media_assets,
        "acceptedVolumeEstimate": latest_volume,
        "acceptedSiteFeatures": latest_site_features,
        "acceptedDetections": detections,
    }


def _site_media_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "quoteId": row["quote_id"],
        "jobId": row["job_id"],
        "siteKind": row["site_kind"],
        "mediaType": row["media_type"],
        "source": row["source"],
        "title": row["title"],
        "mediaUrl": row["media_url"],
        "localPath": row["local_path"],
        "mimeType": row["mime_type"],
        "headingDegrees": row["heading_degrees"],
        "capturedBy": row["captured_by"],
        "status": row["status"],
        "metadata": _json_load(row["metadata_json"]),
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def _site_assessment_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "quoteId": row["quote_id"],
        "jobId": row["job_id"],
        "siteKind": row["site_kind"],
        "source": row["source"],
        "loadingAccessRisk": row["loading_access_risk"],
        "parkingRisk": row["parking_risk"],
        "narrowStreetRisk": row["narrow_street_risk"],
        "stairsRisk": row["stairs_risk"],
        "clearanceRisk": row["clearance_risk"],
        "largeVehicleSuitability": row["large_vehicle_suitability"],
        "uncertaintyFlag": bool(row["uncertainty_flag"]),
        "note": row["note"],
        "reviewedBy": row["reviewed_by"],
        "accepted": bool(row["accepted"]),
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def _inference_row(row: sqlite3.Row) -> dict[str, Any]:
    corrected_payload = _json_load(row["corrected_payload_json"])
    payload = corrected_payload or _json_load(row["payload_json"])
    return {
        "id": int(row["id"]),
        "mediaAssetId": row["media_asset_id"],
        "quoteId": row["quote_id"],
        "jobId": row["job_id"],
        "resultType": row["result_type"],
        "source": row["source"],
        "confidence": row["confidence"],
        "payload": payload,
        "rawPayload": _json_load(row["payload_json"]),
        "correctedPayload": corrected_payload,
        "modelName": row["model_name"],
        "modelVersion": row["model_version"],
        "status": row["status"],
        "reviewedBy": row["reviewed_by"],
        "reviewedAt": row["reviewed_at"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def _risk_level(value: str | None) -> str:
    candidate = (value or "unknown").strip().lower()
    return candidate if candidate in SITE_ASSESSMENT_RISK_LEVELS else "unknown"


def _truck_suitability(value: str | None) -> str:
    candidate = (value or "unknown").strip().lower()
    return candidate if candidate in SITE_TRUCK_SUITABILITY else "unknown"


def _json_load(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)
    return cleaned or "upload.bin"


def _clean_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None
