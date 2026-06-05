"""Shared helpers for Corkysoft's FastAPI surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import secrets
import sqlite3
from typing import Iterable, Optional

from fastapi import Header, HTTPException, Request


API_READ_SCOPE = "api:read"
IMPORT_WRITE_SCOPE = "import:write"
CALLS_WRITE_SCOPE = "calls:write"
WORKER_TIME_WRITE_SCOPE = "worker_time:write"
KENT_WRITE_SCOPE = "kent:write"
LABOR_WRITE_SCOPE = "labor:write"
OPERATIONS_WRITE_SCOPE = "operations:write"
OPERATIONS_CUTOVER_WRITE_SCOPE = "operations.cutover:write"
OPERATIONS_CUTOVER_APPROVE_SCOPE = "operations.cutover:approve"


@dataclass(frozen=True)
class ApiAuthContext:
    credential_id: str
    actor: str
    scopes: tuple[str, ...]
    request_id: str
    legacy: bool = False


def _current_db_path() -> str:
    """Return the SQLite database path configured for the API."""

    return (
        os.environ.get("CORKYSOFT_DB")
        or os.environ.get("ROUTES_DB")
        or "routes.db"
    )


def _required_internal_api_token() -> str:
    token = os.environ.get("CORKYSOFT_API_TOKEN")
    if not token:
        raise HTTPException(
            status_code=503,
            detail="CORKYSOFT_API_TOKEN is not configured for mutating API routes",
        )
    return token


def require_internal_api_token(
    x_corkysoft_api_key: Optional[str] = Header(
        default=None, alias="X-Corkysoft-Api-Key"
    ),
) -> None:
    expected = _required_internal_api_token()
    if x_corkysoft_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid internal API token")


def require_internal_api_read_token(
    x_corkysoft_api_key: Optional[str] = Header(
        default=None, alias="X-Corkysoft-Api-Key"
    ),
) -> None:
    """Require internal authorization for sensitive read endpoints."""

    for credential in _service_credentials():
        if x_corkysoft_api_key == credential["token"]:
            if API_READ_SCOPE not in credential["scopes"]:
                raise HTTPException(status_code=403, detail="Credential scope is not authorized")
            return
    expected = _required_internal_api_token()
    if x_corkysoft_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid internal API token")


def _service_credentials() -> list[dict]:
    raw = os.environ.get("CORKYSOFT_SERVICE_CREDENTIALS_JSON", "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=503,
            detail="CORKYSOFT_SERVICE_CREDENTIALS_JSON is invalid JSON",
        ) from exc
    if isinstance(parsed, dict):
        parsed = parsed.get("credentials", [])
    if not isinstance(parsed, list):
        raise HTTPException(
            status_code=503,
            detail="CORKYSOFT_SERVICE_CREDENTIALS_JSON must define a credentials list",
        )
    credentials = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        token = str(item.get("token") or "").strip()
        credential_id = str(item.get("id") or item.get("credential_id") or "").strip()
        actor = str(item.get("actor") or "").strip()
        scopes = item.get("scopes") or []
        if isinstance(scopes, str):
            scopes = [scopes]
        if not token or not credential_id or not actor or not isinstance(scopes, list):
            continue
        credentials.append(
            {
                "id": credential_id,
                "token": token,
                "actor": actor,
                "scopes": tuple(str(scope).strip() for scope in scopes if str(scope).strip()),
            }
        )
    return credentials


def _request_id(value: str | None) -> str:
    cleaned = (value or "").strip()
    return cleaned or f"req_{secrets.token_hex(12)}"


def require_api_auth_context(
    required_scopes: Iterable[str],
    *,
    allow_legacy_token: bool = True,
):
    required = tuple(required_scopes)

    def dependency(
        x_corkysoft_api_key: Optional[str] = Header(
            default=None, alias="X-Corkysoft-Api-Key"
        ),
        x_corkysoft_request_id: Optional[str] = Header(
            default=None, alias="X-Corkysoft-Request-Id"
        ),
    ) -> ApiAuthContext:
        request_id = _request_id(x_corkysoft_request_id)
        for credential in _service_credentials():
            if x_corkysoft_api_key != credential["token"]:
                continue
            scopes = credential["scopes"]
            missing = [scope for scope in required if scope not in scopes]
            if missing:
                raise HTTPException(status_code=403, detail="Credential scope is not authorized")
            return ApiAuthContext(
                credential_id=credential["id"],
                actor=credential["actor"],
                scopes=scopes,
                request_id=request_id,
            )
        if allow_legacy_token:
            expected = _required_internal_api_token()
            if x_corkysoft_api_key == expected:
                scopes = tuple(sorted({API_READ_SCOPE, *required}))
                return ApiAuthContext(
                    credential_id="legacy-internal-token",
                    actor="legacy-internal-api",
                    scopes=scopes,
                    request_id=request_id,
                    legacy=True,
                )
        raise HTTPException(status_code=401, detail="Invalid internal API token")

    return dependency


def ensure_api_write_receipts_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_write_receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            credential_id TEXT NOT NULL,
            actor TEXT NOT NULL,
            scopes_json TEXT NOT NULL,
            action TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            resource_id TEXT NOT NULL,
            request_id TEXT NOT NULL,
            route TEXT,
            method TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_api_write_receipts_request
        ON api_write_receipts(request_id, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_api_write_receipts_resource
        ON api_write_receipts(resource_type, resource_id, created_at DESC)
        """
    )
    conn.commit()


def record_api_write_receipt(
    conn: sqlite3.Connection,
    *,
    auth: ApiAuthContext,
    action: str,
    resource_type: str,
    resource_id: str,
    request: Request | None = None,
) -> dict:
    ensure_api_write_receipts_table(conn)
    route = str(request.url.path) if request is not None else None
    method = str(request.method) if request is not None else None
    cursor = conn.execute(
        """
        INSERT INTO api_write_receipts (
            credential_id,
            actor,
            scopes_json,
            action,
            resource_type,
            resource_id,
            request_id,
            route,
            method
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            auth.credential_id,
            auth.actor,
            json.dumps(list(auth.scopes)),
            action,
            resource_type,
            str(resource_id),
            auth.request_id,
            route,
            method,
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM api_write_receipts WHERE id = ?",
        (int(cursor.lastrowid),),
    ).fetchone()
    return dict(row) if row is not None else {}
