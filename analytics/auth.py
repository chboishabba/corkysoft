"""Dashboard authentication and local user authorization helpers."""
from __future__ import annotations

import os
import sqlite3
from typing import Any, Optional, Sequence


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_bool(value: str | None) -> Optional[bool]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def normalize_user_email(email: str | None) -> str | None:
    if email is None:
        return None
    normalized = str(email).strip().lower()
    return normalized or None


def resolve_ui_auth_policy() -> dict[str, Any]:
    environment = (os.environ.get("CORKYSOFT_ENV") or "production").strip().lower()
    allow_anonymous = bool(_parse_bool(os.environ.get("CORKYSOFT_ALLOW_ANONYMOUS_UI")))
    require_auth_env = _parse_bool(os.environ.get("CORKYSOFT_REQUIRE_UI_AUTH"))
    require_auth = require_auth_env if require_auth_env is not None else environment != "development"

    if environment != "development" and allow_anonymous:
        raise ValueError(
            "CORKYSOFT_ALLOW_ANONYMOUS_UI is only permitted when CORKYSOFT_ENV=development"
        )
    if environment != "development" and require_auth is False:
        raise ValueError(
            "Shared/deployed environments must not disable UI auth"
        )

    return {
        "environment": environment,
        "allowAnonymous": allow_anonymous,
        "requireAuth": bool(require_auth),
        "autoProvisionGoogleAdmin": bool(
            _parse_bool(os.environ.get("CORKYSOFT_AUTO_PROVISION_GOOGLE_ADMIN"))
        ),
        "enableTestAuth": bool(
            environment == "development"
            and _parse_bool(os.environ.get("CORKYSOFT_ENABLE_TEST_AUTH"))
        ),
        "publicBaseUrl": (os.environ.get("CORKYSOFT_PUBLIC_BASE_URL") or "").strip() or None,
        "bootstrapAdminEmail": normalize_user_email(os.environ.get("CORKYSOFT_BOOTSTRAP_ADMIN_EMAIL")),
        "bootstrapAdminName": (os.environ.get("CORKYSOFT_BOOTSTRAP_ADMIN_NAME") or "").strip() or None,
        "bootstrapAdminRole": (os.environ.get("CORKYSOFT_BOOTSTRAP_ADMIN_ROLE") or "system_rollout_admin").strip() or "system_rollout_admin",
    }


def resolve_test_auth_override(
    conn: sqlite3.Connection,
    *,
    allowed_role_keys: Sequence[str],
) -> Optional[dict[str, Any]]:
    policy = resolve_ui_auth_policy()
    if not policy["enableTestAuth"]:
        return None

    mode = (os.environ.get("CORKYSOFT_TEST_AUTH_MODE") or "").strip().lower()
    if not mode:
        return None

    email = normalize_user_email(os.environ.get("CORKYSOFT_TEST_AUTH_EMAIL")) or "playwright@example.com"
    display_name = (os.environ.get("CORKYSOFT_TEST_AUTH_NAME") or "Playwright User").strip() or "Playwright User"
    role_key = (os.environ.get("CORKYSOFT_TEST_AUTH_ROLE") or "dispatcher").strip() or "dispatcher"
    if role_key not in set(allowed_role_keys):
        raise ValueError(f"Unknown test auth role: {role_key}")

    base_claims = {
        "email": email,
        "name": display_name,
        "sub": f"test-sub:{email}",
        "is_logged_in": mode not in {"login_required", "misconfigured"},
    }
    base_state = {
        "policy": policy,
        "configured": mode != "misconfigured",
        "claims": base_claims,
    }

    if mode == "anonymous":
        return {
            **base_state,
            "mode": "anonymous",
            "user": None,
            "claims": {},
        }
    if mode == "misconfigured":
        return {
            **base_state,
            "mode": "misconfigured",
            "user": None,
            "claims": {},
            "detail": "Streamlit OIDC test mode: auth secrets are intentionally unavailable.",
        }
    if mode == "login_required":
        return {
            **base_state,
            "mode": "login_required",
            "user": None,
        }
    if mode == "unauthorized":
        return {
            **base_state,
            "mode": "unauthorized",
            "user": None,
        }

    active = mode != "inactive"
    user = upsert_dashboard_user(
        conn,
        email=email,
        display_name=display_name,
        role_key=role_key,
        active=active,
        auth_provider="test",
        google_sub=str(base_claims["sub"]),
        allowed_role_keys=allowed_role_keys,
    )
    if not active:
        return {
            **base_state,
            "mode": "inactive",
            "user": user,
        }
    if mode == "authenticated":
        refreshed_user = record_dashboard_user_login(
            conn,
            email=email,
            google_sub=str(base_claims["sub"]),
            display_name=display_name,
        ) or user
        return {
            **base_state,
            "mode": "authenticated",
            "user": refreshed_user,
        }
    raise ValueError(f"Unknown test auth mode: {mode}")


def get_dashboard_user_by_email(
    conn: sqlite3.Connection,
    *,
    email: str | None,
) -> Optional[dict[str, Any]]:
    normalized = normalize_user_email(email)
    if normalized is None:
        return None
    row = conn.execute(
        """
        SELECT id, email, display_name, auth_provider, google_sub, role_key, active,
               created_at, updated_at, last_login_at
        FROM dashboard_users
        WHERE email = ?
        """,
        (normalized,),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": int(row["id"]),
        "email": str(row["email"]),
        "displayName": row["display_name"],
        "authProvider": str(row["auth_provider"]),
        "googleSub": row["google_sub"],
        "roleKey": str(row["role_key"]),
        "active": bool(row["active"]),
        "createdAt": str(row["created_at"]),
        "updatedAt": str(row["updated_at"]),
        "lastLoginAt": row["last_login_at"],
    }


def list_dashboard_users(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, email, display_name, auth_provider, google_sub, role_key, active,
               created_at, updated_at, last_login_at
        FROM dashboard_users
        ORDER BY active DESC, role_key, email
        """
    ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "email": str(row["email"]),
            "displayName": row["display_name"],
            "authProvider": str(row["auth_provider"]),
            "googleSub": row["google_sub"],
            "roleKey": str(row["role_key"]),
            "active": bool(row["active"]),
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"]),
            "lastLoginAt": row["last_login_at"],
        }
        for row in rows
    ]


def upsert_dashboard_user(
    conn: sqlite3.Connection,
    *,
    email: str,
    role_key: str,
    display_name: str | None = None,
    active: bool = True,
    auth_provider: str = "google",
    google_sub: str | None = None,
    allowed_role_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    normalized_email = normalize_user_email(email)
    if normalized_email is None:
        raise ValueError("Email is required")
    normalized_role = str(role_key).strip()
    if not normalized_role:
        raise ValueError("Role key is required")
    if allowed_role_keys is not None and normalized_role not in set(allowed_role_keys):
        raise ValueError(f"Unknown dashboard role: {normalized_role}")

    timestamp = _utc_now_iso()
    existing = conn.execute(
        "SELECT id FROM dashboard_users WHERE email = ?",
        (normalized_email,),
    ).fetchone()
    if existing is None:
        conn.execute(
            """
            INSERT INTO dashboard_users (
                email,
                display_name,
                auth_provider,
                google_sub,
                role_key,
                active,
                created_at,
                updated_at,
                last_login_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                normalized_email,
                (display_name or "").strip() or None,
                str(auth_provider).strip() or "google",
                (google_sub or "").strip() or None,
                normalized_role,
                1 if active else 0,
                timestamp,
                timestamp,
            ),
        )
    else:
        conn.execute(
            """
            UPDATE dashboard_users
            SET display_name = ?,
                auth_provider = ?,
                google_sub = COALESCE(?, google_sub),
                role_key = ?,
                active = ?,
                updated_at = ?
            WHERE email = ?
            """,
            (
                (display_name or "").strip() or None,
                str(auth_provider).strip() or "google",
                (google_sub or "").strip() or None,
                normalized_role,
                1 if active else 0,
                timestamp,
                normalized_email,
            ),
        )
    conn.commit()
    user = get_dashboard_user_by_email(conn, email=normalized_email)
    if user is None:
        raise RuntimeError("Failed to persist dashboard user")
    return user


def record_dashboard_user_login(
    conn: sqlite3.Connection,
    *,
    email: str | None,
    google_sub: str | None = None,
    display_name: str | None = None,
) -> Optional[dict[str, Any]]:
    normalized_email = normalize_user_email(email)
    if normalized_email is None:
        return None
    timestamp = _utc_now_iso()
    conn.execute(
        """
        UPDATE dashboard_users
        SET display_name = COALESCE(?, display_name),
            google_sub = COALESCE(?, google_sub),
            last_login_at = ?,
            updated_at = ?
        WHERE email = ?
        """,
        (
            (display_name or "").strip() or None,
            (google_sub or "").strip() or None,
            timestamp,
            timestamp,
            normalized_email,
        ),
    )
    conn.commit()
    return get_dashboard_user_by_email(conn, email=normalized_email)


def bootstrap_dashboard_admin(conn: sqlite3.Connection, *, allowed_role_keys: Sequence[str]) -> Optional[dict[str, Any]]:
    policy = resolve_ui_auth_policy()
    bootstrap_email = policy["bootstrapAdminEmail"]
    if bootstrap_email is None:
        return None
    existing_user_count = int(
        conn.execute("SELECT COUNT(*) FROM dashboard_users").fetchone()[0]
    )
    if existing_user_count > 0:
        return get_dashboard_user_by_email(conn, email=bootstrap_email)
    role_key = str(policy["bootstrapAdminRole"])
    if role_key not in set(allowed_role_keys):
        raise ValueError(f"Unknown bootstrap admin role: {role_key}")
    return upsert_dashboard_user(
        conn,
        email=bootstrap_email,
        display_name=policy["bootstrapAdminName"],
        role_key=role_key,
        active=True,
        auth_provider="google",
        allowed_role_keys=allowed_role_keys,
    )


def auto_provision_google_admin_user(
    conn: sqlite3.Connection,
    *,
    email: str | None,
    google_sub: str | None,
    display_name: str | None,
    allowed_role_keys: Sequence[str],
) -> Optional[dict[str, Any]]:
    policy = resolve_ui_auth_policy()
    if not policy["autoProvisionGoogleAdmin"]:
        return None
    normalized_email = normalize_user_email(email)
    if normalized_email is None:
        return None
    return upsert_dashboard_user(
        conn,
        email=normalized_email,
        display_name=display_name,
        role_key="system_rollout_admin",
        active=True,
        auth_provider="google",
        google_sub=google_sub,
        allowed_role_keys=allowed_role_keys,
    )


__all__ = [
    "auto_provision_google_admin_user",
    "bootstrap_dashboard_admin",
    "get_dashboard_user_by_email",
    "list_dashboard_users",
    "normalize_user_email",
    "record_dashboard_user_login",
    "resolve_test_auth_override",
    "resolve_ui_auth_policy",
    "upsert_dashboard_user",
]
