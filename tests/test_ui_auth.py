from __future__ import annotations

from analytics.auth import (
    bootstrap_dashboard_admin,
    get_dashboard_user_by_email,
    record_dashboard_user_login,
    resolve_ui_auth_policy,
    upsert_dashboard_user,
)
from analytics.db.schema import ensure_dashboard_tables
from analytics.db_connection import get_connection


def test_dashboard_user_table_and_upsert_round_trip(tmp_path) -> None:
    conn = get_connection(str(tmp_path / "auth.db"))
    try:
        ensure_dashboard_tables(conn)
        user = upsert_dashboard_user(
            conn,
            email="Ops@example.com",
            display_name="Ops User",
            role_key="dispatcher",
            active=True,
            allowed_role_keys=("dispatcher", "system_rollout_admin"),
        )
        assert user["email"] == "ops@example.com"
        assert user["roleKey"] == "dispatcher"

        refreshed = get_dashboard_user_by_email(conn, email="OPS@example.com")
        assert refreshed is not None
        assert refreshed["displayName"] == "Ops User"
    finally:
        conn.close()


def test_record_dashboard_user_login_updates_subject_and_last_login(tmp_path) -> None:
    conn = get_connection(str(tmp_path / "auth-login.db"))
    try:
        ensure_dashboard_tables(conn)
        upsert_dashboard_user(
            conn,
            email="manager@example.com",
            display_name="Manager",
            role_key="system_rollout_admin",
            active=True,
            allowed_role_keys=("dispatcher", "system_rollout_admin"),
        )

        updated = record_dashboard_user_login(
            conn,
            email="manager@example.com",
            google_sub="google-123",
            display_name="Manager Name",
        )
        assert updated is not None
        assert updated["googleSub"] == "google-123"
        assert updated["lastLoginAt"] is not None
        assert updated["displayName"] == "Manager Name"
    finally:
        conn.close()


def test_resolve_ui_auth_policy_fails_closed_outside_development(monkeypatch) -> None:
    monkeypatch.setenv("CORKYSOFT_ENV", "production")
    monkeypatch.setenv("CORKYSOFT_ALLOW_ANONYMOUS_UI", "1")
    try:
        resolve_ui_auth_policy()
    except ValueError as exc:
        assert "only permitted" in str(exc)
    else:
        raise AssertionError("Expected production anonymous UI configuration to fail")


def test_bootstrap_dashboard_admin_from_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CORKYSOFT_ENV", "production")
    monkeypatch.delenv("CORKYSOFT_ALLOW_ANONYMOUS_UI", raising=False)
    monkeypatch.setenv("CORKYSOFT_BOOTSTRAP_ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("CORKYSOFT_BOOTSTRAP_ADMIN_NAME", "Admin User")
    monkeypatch.setenv("CORKYSOFT_BOOTSTRAP_ADMIN_ROLE", "system_rollout_admin")

    conn = get_connection(str(tmp_path / "bootstrap.db"))
    try:
        ensure_dashboard_tables(conn)
        seeded = bootstrap_dashboard_admin(
            conn,
            allowed_role_keys=("dispatcher", "system_rollout_admin"),
        )
        assert seeded is not None
        assert seeded["email"] == "admin@example.com"
        assert seeded["roleKey"] == "system_rollout_admin"
    finally:
        conn.close()


def test_bootstrap_dashboard_admin_does_not_override_existing_users(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CORKYSOFT_ENV", "production")
    monkeypatch.delenv("CORKYSOFT_ALLOW_ANONYMOUS_UI", raising=False)
    monkeypatch.setenv("CORKYSOFT_BOOTSTRAP_ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("CORKYSOFT_BOOTSTRAP_ADMIN_NAME", "Admin User")
    monkeypatch.setenv("CORKYSOFT_BOOTSTRAP_ADMIN_ROLE", "system_rollout_admin")

    conn = get_connection(str(tmp_path / "bootstrap-existing.db"))
    try:
        ensure_dashboard_tables(conn)
        upsert_dashboard_user(
            conn,
            email="other@example.com",
            display_name="Existing User",
            role_key="dispatcher",
            active=True,
            allowed_role_keys=("dispatcher", "system_rollout_admin"),
        )

        seeded = bootstrap_dashboard_admin(
            conn,
            allowed_role_keys=("dispatcher", "system_rollout_admin"),
        )
        assert seeded is None
        assert get_dashboard_user_by_email(conn, email="admin@example.com") is None
    finally:
        conn.close()
