from __future__ import annotations

import sqlite3

import pytest

from analytics.auth import (
    auto_provision_google_admin_user,
    get_dashboard_user_by_email,
    resolve_test_auth_override,
    resolve_ui_auth_policy,
)
from analytics.db import ensure_dashboard_tables


def test_resolve_ui_auth_policy_exposes_test_auth_and_public_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CORKYSOFT_ENV", "development")
    monkeypatch.setenv("CORKYSOFT_ALLOW_ANONYMOUS_UI", "1")
    monkeypatch.setenv("CORKYSOFT_ENABLE_TEST_AUTH", "1")
    monkeypatch.setenv("CORKYSOFT_PUBLIC_BASE_URL", "https://example.localhost.run")

    policy = resolve_ui_auth_policy()

    assert policy["enableTestAuth"] is True
    assert policy["publicBaseUrl"] == "https://example.localhost.run"


def test_resolve_ui_auth_policy_exposes_auto_provision_google_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CORKYSOFT_ENV", "production")
    monkeypatch.delenv("CORKYSOFT_ALLOW_ANONYMOUS_UI", raising=False)
    monkeypatch.setenv("CORKYSOFT_AUTO_PROVISION_GOOGLE_ADMIN", "1")

    policy = resolve_ui_auth_policy()

    assert policy["autoProvisionGoogleAdmin"] is True


def test_auto_provision_google_admin_user_creates_local_admin_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CORKYSOFT_ENV", "production")
    monkeypatch.delenv("CORKYSOFT_ALLOW_ANONYMOUS_UI", raising=False)
    monkeypatch.setenv("CORKYSOFT_REQUIRE_UI_AUTH", "1")
    monkeypatch.setenv("CORKYSOFT_AUTO_PROVISION_GOOGLE_ADMIN", "1")

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    provisioned = auto_provision_google_admin_user(
        conn,
        email="owner@example.com",
        google_sub="google-owner-sub",
        display_name="Business Owner",
        allowed_role_keys=("dispatcher", "system_rollout_admin"),
    )

    assert provisioned is not None
    assert provisioned["email"] == "owner@example.com"
    assert provisioned["roleKey"] == "system_rollout_admin"
    assert provisioned["active"] is True

    persisted = get_dashboard_user_by_email(conn, email="owner@example.com")
    assert persisted is not None
    assert persisted["googleSub"] == "google-owner-sub"


def test_resolve_test_auth_override_supports_authenticated_and_inactive_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CORKYSOFT_ENV", "development")
    monkeypatch.setenv("CORKYSOFT_ENABLE_TEST_AUTH", "1")
    monkeypatch.setenv("CORKYSOFT_REQUIRE_UI_AUTH", "1")
    monkeypatch.setenv("CORKYSOFT_TEST_AUTH_EMAIL", "dispatcher@example.com")
    monkeypatch.setenv("CORKYSOFT_TEST_AUTH_ROLE", "dispatcher")

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    monkeypatch.setenv("CORKYSOFT_TEST_AUTH_MODE", "authenticated")
    authenticated = resolve_test_auth_override(
        conn,
        allowed_role_keys=("dispatcher", "system_rollout_admin"),
    )
    assert authenticated is not None
    assert authenticated["mode"] == "authenticated"
    assert authenticated["user"]["roleKey"] == "dispatcher"
    assert authenticated["claims"]["email"] == "dispatcher@example.com"

    monkeypatch.setenv("CORKYSOFT_TEST_AUTH_MODE", "inactive")
    inactive = resolve_test_auth_override(
        conn,
        allowed_role_keys=("dispatcher", "system_rollout_admin"),
    )
    assert inactive is not None
    assert inactive["mode"] == "inactive"
    assert inactive["user"]["active"] is False


def test_resolve_test_auth_override_supports_unauthorized_and_misconfigured_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CORKYSOFT_ENV", "development")
    monkeypatch.setenv("CORKYSOFT_ENABLE_TEST_AUTH", "1")
    monkeypatch.setenv("CORKYSOFT_REQUIRE_UI_AUTH", "1")

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    monkeypatch.setenv("CORKYSOFT_TEST_AUTH_MODE", "unauthorized")
    unauthorized = resolve_test_auth_override(
        conn,
        allowed_role_keys=("dispatcher", "system_rollout_admin"),
    )
    assert unauthorized is not None
    assert unauthorized["mode"] == "unauthorized"
    assert unauthorized["user"] is None

    monkeypatch.setenv("CORKYSOFT_TEST_AUTH_MODE", "misconfigured")
    misconfigured = resolve_test_auth_override(
        conn,
        allowed_role_keys=("dispatcher", "system_rollout_admin"),
    )
    assert misconfigured is not None
    assert misconfigured["mode"] == "misconfigured"
    assert "auth secrets" in misconfigured["detail"]
