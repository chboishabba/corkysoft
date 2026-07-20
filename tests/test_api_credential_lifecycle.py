from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from corkysoft.api_shared import (
    API_READ_SCOPE,
    require_api_auth_context,
    require_internal_api_read_token,
    require_internal_api_token,
)


def _set_credentials(monkeypatch, credentials: list[dict]) -> None:
    monkeypatch.setenv(
        "CORKYSOFT_SERVICE_CREDENTIALS_JSON",
        json.dumps({"credentials": credentials}),
    )
    monkeypatch.setenv("CORKYSOFT_API_TOKEN", "legacy-token")
    monkeypatch.delenv("CORKYSOFT_ALLOW_LEGACY_API_TOKEN", raising=False)
    monkeypatch.delenv("CORKYSOFT_ALLOW_LEGACY_API_WRITE_TOKEN", raising=False)


def _credential(
    *,
    credential_id: str,
    token: str,
    status: str = "active",
    not_before: str | None = None,
    expires_at: str | None = None,
    revoked_at: str | None = None,
) -> dict:
    return {
        "id": credential_id,
        "token": token,
        "actor": f"actor:{credential_id}",
        "scopes": [API_READ_SCOPE, "calls:write"],
        "status": status,
        "not_before": not_before,
        "expires_at": expires_at,
        "revoked_at": revoked_at,
    }


def test_rotation_overlap_accepts_old_and_new_credentials(monkeypatch) -> None:
    _set_credentials(
        monkeypatch,
        [
            _credential(credential_id="old", token="old-token", status="overlap"),
            _credential(credential_id="new", token="new-token"),
        ],
    )
    dependency = require_api_auth_context(["calls:write"], allow_legacy_token=False)

    assert dependency("old-token", "req-old").credential_id == "old"
    assert dependency("new-token", "req-new").credential_id == "new"


@pytest.mark.parametrize(
    ("credential", "detail"),
    [
        (_credential(credential_id="revoked", token="token", status="revoked"), "revoked"),
        (_credential(credential_id="disabled", token="token", status="disabled"), "revoked"),
        (_credential(credential_id="retired", token="token", status="retired"), "revoked"),
        (_credential(credential_id="unknown", token="token", status="unknown"), "not active"),
        (
            _credential(
                credential_id="revoked-at",
                token="token",
                revoked_at="2026-07-17T00:00:00Z",
            ),
            "revoked",
        ),
        (
            _credential(
                credential_id="expired",
                token="token",
                expires_at="2020-01-01T00:00:00Z",
            ),
            "expired",
        ),
        (
            _credential(
                credential_id="future",
                token="token",
                not_before="2999-01-01T00:00:00Z",
            ),
            "not active yet",
        ),
    ],
)
def test_unusable_credentials_fail_closed(
    monkeypatch,
    credential: dict,
    detail: str,
) -> None:
    _set_credentials(monkeypatch, [credential])
    dependency = require_api_auth_context(["calls:write"], allow_legacy_token=False)

    with pytest.raises(HTTPException) as exc_info:
        dependency("token", "req-denied")

    assert exc_info.value.status_code == 401
    assert detail in str(exc_info.value.detail).lower()


def test_wrong_scope_is_forbidden_after_lifecycle_validation(monkeypatch) -> None:
    credential = _credential(credential_id="reader", token="reader-token")
    credential["scopes"] = [API_READ_SCOPE]
    _set_credentials(monkeypatch, [credential])
    dependency = require_api_auth_context(["calls:write"], allow_legacy_token=False)

    with pytest.raises(HTTPException) as exc_info:
        dependency("reader-token", "req-scope")

    assert exc_info.value.status_code == 403


def test_scoped_write_legacy_token_is_disabled_by_default(monkeypatch) -> None:
    _set_credentials(monkeypatch, [])
    dependency = require_api_auth_context(["calls:write"], allow_legacy_token=True)

    with pytest.raises(HTTPException) as exc_info:
        dependency("legacy-token", "req-legacy")

    assert exc_info.value.status_code == 401


def test_general_legacy_switch_supports_bounded_migration(monkeypatch) -> None:
    _set_credentials(monkeypatch, [])
    monkeypatch.setenv("CORKYSOFT_ALLOW_LEGACY_API_TOKEN", "1")
    dependency = require_api_auth_context(["calls:write"], allow_legacy_token=True)

    context = dependency("legacy-token", "req-legacy")

    assert context.legacy is True
    assert context.credential_id == "legacy-internal-token"


def test_direct_legacy_token_can_be_explicitly_disabled(monkeypatch) -> None:
    _set_credentials(monkeypatch, [])
    monkeypatch.setenv("CORKYSOFT_ALLOW_LEGACY_API_TOKEN", "0")

    with pytest.raises(HTTPException) as exc_info:
        require_internal_api_token("legacy-token")

    assert exc_info.value.status_code == 401


def test_sensitive_read_rejects_revoked_credential(monkeypatch) -> None:
    _set_credentials(
        monkeypatch,
        [_credential(credential_id="revoked-reader", token="reader-token", status="revoked")],
    )

    with pytest.raises(HTTPException) as exc_info:
        require_internal_api_read_token("reader-token")

    assert exc_info.value.status_code == 401
    assert "revoked" in str(exc_info.value.detail).lower()


@pytest.mark.parametrize("duplicate_field", ["token", "id"])
def test_duplicate_credential_identity_fails_configuration_closed(
    monkeypatch,
    duplicate_field: str,
) -> None:
    first = _credential(credential_id="one", token="token-one")
    second = _credential(credential_id="two", token="token-two")
    second[duplicate_field] = first[duplicate_field]
    _set_credentials(monkeypatch, [first, second])
    dependency = require_api_auth_context(["calls:write"], allow_legacy_token=False)

    with pytest.raises(HTTPException) as exc_info:
        dependency(first["token"], "req-duplicate")

    assert exc_info.value.status_code == 503
    assert "unique" in str(exc_info.value.detail).lower()


def test_lifecycle_timestamp_requires_timezone(monkeypatch) -> None:
    _set_credentials(
        monkeypatch,
        [
            _credential(
                credential_id="naive",
                token="naive-token",
                expires_at="2026-07-20T12:00:00",
            )
        ],
    )
    dependency = require_api_auth_context(["calls:write"], allow_legacy_token=False)

    with pytest.raises(HTTPException) as exc_info:
        dependency("naive-token", "req-naive")

    assert exc_info.value.status_code == 503
    assert "timezone" in str(exc_info.value.detail).lower()
