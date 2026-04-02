"""Auth helpers that keep the dashboard auth gate and user banner logic isolated."""
from __future__ import annotations

import sqlite3
from typing import Any, Dict

import pandas as pd
import streamlit as st

from analytics.dashboard_layouts import ROLE_LAYOUT_DEFAULTS
from analytics.db import (
    auto_provision_google_admin_user,
    bootstrap_dashboard_admin,
    get_dashboard_user_by_email,
    list_dashboard_users,
    normalize_user_email,
    record_dashboard_user_login,
    resolve_test_auth_override,
    resolve_ui_auth_policy,
    upsert_dashboard_user,
)
from dashboard.state import _rerun_app

__all__ = [
    "_auth_redirect_config_issue",
    "_render_anonymous_dev_banner",
    "_render_authenticated_user_banner",
    "_render_authenticated_user_sidebar_card",
    "_render_auth_gate",
    "_render_dashboard_user_admin",
    "_resolve_dashboard_identity",
    "_streamlit_auth_configured",
    "_streamlit_user_claims",
]


def _streamlit_auth_configured() -> bool:
    user_obj = getattr(st, "user", None)
    return getattr(user_obj, "is_logged_in", None) is not None


def _streamlit_user_claims() -> Dict[str, Any]:
    user_obj = getattr(st, "user", None)
    if user_obj is None:
        return {}
    claims: Dict[str, Any] = {}
    for key in ("email", "name", "sub"):
        value = getattr(user_obj, key, None)
        if isinstance(value, str) and value.strip():
            claims[key] = value.strip()
    claims["is_logged_in"] = bool(getattr(user_obj, "is_logged_in", False))
    return claims


def _resolve_dashboard_identity(
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    policy = resolve_ui_auth_policy()
    bootstrap_dashboard_admin(conn, allowed_role_keys=tuple(ROLE_LAYOUT_DEFAULTS.keys()))
    test_override = resolve_test_auth_override(
        conn,
        allowed_role_keys=tuple(ROLE_LAYOUT_DEFAULTS.keys()),
    )
    if test_override is not None:
        return test_override

    if not policy["requireAuth"]:
        return {
            "mode": "anonymous",
            "policy": policy,
            "user": None,
            "claims": {},
            "configured": _streamlit_auth_configured(),
        }

    configured = _streamlit_auth_configured()
    claims = _streamlit_user_claims()
    redirect_config_issue = _auth_redirect_config_issue(policy)
    if redirect_config_issue is not None:
        return {
            "mode": "misconfigured",
            "policy": policy,
            "user": None,
            "claims": claims,
            "configured": configured,
            "detail": redirect_config_issue,
        }
    if not configured:
        return {
            "mode": "misconfigured",
            "policy": policy,
            "user": None,
            "claims": claims,
            "configured": False,
        }

    if not claims.get("is_logged_in"):
        return {
            "mode": "login_required",
            "policy": policy,
            "user": None,
            "claims": claims,
            "configured": True,
        }

    email = normalize_user_email(claims.get("email"))
    local_user = get_dashboard_user_by_email(conn, email=email)
    if local_user is None:
        local_user = auto_provision_google_admin_user(
            conn,
            email=email,
            google_sub=claims.get("sub"),
            display_name=claims.get("name"),
            allowed_role_keys=tuple(ROLE_LAYOUT_DEFAULTS.keys()),
        )
    if local_user is None:
        return {
            "mode": "unauthorized",
            "policy": policy,
            "user": None,
            "claims": claims,
            "configured": True,
        }
    if not local_user["active"]:
        return {
            "mode": "inactive",
            "policy": policy,
            "user": local_user,
            "claims": claims,
            "configured": True,
        }

    refreshed_user = record_dashboard_user_login(
        conn,
        email=email,
        google_sub=claims.get("sub"),
        display_name=claims.get("name"),
    ) or local_user
    return {
        "mode": "authenticated",
        "policy": policy,
        "user": refreshed_user,
        "claims": claims,
        "configured": True,
    }


def _render_auth_gate(auth_state: dict[str, Any]) -> None:
    st.title("Corkysoft")
    st.caption("Private dashboard access is controlled by Google sign-in plus a local Corkysoft allowlist.")

    mode = str(auth_state["mode"])
    if mode == "misconfigured":
        detail = str(auth_state.get("detail") or "").strip()
        st.error(
            "UI auth is required but Streamlit OIDC is not configured correctly. Add `.streamlit/secrets.toml` auth settings before starting this deployment."
        )
        if detail:
            st.caption(detail)
        st.stop()

    if mode == "login_required":
        st.info("Sign in with Google to continue.")
        login = getattr(st, "login", None)
        if not callable(login):
            st.error("Streamlit login support is unavailable in this runtime.")
        elif st.button("Sign in with Google", key="dashboard_auth_login_google"):
            login("google")
        st.stop()

    claims = auth_state.get("claims", {})
    email = claims.get("email") or "Unknown account"
    if mode == "unauthorized":
        st.error(f"`{email}` is not in the local Corkysoft allowlist.")
    elif mode == "inactive":
        st.error(f"`{email}` is currently inactive in Corkysoft.")
    else:
        st.error("Authentication failed.")

    logout = getattr(st, "logout", None)
    if callable(logout) and st.button("Sign out", key="dashboard_auth_logout_gate"):
        logout()
    st.stop()


def _render_authenticated_user_banner(auth_state: dict[str, Any]) -> None:
    _render_authenticated_user_sidebar_card(auth_state)


def _render_authenticated_user_sidebar_card(auth_state: dict[str, Any]) -> None:
    user = auth_state.get("user") or {}
    claims = auth_state.get("claims") or {}
    policy = auth_state.get("policy") or {}
    display_name = user.get("displayName") or claims.get("name") or user.get("email") or "Unknown user"
    email = user.get("email") or claims.get("email") or ""
    role_key = user.get("roleKey") or "dispatcher"
    role_label = ROLE_LAYOUT_DEFAULTS.get(role_key, {}).get("label", role_key)

    with st.sidebar:
        st.markdown("---")
        st.markdown("#### Account")
        st.caption(
            f"Authenticated via Google as {display_name} ({email}) · role: {role_label}"
        )
        st.caption(f"Google account: **{display_name}**")
        st.caption(f"{email} · {role_label}")
        action_cols = st.columns(2)
        logout = getattr(st, "logout", None)
        if callable(logout) and action_cols[0].button(
            "Sign out",
            key="dashboard_auth_logout_button",
            use_container_width=True,
        ):
            logout()
        action_cols[1].button(
            "Settings",
            key="dashboard_auth_settings_button",
            disabled=True,
            help="Account settings are not yet available in the dashboard UI.",
            use_container_width=True,
        )
        if policy.get("autoProvisionGoogleAdmin"):
            st.warning(
                "Temporary auth mode is active: any successful Google login is auto-provisioned locally as System / Rollout Admin."
            )


def _render_anonymous_dev_banner(auth_state: dict[str, Any]) -> None:
    policy = auth_state.get("policy") or {}
    environment = str(policy.get("environment") or "development")
    st.warning(
        "Anonymous development mode is active. Google sign-in is bypassed for this local run."
    )
    st.caption(
        f"Mode: anonymous local development · environment: {environment} · set `CORKYSOFT_REQUIRE_UI_AUTH=1` and unset `CORKYSOFT_ALLOW_ANONYMOUS_UI` to force login."
    )


def _auth_redirect_config_issue(policy: dict[str, Any]) -> str | None:
    public_base_url = str(policy.get("publicBaseUrl") or "").strip().rstrip("/")
    if not public_base_url:
        return None
    auth_section = getattr(st, "secrets", {}).get("auth", {})
    if not isinstance(auth_section, dict):
        return (
            "CORKYSOFT_PUBLIC_BASE_URL is set but Streamlit auth secrets are missing. "
            "Configure the deployed or tunneled redirect URI explicitly."
        )
    redirect_uri = str(auth_section.get("redirect_uri") or "").strip()
    if not redirect_uri:
        return (
            "CORKYSOFT_PUBLIC_BASE_URL is set but [auth].redirect_uri is missing from "
            ".streamlit/secrets.toml."
        )
    expected_redirect = f"{public_base_url}/oauth2callback"
    if redirect_uri.rstrip("/") != expected_redirect:
        return (
            "OIDC redirect URI does not match the configured public origin. "
            f"Expected `{expected_redirect}` but found `{redirect_uri}`."
        )
    return None


def _render_dashboard_user_admin(
    conn: sqlite3.Connection,
    *,
    current_role_key: str,
) -> None:
    if current_role_key != "system_rollout_admin":
        return

    st.markdown("#### Dashboard users")
    users = list_dashboard_users(conn)
    if users:
        users_df = pd.DataFrame(
            [
                {
                    "Email": item["email"],
                    "Name": item["displayName"] or "",
                    "Role": item["roleKey"],
                    "Active": item["active"],
                    "Provider": item["authProvider"],
                    "Google sub": item["googleSub"] or "",
                    "Last login": item["lastLoginAt"] or "",
                }
                for item in users
            ]
        )
        st.dataframe(users_df, width="stretch", hide_index=True)
    else:
        st.caption("No local dashboard users have been created yet.")

    with st.form("dashboard_user_admin_form"):
        user_cols = st.columns(4)
        email = user_cols[0].text_input("Email")
        display_name = user_cols[1].text_input("Name")
        role_key = user_cols[2].selectbox(
            "Role",
            options=list(ROLE_LAYOUT_DEFAULTS.keys()),
            format_func=lambda key: str(ROLE_LAYOUT_DEFAULTS[key]["label"]),
        )
        active = user_cols[3].checkbox("Active", value=True)
        if st.form_submit_button("Save dashboard user"):
            try:
                upsert_dashboard_user(
                    conn,
                    email=email,
                    display_name=display_name or None,
                    role_key=role_key,
                    active=active,
                    allowed_role_keys=tuple(ROLE_LAYOUT_DEFAULTS.keys()),
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success("Dashboard user saved.")
                _rerun_app()
