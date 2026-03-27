# Auth Red-Team Plan

This document defines the first focused red-team pass for Corkysoft's new dashboard auth surfaces.

## Scope

In scope:

- Streamlit OIDC login/logout integration
- local `dashboard_users` allowlist and role binding
- anonymous local-development mode gating
- bootstrap-admin env path
- authenticated UI state and role-locked layout behavior
- dashboard user-management surface

Out of scope for the first pass:

- Google provider internals
- broader operational or financial workflow abuse outside auth-bound surfaces
- full Google-account automation in CI

## Threat Model

Primary attacker models:

- unauthenticated external user trying to bypass login or exploit config drift
- authenticated low-privilege user trying to exceed role/tab/admin limits

Secondary concern:

- operator/deployer mistakes that unintentionally leave an auth backdoor active

## High-Priority Checks

Configuration / startup:

- auth-required mode fails closed when OIDC secrets are missing
- shared/deployed environments cannot enable anonymous UI mode
- public/shared startup paths do not default into bypass behavior

Identity / authorization:

- unauthorized Google accounts are denied even after provider sign-in
- inactive local users are denied
- local role binding is the authority for dashboard layout/admin access
- hidden admin tabs cannot be re-exposed through query params or stale session layout state

Bootstrap / admin:

- bootstrap-admin env only creates the first admin record when no users exist
- lingering bootstrap env does not keep reasserting admin access after setup
- low-privilege users cannot reach dashboard-user management controls

Session / UX:

- UI clearly distinguishes authenticated Google mode from anonymous local-dev mode
- logout path is visible and functional in authenticated mode
- stale session/query state does not bypass current role restrictions
- tunneled or remote origins do not silently break auth without an explicit operator-visible explanation

## Manual Test Matrix

External:

- start auth-required mode with missing secrets and verify fail-closed behavior
- sign in with a non-allowlisted Google account and confirm denial
- sign in with an inactive local account and confirm denial
- try `?view=Kent admin` or other hidden-tab query params as a non-admin role

Low-privileged authenticated:

- log in as dispatcher or estimator and confirm admin-only user-management is unreachable
- change tabs/query params after login to verify hidden tab state is not escalated
- deactivate or downgrade a user and verify their next load reflects the new state

Operator misconfiguration:

- run local dev with anonymous mode and confirm warning banner is obvious
- run public/shared mode and confirm anonymous mode cannot be enabled
- leave bootstrap-admin env vars set after first-user creation and confirm they do not silently mutate existing users
- run a tunneled/remote URL with a mismatched OIDC `redirect_uri` and confirm the app fails clearly rather than looking half-authenticated

## Automated Coverage

Current automated coverage now includes:

- env-policy tests around anonymous mode and fail-closed startup semantics
- bootstrap-admin tests for valid first-user creation and no-op behavior once users exist
- layout-resolution tests proving hidden tabs are not re-exposed by query params
- source/unit tests confirming authenticated and anonymous auth-state banners remain present
- a test-only auth-state harness for browser automation so Playwright can exercise allowlist, inactive-user, and hidden-tab-denial flows without depending on live Google login
- Playwright checks for anonymous, misconfigured, unauthorized, inactive, and low-privilege role-hidden-tab scenarios

Later browser-based round:

- real Google-backed login/logout flows
- tunneled-origin / redirect mismatch verification against a deployed or tunnel-exposed run
- evidence capture for visible auth state, console output, websocket/auth errors, and redirect behavior
