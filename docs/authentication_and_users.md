# Authentication And Users

Corkysoft now treats dashboard access as a real application boundary rather than a purely anonymous Streamlit session.

## Model

- Google sign-in is the identity provider.
- Streamlit OIDC owns the provider-facing login flow and identity cookie.
- Corkysoft owns authorization through a local `dashboard_users` table.
- Roles still map onto the existing dashboard role-layout system.
- Role-hidden tabs should be treated as part of the authorization boundary for auth-sensitive surfaces; query params must not re-expose them.

That means:

- a Google account is not enough on its own
- the email must exist locally as an active Corkysoft user
- the local user record determines the dashboard role key

## Local User Records

`dashboard_users` stores:

- email
- display name
- auth provider
- Google subject id
- role key
- active flag
- created/updated timestamps
- last login timestamp

This keeps access reviewable and lets Corkysoft disable a user without depending on Google-group lookups.

## Environment Rules

Shared/deployed environments:

- should run with UI auth required
- should not expose anonymous access
- should rely on Streamlit OIDC secrets for Google login
- should align the configured OIDC `redirect_uri` with the actual public origin used by operators
- may temporarily enable `CORKYSOFT_AUTO_PROVISION_GOOGLE_ADMIN=1` during tightly controlled owner/testing phases when every successful Google login should land as local admin

Local development:

- may run anonymously only when both:
  - `CORKYSOFT_ENV=development`
  - `CORKYSOFT_ALLOW_ANONYMOUS_UI=1`
- should use the normal Google sign-in path otherwise
- should display an explicit in-app banner stating that Google sign-in is being bypassed for local development

Temporary owner/testing shortcut:

- `CORKYSOFT_AUTO_PROVISION_GOOGLE_ADMIN=1` auto-creates any successfully signed-in Google user as a local `system_rollout_admin`
- this keeps the local `dashboard_users` table populated so the shortcut is easier to unwind later than a pure auth bypass
- the dashboard should display an explicit warning banner while this mode is active
- this is a temporary sharing/testing posture, not the intended long-term authorization model

Tunneled or remote development:

- localhost-only OIDC settings are not sufficient when the app is exposed through a tunnel or other remote origin
- if the public origin does not match the configured `redirect_uri`, Streamlit/OIDC may fail with origin-mismatch behavior even though the app shell loads
- this should be treated as configuration error, not as a valid partially authenticated mode

Authenticated runs should also display:

- authenticated via Google
- signed-in email
- resolved Corkysoft role
- logout control

## Bootstrap

The first admin user should be seeded explicitly with environment variables rather than by allowing arbitrary first-login account creation.

Supported bootstrap fields:

- `CORKYSOFT_BOOTSTRAP_ADMIN_EMAIL`
- `CORKYSOFT_BOOTSTRAP_ADMIN_NAME`
- `CORKYSOFT_BOOTSTRAP_ADMIN_ROLE`

This keeps the first-user path explicit and avoids an accidental permissive sign-up flow.
Bootstrap seeding should only act as a first-user path; once dashboard users exist,
lingering bootstrap env vars should not keep reasserting or mutating admin access.

## Admin Surface

User management belongs in the dashboard admin surface:

- list users
- change role
- activate/deactivate users
- review last login timestamps

This is separate from Google account management. Google proves identity; Corkysoft decides access.

## Red-Team Priorities

The first auth red-team wave should focus on:

- anonymous-mode leakage into shared/deployed runs
- allowlist enforcement for unauthorized or inactive Google accounts
- hidden-tab exposure via query params or stale session state
- bootstrap-admin env misuse
- low-privileged users reaching admin-only user-management behavior
- remote-origin / `redirect_uri` mismatch clarity when the app is exposed through a tunnel or public URL

Near-term execution order:

1. keep the implemented local Playwright auth harness current as auth behavior changes
2. extend coverage from the local harness into real Google-backed browser automation
3. add deployed/tunneled origin verification against real OIDC settings when deployment posture matters
4. follow with deeper audit attribution and broader admin-governance hardening

See [Auth Red-Team Plan](auth_red_team_plan.md).
