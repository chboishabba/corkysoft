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

Local development:

- may run anonymously only when both:
  - `CORKYSOFT_ENV=development`
  - `CORKYSOFT_ALLOW_ANONYMOUS_UI=1`
- should use the normal Google sign-in path otherwise
- should display an explicit in-app banner stating that Google sign-in is being bypassed for local development

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

See [Auth Red-Team Plan](auth_red_team_plan.md).
