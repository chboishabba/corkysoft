import { expect, test } from '@playwright/test';

import { startStreamlitHarness } from './support/streamlitHarness';

const PORT_BASE = 8610;

test.describe.configure({ mode: 'serial' });

async function withHarness(
  portOffset: number,
  env: Record<string, string | undefined>,
  run: (baseUrl: string) => Promise<void>,
): Promise<void> {
  const harness = await startStreamlitHarness(PORT_BASE + portOffset, env);
  try {
    await run(harness.baseUrl);
  } finally {
    await harness.cleanup();
  }
}

test('anonymous local-development banner/path', async ({ page }) => {
  await withHarness(
    1,
    {
      CORKYSOFT_ALLOW_ANONYMOUS_UI: '1',
    },
    async (baseUrl) => {
      await page.goto(baseUrl, { waitUntil: 'commit' });

      await expect(page.getByText(/Anonymous development mode is active/i)).toBeVisible();
      await expect(page.getByText(/Mode: anonymous local development/i)).toBeVisible();
      await expect(page).toHaveURL(baseUrl);
    },
  );
});

test('auth-required misconfigured gate', async ({ page }) => {
  await withHarness(
    2,
    {
      CORKYSOFT_ENABLE_TEST_AUTH: '1',
      CORKYSOFT_REQUIRE_UI_AUTH: '1',
      CORKYSOFT_TEST_AUTH_MODE: 'misconfigured',
    },
    async (baseUrl) => {
      await page.goto(baseUrl, { waitUntil: 'commit' });

      await expect(page.getByText(/UI auth is required but Streamlit OIDC is not configured correctly/i)).toBeVisible();
      await expect(page.getByText(/auth secrets are intentionally unavailable/i)).toBeVisible();
      await expect(page.getByText('Price distribution (Airbnb-style)')).toHaveCount(0);
    },
  );
});

test('unauthorized account denial', async ({ page }) => {
  await withHarness(
    3,
    {
      CORKYSOFT_ENABLE_TEST_AUTH: '1',
      CORKYSOFT_REQUIRE_UI_AUTH: '1',
      CORKYSOFT_TEST_AUTH_MODE: 'unauthorized',
      CORKYSOFT_TEST_AUTH_EMAIL: 'unauthorized@example.com',
      CORKYSOFT_TEST_AUTH_NAME: 'Unauthorized User',
    },
    async (baseUrl) => {
      await page.goto(baseUrl, { waitUntil: 'commit' });

      await expect(page.getByText(/unauthorized@example\.com/i)).toBeVisible();
      await expect(page.getByText(/is not in the local Corkysoft allowlist/i)).toBeVisible();
      await expect(page.getByText('Price distribution (Airbnb-style)')).toHaveCount(0);
    },
  );
});

test('inactive account denial', async ({ page }) => {
  await withHarness(
    4,
    {
      CORKYSOFT_ENABLE_TEST_AUTH: '1',
      CORKYSOFT_REQUIRE_UI_AUTH: '1',
      CORKYSOFT_TEST_AUTH_MODE: 'inactive',
      CORKYSOFT_TEST_AUTH_EMAIL: 'inactive@example.com',
      CORKYSOFT_TEST_AUTH_NAME: 'Inactive User',
      CORKYSOFT_TEST_AUTH_ROLE: 'dispatcher',
    },
    async (baseUrl) => {
      await page.goto(baseUrl, { waitUntil: 'commit' });

      await expect(page.getByText(/inactive@example\.com/i)).toBeVisible();
      await expect(page.getByText(/is currently inactive in Corkysoft/i)).toBeVisible();
      await expect(page.getByText('Price distribution (Airbnb-style)')).toHaveCount(0);
    },
  );
});

test('hidden-tab query-param denial for low-privilege role', async ({ page }) => {
  await withHarness(
    5,
    {
      CORKYSOFT_ENABLE_TEST_AUTH: '1',
      CORKYSOFT_REQUIRE_UI_AUTH: '1',
      CORKYSOFT_TEST_AUTH_MODE: 'authenticated',
      CORKYSOFT_TEST_AUTH_EMAIL: 'dispatcher@example.com',
      CORKYSOFT_TEST_AUTH_NAME: 'Dispatcher User',
      CORKYSOFT_TEST_AUTH_ROLE: 'dispatcher',
    },
    async (baseUrl) => {
      const deniedUrl = new URL(baseUrl);
      deniedUrl.searchParams.set('view', 'Kent admin');
      await page.goto(deniedUrl.toString(), { waitUntil: 'commit' });

      await expect(page.getByText(/Authenticated via Google as Dispatcher User/i)).toBeVisible();
      await expect(page.getByRole('tab', { name: 'Dispatch' })).toBeVisible();
      await expect(page.getByRole('tab', { name: 'Kent admin' })).toHaveCount(0);
      await expect(page.getByText('Dashboard users')).toHaveCount(0);
    },
  );
});
