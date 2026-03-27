import { defineConfig, devices } from '@playwright/test';

const projectRoot = __dirname;
const isCI = Boolean(process.env.CI);
const skipManagedWebServer = process.env.PLAYWRIGHT_SKIP_WEBSERVER === '1';
const reporter = process.env.PLAYWRIGHT_REPORTER ?? 'line';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: isCI,
  retries: isCI ? 2 : 0,
  workers: isCI ? 1 : undefined,
  reporter,
  use: {
    baseURL: process.env.DASHBOARD_URL ?? 'http://127.0.0.1:8501',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        ...(isCI ? {} : { executablePath: '/usr/bin/chromium' }),
        viewport: { width: 1440, height: 1600 },
      },
    },
  ],
  ...(skipManagedWebServer
    ? {}
    : {
        webServer: {
          command: [
            'env',
            `PYTHONPATH=${projectRoot}`,
            'CORKYSOFT_ENV=development',
            'CORKYSOFT_ALLOW_ANONYMOUS_UI=1',
            'venv/bin/python',
            '-m',
            'streamlit',
            'run',
            'dashboard/app.py',
            '--server.address',
            '127.0.0.1',
            '--server.port',
            '8501',
            '--server.headless',
            'true',
          ].join(' '),
          cwd: projectRoot,
          url: 'http://127.0.0.1:8501/',
          reuseExistingServer: !isCI,
          timeout: 120_000,
        },
      }),
});
