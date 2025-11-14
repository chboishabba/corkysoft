import { defineConfig, devices } from '@playwright/test';
import path from 'node:path';

/**
 * Read environment variables from file.
 * https://github.com/motdotla/dotenv
 */
// import dotenv from 'dotenv';
// dotenv.config({ path: path.resolve(__dirname, '.env') });

const projectRoot = __dirname;
const pythonPath = process.env.PYTHONPATH
  ? `${projectRoot}${path.delimiter}${process.env.PYTHONPATH}`
  : projectRoot;

const isCI = !!process.env.CI;

/**
 * See https://playwright.dev/docs/test-configuration.
 */
export default defineConfig({
  testDir: './e2e',
  /* Run tests in files in parallel */
  fullyParallel: true,
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: isCI,
  /* Retry on CI only */
  retries: isCI ? 2 : 0,
  /* Opt out of parallel tests on CI. */
  workers: isCI ? 1 : undefined,
  /* Reporter to use. */
  reporter: 'html',

  use: {
    baseURL: 'http://127.0.0.1:8501',
    trace: 'on-first-retry',
  },

  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],

        /*
         * Use system browser on Arch; Playwright's own in CI.
         */
        ...(isCI
        ? {}
        : {
          // 1) System Chromium (recommended, fewer crashes)
          executablePath: '/usr/bin/chromium',
          // 2) OR system Google Chrome Stable:
          // executablePath: '/usr/bin/google-chrome-stable',
        }),

        viewport: { width: 2160, height: 3840 },
      },
    },

    {
      name: 'firefox',
      use: {
        ...devices['Desktop Firefox'],

        ...(isCI ? {} : { executablePath: '/usr/bin/firefox' }),

                            viewport: { width: 2160, height: 3840 },
      },
    },

    // WebKit fails on Arch; enable only in CI.
    ...(isCI
    ? [
      {
        name: 'webkit',
        use: { ...devices['Desktop Safari'] },
      },
    ]
    : []),
  ],

  webServer: {
    // Use python -m so it works both locally and on CI
    command:
    'python -m streamlit run dashboard/app.py --server.port=8501 --server.headless=true',
    url: 'http://127.0.0.1:8501/',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },


  webServer: {
    // adjust path if your venv is elsewhere
    command:
            'python -m streamlit run dashboard/app.py --server.port=8501 --server.headless=true',
    url: 'http://127.0.0.1:8501/',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
