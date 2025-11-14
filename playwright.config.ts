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
         * Use **system browser** on Arch Linux.
         * Uncomment ONE of the following:
         */

        // 1) System Chromium (recommended, fewer crashes)
        executablePath: '/usr/bin/chromium',

        // 2) OR system Google Chrome Stable:
        // executablePath: '/usr/bin/google-chrome-stable',
      },
    },

    {
      name: 'firefox',
      use: {
        ...devices['Desktop Firefox'],
        executablePath: '/usr/bin/firefox',
      },
    },

    // WebKit fails on Arch (missing libicudata.so.66 etc.)
    // Enable only in CI environments where Playwright deps can be satisfied.
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
    // adjust path if your venv is elsewhere
    command:
      'venv/bin/streamlit run dashboard/app.py --server.port=8501 --server.headless=true',
    url: 'http://127.0.0.1:8501/',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
