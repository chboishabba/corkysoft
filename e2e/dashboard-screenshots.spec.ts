import { test } from '@playwright/test';
import path from 'node:path';
import fs from 'node:fs/promises';

type DashboardView = {
  label: string;
  filename: string;
  settleMs: number;
  waitForSelector?: string;
};

const DEFAULT_BASE_URL = 'http://127.0.0.1:8501/';
const OUTPUT_DIR =
  process.env.DASHBOARD_OUTPUT_DIR ?? path.join(__dirname, '..', 'docs', 'img');
const BASE_URL = process.env.DASHBOARD_URL ?? DEFAULT_BASE_URL;
const WAIT_FOR_SELECTOR_TIMEOUT_BUFFER = 5000;

const VIEWPORT = {
  width: Number(process.env.DASHBOARD_WIDTH ?? 1600),
  height: Number(process.env.DASHBOARD_HEIGHT ?? 900),
};

const VIEWS: DashboardView[] = [
  {
    label: 'Histogram',
    filename: 'dashboard-histogram.png',
    settleMs: 4000,
  },
  {
    label: 'Price history',
    filename: 'dashboard-price-history.png',
    settleMs: 5000,
  },
  {
    label: 'Profitability insights',
    filename: 'dashboard-profitability.png',
    settleMs: 5000,
  },
  {
    label: 'Live network overview',
    filename: 'dashboard-live-network.png',
    settleMs: 12000,
    waitForSelector: '[data-testid="stDeckGlJsonChart"] canvas',
  },
  {
    label: 'Route maps',
    filename: 'dashboard-route-maps.png',
    settleMs: 12000,
    waitForSelector: '[data-testid="stDeckGlJsonChart"] canvas',
  },
  {
    label: 'Quote builder',
    filename: 'dashboard-quote-builder.png',
    settleMs: 5000,
  },
  {
    label: 'Optimizer',
    filename: 'dashboard-optimizer.png',
    settleMs: 5000,
  },
];

async function ensureOutputDir(): Promise<void> {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
}

function buildViewUrl(view: string): string {
  const base = BASE_URL.endsWith('/') ? BASE_URL : `${BASE_URL}/`;
  const url = new URL(base);
  url.searchParams.set('view', view);
  return url.toString();
}

test.describe('dashboard screenshots', () => {
  test.describe.configure({ timeout: 60_000 });

  test.beforeAll(async () => {
    await ensureOutputDir();
  });

  for (const { label, filename, settleMs, waitForSelector } of VIEWS) {
    test(`${label} view`, async ({ page }, testInfo) => {
      test.skip(
        testInfo.project.name !== 'chromium',
        'Screenshots are only generated via the Chromium project.'
      );

      await page.setViewportSize(VIEWPORT);
      await page.goto(buildViewUrl(label), { waitUntil: 'networkidle' });

      if (waitForSelector) {
        try {
          await page.waitForSelector(waitForSelector, {
            timeout: settleMs + WAIT_FOR_SELECTOR_TIMEOUT_BUFFER,
          });
        } catch (error) {
          testInfo.annotations.push({
            type: 'warning',
            description: `Timed out waiting for selector ${waitForSelector}: ${String(
              error,
            )}`,
          });
        }
      }

      await page.waitForTimeout(settleMs);

      const outputPath = path.join(OUTPUT_DIR, filename);
      await page.screenshot({
        path: outputPath,
        fullPage: false,
      });
    });
  }
});
