import { expect, test, type Page } from '@playwright/test';
import path from 'node:path';
import fs from 'node:fs/promises';

type DashboardView = {
  label: string;
  shellView: 'Quote' | 'Pricing Intelligence' | 'Network' | 'Operations' | 'Admin';
  filename: string;
  settleMs: number;
  expand?: string;
  tab?: string;
  radio?: string;
  waitForSelector?: string;
};

const DEFAULT_BASE_URL = 'http://127.0.0.1:8501/';
const OUTPUT_DIR =
  process.env.DASHBOARD_OUTPUT_DIR ?? path.join(__dirname, '..', 'docs', 'img');
const BASE_URL = process.env.DASHBOARD_URL ?? DEFAULT_BASE_URL;
const WAIT_FOR_SELECTOR_TIMEOUT_BUFFER = 5000;

const VIEWPORT = {
  width: Number(process.env.DASHBOARD_WIDTH ?? 2160),
  height: Number(process.env.DASHBOARD_HEIGHT ?? 3840),
};

/**
 * README screenshot contract.
 *
 * Keep shell routing and nested-surface navigation in one manifest so screenshots
 * cannot silently drift when dashboard tabs are regrouped. Each capture starts from
 * a valid top-level `view` and then selects the atomic surface it documents.
 */
const VIEWS: DashboardView[] = [
  {
    label: 'Histogram',
    shellView: 'Pricing Intelligence',
    expand: 'Historic Analytics & Distribution',
    tab: 'Histogram',
    filename: 'dashboard-histogram.png',
    settleMs: 4000,
  },
  {
    label: 'Price history',
    shellView: 'Pricing Intelligence',
    expand: 'Historic Analytics & Distribution',
    tab: 'Price history',
    filename: 'dashboard-price-history.png',
    settleMs: 5000,
  },
  {
    label: 'Profitability insights',
    shellView: 'Pricing Intelligence',
    expand: 'Historic Analytics & Distribution',
    tab: 'Profitability insights',
    filename: 'dashboard-profitability.png',
    settleMs: 5000,
  },
  {
    label: 'Live network overview',
    shellView: 'Network',
    radio: 'Live network',
    filename: 'dashboard-live-network.png',
    settleMs: 12000,
    waitForSelector: '[data-testid="stDeckGlJsonChart"] canvas',
  },
  {
    label: 'Route maps',
    shellView: 'Network',
    radio: 'Routes/points',
    filename: 'dashboard-route-maps.png',
    settleMs: 12000,
    waitForSelector: '[data-testid="stDeckGlJsonChart"] canvas',
  },
  {
    label: 'Quote builder',
    shellView: 'Quote',
    filename: 'dashboard-quote-builder.png',
    settleMs: 5000,
  },
  {
    label: 'Optimizer',
    shellView: 'Pricing Intelligence',
    filename: 'dashboard-optimizer.png',
    settleMs: 5000,
  },
];

async function ensureOutputDir(): Promise<void> {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
}

function buildViewUrl(view: DashboardView['shellView']): string {
  const base = BASE_URL.endsWith('/') ? BASE_URL : `${BASE_URL}/`;
  const url = new URL(base);
  url.searchParams.set('view', view);
  return url.toString();
}

async function openSurface(page: Page, view: DashboardView): Promise<void> {
  await page.goto(buildViewUrl(view.shellView), { waitUntil: 'networkidle' });

  if (view.expand) {
    const expander = page.getByText(view.expand, { exact: false }).first();
    await expect(expander).toBeVisible();
    await expander.click();
  }

  if (view.tab) {
    const tab = page.getByRole('tab', { name: view.tab, exact: true });
    await expect(tab).toBeVisible();
    await tab.click();
  }

  if (view.radio) {
    const radio = page.getByText(view.radio, { exact: true }).first();
    await expect(radio).toBeVisible();
    await radio.click();
  }
}

test.describe('dashboard screenshots', () => {
  test.describe.configure({ timeout: 90_000 });

  test.beforeAll(async () => {
    await ensureOutputDir();
  });

  for (const view of VIEWS) {
    test(`${view.label} view`, async ({ page }, testInfo) => {
      test.skip(
        testInfo.project.name !== 'chromium',
        'Screenshots are only generated via the Chromium project.'
      );

      await page.setViewportSize(VIEWPORT);
      await openSurface(page, view);

      if (view.waitForSelector) {
        try {
          await page.waitForSelector(view.waitForSelector, {
            timeout: view.settleMs + WAIT_FOR_SELECTOR_TIMEOUT_BUFFER,
          });
        } catch (error) {
          testInfo.annotations.push({
            type: 'warning',
            description: `Timed out waiting for selector ${view.waitForSelector}: ${String(
              error,
            )}`,
          });
        }
      }

      await page.waitForTimeout(view.settleMs);

      const outputPath = path.join(OUTPUT_DIR, view.filename);
      await page.screenshot({
        path: outputPath,
        fullPage: false,
      });
    });
  }
});
