import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawn, type ChildProcess } from 'node:child_process';

type AuthScenarioEnv = {
  [key: string]: string | undefined;
};

export type StreamlitHarness = {
  baseUrl: string;
  cleanup: () => Promise<void>;
};

async function waitForServer(url: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let lastError: unknown = null;

  while (Date.now() < deadline) {
    try {
      const response = await fetch(url, { redirect: 'manual' });
      if (response.ok || response.status === 304) {
        return;
      }
      lastError = new Error(`Unexpected status ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }

  throw new Error(`Timed out waiting for ${url}: ${String(lastError)}`);
}

async function stopProcess(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) {
    return;
  }

  await new Promise<void>((resolve) => {
    const timeout = setTimeout(() => {
      if (child.exitCode === null && child.signalCode === null) {
        child.kill('SIGKILL');
      }
    }, 5_000);

    child.once('exit', () => {
      clearTimeout(timeout);
      resolve();
    });

    child.kill('SIGTERM');
  });
}

export async function startStreamlitHarness(
  port: number,
  extraEnv: AuthScenarioEnv,
): Promise<StreamlitHarness> {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'corkysoft-playwright-auth-'));
  const dbPath = path.join(tmpDir, 'auth-harness.db');
  const baseUrl = `http://127.0.0.1:${port}/`;
  const child = spawn(
    process.env.PYTHON ?? path.join(process.cwd(), 'venv', 'bin', 'python'),
    [
      '-m',
      'streamlit',
      'run',
      'dashboard/app.py',
      '--server.address',
      '127.0.0.1',
      '--server.port',
      String(port),
      '--server.headless',
      'true',
      '--browser.gatherUsageStats',
      'false',
    ],
    {
      cwd: process.cwd(),
      env: {
        ...process.env,
        PYTHONPATH: process.cwd(),
        CORKYSOFT_DB: dbPath,
        CORKYSOFT_ENV: 'development',
        STREAMLIT_BROWSER_GATHER_USAGE_STATS: 'false',
        ...extraEnv,
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  );

  let startupOutput = '';
  child.stdout?.on('data', (chunk: Buffer | string) => {
    startupOutput += chunk.toString();
  });
  child.stderr?.on('data', (chunk: Buffer | string) => {
    startupOutput += chunk.toString();
  });

  try {
    await waitForServer(`${baseUrl}_stcore/health`, 120_000);
  } catch (error) {
    await stopProcess(child);
    throw new Error(`Failed to start Streamlit auth harness: ${String(error)}\n${startupOutput}`);
  }

  return {
    baseUrl,
    cleanup: async () => {
      await stopProcess(child);
      await fs.rm(tmpDir, { recursive: true, force: true });
    },
  };
}
