import { spawn, type ChildProcess } from "node:child_process";
import { createServer } from "node:net";
import { setTimeout as delay } from "node:timers/promises";

import { OpenKindClient } from "./index.js";

export interface ServerOptions {
  binary?: string;
  httpAddr?: string;
  models?: string[];
  apiKey?: string;
  startupTimeoutMs?: number;
  shutdownTimeoutMs?: number;
  extraArgs?: string[];
}

export class ServerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ServerError";
  }
}

const reserved = ["--http-addr", "--grpc-addr", "--models", "--api-key"];

function validateArgs(args: string[]): void {
  if (args.some((arg) => reserved.some((flag) => arg === flag || arg.startsWith(`${flag}=`)))) {
    throw new TypeError("extraArgs cannot override managed server flags");
  }
}

async function portIsFree(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const probe = createServer();
    probe.once("error", () => resolve(false));
    probe.listen(port, "127.0.0.1", () => probe.close(() => resolve(true)));
  });
}

/** Owns one local openkindd child process. It never stops a server it did not start. */
export class OpenKindServer {
  readonly binary: string;
  readonly httpAddr: string;
  readonly baseUrl: string;
  readonly models: readonly string[];
  readonly apiKey?: string;
  private readonly startupTimeoutMs: number;
  private readonly shutdownTimeoutMs: number;
  private readonly extraArgs: readonly string[];
  private child: ChildProcess | null = null;
  private startup: AbortController | null = null;
  private stopping: Promise<void> | null = null;

  constructor(options: ServerOptions = {}) {
    this.binary = options.binary ?? "openkindd";
    this.httpAddr = options.httpAddr ?? "127.0.0.1:18080";
    const match = /^127\.0\.0\.1:(\d+)$/.exec(this.httpAddr);
    const port = match === null ? NaN : Number(match[1]);
    if (!Number.isInteger(port) || port < 1 || port > 65535) {
      throw new TypeError("httpAddr must use 127.0.0.1 and a nonzero port");
    }
    this.baseUrl = `http://${this.httpAddr}`;
    this.models = [...(options.models ?? ["mock", "jev-latest"])];
    if (this.models.length === 0 || this.models.some((model) => !model || model.includes(","))) {
      throw new TypeError("models must contain nonempty aliases without commas");
    }
    this.apiKey = options.apiKey;
    this.startupTimeoutMs = options.startupTimeoutMs ?? 10_000;
    this.shutdownTimeoutMs = options.shutdownTimeoutMs ?? 5_000;
    if (!Number.isFinite(this.startupTimeoutMs) || this.startupTimeoutMs <= 0 ||
        !Number.isFinite(this.shutdownTimeoutMs) || this.shutdownTimeoutMs <= 0) {
      throw new TypeError("timeouts must be positive");
    }
    this.extraArgs = [...(options.extraArgs ?? [])];
    validateArgs([...this.extraArgs]);
  }

  get running(): boolean {
    return this.child !== null && this.child.pid !== undefined &&
      this.child.exitCode === null && this.child.signalCode === null;
  }

  client(): OpenKindClient {
    return new OpenKindClient({ baseUrl: this.baseUrl, apiKey: this.apiKey });
  }

  async start(): Promise<this> {
    // Reserve ownership before the asynchronous port probe so two starts cannot spawn competing children.
    if (this.child !== null || this.startup !== null || this.stopping !== null) {
      throw new ServerError("server has already been started");
    }
    const startup = new AbortController();
    this.startup = startup;
    let child: ChildProcess | null = null;
    try {
      const port = Number(this.httpAddr.slice(this.httpAddr.lastIndexOf(":") + 1));
      if (!await portIsFree(port)) throw new ServerError(`HTTP address is unavailable: ${this.httpAddr}`);
      if (startup.signal.aborted) throw new ServerError("openkindd startup was cancelled");
      const env = { ...process.env };
      delete env.OPENKIND_API_KEY;
      delete env.OPENDECISION_API_KEY;
      delete env.OPENPICK_API_KEY;
      delete env.TYPESAFE_API_KEY;
      if (this.apiKey !== undefined) env.OPENKIND_API_KEY = this.apiKey;
      child = spawn(this.binary, [...this.extraArgs, "--http-addr", this.httpAddr,
        "--grpc-addr", "0", "--models", this.models.join(",")],
      { env, stdio: ["ignore", "ignore", "inherit"] });
      this.child = child;
      const spawnState: { error?: Error } = {};
      child.once("error", (error) => { spawnState.error = error; });
      const deadline = Date.now() + this.startupTimeoutMs;
      while (Date.now() < deadline) {
        if (startup.signal.aborted) throw new ServerError("openkindd startup was cancelled");
        if (spawnState.error !== undefined) {
          throw new ServerError(`could not start openkindd: ${spawnState.error.message}`);
        }
        if (child.exitCode !== null || child.signalCode !== null) {
          throw new ServerError(`openkindd exited before readiness with code ${child.exitCode}`);
        }
        try {
          const response = await fetch(`${this.baseUrl}/health`, {
            signal: AbortSignal.timeout(Math.max(1, Math.min(250, deadline - Date.now()))),
            redirect: "error",
          });
          if (response.ok && (await response.json() as { status?: string }).status === "ok" &&
              !startup.signal.aborted && this.running) return this;
          await response.body?.cancel();
        } catch { /* The HTTP listener may still be starting. */ }
        await delay(50);
      }
      throw new ServerError("openkindd did not become healthy before the startup timeout");
    } catch (error) {
      if (child !== null) await this.stopChild(child);
      throw error;
    } finally {
      if (this.startup === startup) this.startup = null;
    }
  }

  async stop(): Promise<void> {
    this.startup?.abort();
    const child = this.child;
    if (child === null) return;
    await this.stopChild(child);
  }

  private async stopChild(child: ChildProcess): Promise<void> {
    if (this.stopping !== null) return this.stopping;
    this.stopping = (async () => {
      if (child.exitCode !== null || child.signalCode !== null || child.pid === undefined) return;
      const closed = new Promise<void>((resolve) => child.once("close", () => resolve()));
      child.kill("SIGTERM");
      let timer: ReturnType<typeof setTimeout> | undefined;
      const timedOut = new Promise<void>((resolve) => {
        timer = setTimeout(resolve, this.shutdownTimeoutMs);
      });
      await Promise.race([closed, timedOut]);
      if (timer !== undefined) clearTimeout(timer);
      if (child.exitCode === null && child.signalCode === null) {
        child.kill("SIGKILL");
        await closed;
      }
    })();
    try {
      await this.stopping;
    } finally {
      if (this.child === child) this.child = null;
      this.stopping = null;
    }
  }
}
