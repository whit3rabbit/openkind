/** Primitive, array, or object JSON-serializable value. */
export type JSONValue = null | boolean | number | string | JSONValue[] | { [key: string]: JSONValue };

/** State payload for an evaluation request, represented as text, an array, or an object. */
export type State = string | JSONValue[] | { [key: string]: JSONValue };

/** Instruction input for an evaluation question. */
export type Instructions = boolean | number | string | JSONValue[] | { [key: string]: JSONValue };

/** Discriminated union of evaluation questions: binary noul, multiple-choice, or rubric score. */
export type Question =
  | { type: "noul"; instructions: Instructions; criteria?: { true: string; false: string } }
  | { type: "choice"; instructions: Instructions; criteria: Record<string, string | null> }
  | { type: "score"; instructions: Instructions; criteria: string[] };

/** Top-level System One evaluation request containing state context, model alias, and questions. */
export interface SystemRequest {
  state: State;
  model: string;
  questions: Record<string, Question>;
}

/** Discriminated union of evaluated answers matching the requested question types. */
export type Answer =
  | { type: "noul"; noul: number }
  | { type: "choice"; choice: string; probabilities: Record<string, number>; confidence: number }
  | { type: "score"; score: number; legend: Record<string, string>; probabilities: Record<string, number>; confidence: number };

/** System One evaluation response containing the model name, evaluated answers, and token usage. */
export interface SystemResponse {
  model: string;
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: number };
}

/** Metadata describing an available model profile in the registry or catalog. */
export interface ModelMetadata { name: string; description: string; release_date: string }

/** Response payload from the `/v1/models` endpoint listing available models. */
export interface ModelsResponse { models: ModelMetadata[] }

/** Daemon health check status response. */
export interface Health { status: string }

/** Result container wrapping the decoded API response data and optional request ID. */
export interface ApiResult<T> { data: T; requestId: string | null }

/** Error thrown when an HTTP call to the daemon fails with a non-2xx status code. */
export class ApiError extends Error {
  /**
   * Create an ApiError with the HTTP status, error code, message, and request ID.
   */
  constructor(
    readonly status: number,
    readonly code: string | null,
    message: string,
    readonly requestId: string | null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Error thrown when a response payload violates the wire contract or response schema. */
export class InvalidResponseError extends Error {
  /** Create an InvalidResponseError with the failure explanation. */
  constructor(message: string) {
    super(message);
    this.name = "InvalidResponseError";
  }
}

/** Configuration options for instantiating an OpenKindClient. */
export interface ClientOptions {
  /** Base URL of the daemon (default: `http://127.0.0.1:18080`). */
  baseUrl?: string;
  /** Optional bearer API key for authenticated requests. */
  apiKey?: string;
  /** Default model alias to evaluate against when omitted (default: `jev-latest`). */
  defaultModel?: string;
  /** HTTP request timeout in milliseconds (default: 10,000). */
  timeoutMs?: number;
  /** Custom fetch implementation (defaults to global fetch). */
  fetch?: typeof fetch;
}

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function fail(message: string): never {
  throw new InvalidResponseError(message);
}

function probability(value: unknown, label: string): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    fail(`${label} must be a probability`);
  }
}

function distribution(value: unknown, label: string): Record<string, number> {
  if (!object(value)) fail(`${label} must be an object`);
  let sum = 0;
  for (const [key, item] of Object.entries(value)) {
    probability(item, `${label}.${key}`);
    sum += item as number;
  }
  if (Math.abs(sum - 1) > 1e-3) fail(`${label} must sum to one`);
  return value as Record<string, number>;
}

function sameKeys(a: Record<string, unknown>, b: Record<string, unknown>): boolean {
  const left = Object.keys(a);
  return left.length === Object.keys(b).length && left.every((key) => Object.hasOwn(b, key));
}

/**
 * Validate that an evaluation response satisfies the wire contract for the specified request.
 * Throws {@link InvalidResponseError} if the response structure, IDs, distributions, or bounds are invalid.
 */
export function validateResponse(value: unknown, request: SystemRequest): asserts value is SystemResponse {
  if (!object(value) || typeof value.model !== "string" || !object(value.answers) || !object(value.usage)) {
    fail("invalid evaluation response");
  }
  const response = value as Record<string, unknown>;
  const answers = response.answers as Record<string, unknown>;
  const usage = response.usage as Record<string, unknown>;
  if (!sameKeys(answers, request.questions)) fail("answer IDs do not match question IDs");
  for (const field of ["input_tokens", "output_tokens"]) {
    const count = usage[field];
    if (typeof count !== "number" || !Number.isSafeInteger(count) || count < 0 || count > 0xFFFFFFFF) {
      fail(`invalid usage.${field}`);
    }
  }
  for (const [id, question] of Object.entries(request.questions)) {
    const answer = answers[id];
    if (!object(answer) || answer.type !== question.type) fail(`answer type mismatch for ${id}`);
    if (question.type === "noul") {
      probability(answer.noul, `${id}.noul`);
    } else if (question.type === "choice") {
      if (typeof answer.choice !== "string" || !Object.hasOwn(question.criteria, answer.choice)) {
        fail(`choice is outside criteria for ${id}`);
      }
      const probabilities = distribution(answer.probabilities, `${id}.probabilities`);
      if (!sameKeys(probabilities, question.criteria)) fail(`probability keys mismatch for ${id}`);
      probability(answer.confidence, `${id}.confidence`);
    } else {
      if (!object(answer.legend)) fail(`invalid score legend for ${id}`);
      const probabilities = distribution(answer.probabilities, `${id}.probabilities`);
      if (!sameKeys(probabilities, answer.legend)) fail(`score legend mismatch for ${id}`);
      for (const [key, legend] of Object.entries(answer.legend)) {
        if (!/^[0-9]+$/.test(key) || Number(key) > 0xFFFFFFFF || typeof legend !== "string") {
          fail(`invalid score legend for ${id}`);
        }
      }
      const max = Math.max(0, ...Object.keys(probabilities).map(Number));
      if (typeof answer.score !== "number" || !Number.isFinite(answer.score) || answer.score < 0 || answer.score > max) {
        fail(`score out of range for ${id}`);
      }
      probability(answer.confidence, `${id}.confidence`);
    }
  }
}

/** HTTP client for the OpenKind / System One decision inference API. */
export class OpenKindClient {
  /** Normalized base URL of the target daemon without trailing slashes. */
  readonly baseUrl: string;
  /** Default model profile alias used when omitted from evaluation calls. */
  readonly defaultModel: string;
  private readonly apiKey?: string;
  private readonly timeoutMs: number;
  private readonly fetcher: typeof fetch;

  /** Create an OpenKindClient with configuration options. */
  constructor(options: ClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? "http://127.0.0.1:18080").replace(/\/+$/, "");
    this.defaultModel = options.defaultModel ?? "jev-latest";
    this.apiKey = options.apiKey;
    this.timeoutMs = options.timeoutMs ?? 10_000;
    this.fetcher = options.fetch ?? globalThis.fetch.bind(globalThis);
    if (!/^https?:\/\/[^/]+/.test(this.baseUrl)) throw new TypeError("baseUrl must be an HTTP URL");
    if (!Number.isFinite(this.timeoutMs) || this.timeoutMs <= 0) throw new TypeError("timeoutMs must be positive");
  }

  private async send<T>(path: string, method: "GET" | "POST", body?: unknown): Promise<ApiResult<T>> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    const headers: Record<string, string> = { Accept: "application/json" };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    if (this.apiKey && path !== "/health") headers.Authorization = `Bearer ${this.apiKey}`;
    try {
      const response = await this.fetcher(`${this.baseUrl}${path}`, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body, (_key, value: unknown) => {
          if (typeof value === "number" && !Number.isFinite(value)) throw new TypeError("request contains a nonfinite number");
          return value;
        }),
        signal: controller.signal,
      });
      const requestId = response.headers.get("x-typesafe-request-id");
      const raw = await response.text();
      let data: unknown;
      try { data = JSON.parse(raw); } catch {
        if (!response.ok) throw new ApiError(response.status, null, `HTTP ${response.status}`, requestId);
        throw new InvalidResponseError("response is not JSON");
      }
      if (!response.ok) {
        const details = object(data) && object(data.error) ? data.error : {};
        throw new ApiError(response.status, typeof details.code === "string" ? details.code : null,
          typeof details.message === "string" ? details.message : `HTTP ${response.status}`, requestId);
      }
      return { data: data as T, requestId };
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Submit an evaluation request to `/v1/systemone` and validate the returned answers.
   *
   * @param request - Full System One evaluation request payload.
   * @returns Validated response with answers and token usage.
   */
  async evaluate(request: SystemRequest): Promise<ApiResult<SystemResponse>> {
    const result = await this.send<SystemResponse>("/v1/systemone", "POST", request);
    validateResponse(result.data, request);
    return result;
  }

  /**
   * Convenience helper to evaluate questions against a given state.
   *
   * @param state - Context state string, array, or object.
   * @param questions - Map of question IDs to question definitions.
   * @param model - Optional model override (defaults to `defaultModel`).
   */
  systemOne(state: State, questions: Record<string, Question>, model = this.defaultModel): Promise<ApiResult<SystemResponse>> {
    return this.evaluate({ state, model, questions });
  }

  /** Retrieve the list of available model profiles from `/v1/models`. */
  async listModels(): Promise<ApiResult<ModelsResponse>> {
    const result = await this.send<ModelsResponse>("/v1/models", "GET");
    if (!object(result.data) || !Array.isArray(result.data.models) ||
      !result.data.models.every((item: unknown) => object(item) && typeof item.name === "string" &&
        typeof item.description === "string" && typeof item.release_date === "string")) {
      fail("invalid models response");
    }
    return result;
  }

  /** Query the `/health` endpoint to check daemon availability. */
  async health(): Promise<ApiResult<Health>> {
    const result = await this.send<Health>("/health", "GET");
    if (!object(result.data) || typeof result.data.status !== "string") fail("invalid health response");
    return result;
  }
}
