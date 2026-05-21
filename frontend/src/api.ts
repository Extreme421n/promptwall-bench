/**
 * Tiny typed API client for the DemoCorp backend.
 *
 * The base URL comes from `VITE_API_BASE_URL` at build time. We fall back to
 * `http://localhost:8000` so a developer who hasn't created `.env.local` can
 * still run the frontend against a default uvicorn instance.
 *
 * NOTE: this client wraps the *DemoCorp* simulation endpoints only. The
 * ``mode`` parameter mirrors the backend's union type so the schema stays
 * honest, but the UI surfaces ``baseline`` only — no PromptWall behaviour
 * is invoked from the frontend.
 */

const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://localhost:8000";

// ---------------------------------------------------------------------------
// Types — mirror the FastAPI response models in app/main.py.
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: string;
}

export interface ToolSummary {
  name: string;
  description: string;
  domain: string;
  risk_level: string;
  read_only: boolean;
}

export interface ListToolsResponse {
  count: number;
  tools: ToolSummary[];
}

/** GET /tools/{name} — same fields as ToolSummary plus JSON-Schema bodies. */
export interface ToolDescription extends ToolSummary {
  input_schema: Record<string, unknown>;
  output_schema: Record<string, unknown>;
}

export interface ChatToolCallSummary {
  name: string;
  arguments: Record<string, unknown>;
  success: boolean;
  evidence_id: string | null;
  error_type: string | null;
  error_message: string | null;
  latency_ms: number;
}

export interface ChatResponse {
  answer: string;
  trace_id: number;
  session_id: string;
  tools_called: ChatToolCallSummary[];
  evidence_ids: string[];
  latency_ms: number;
  estimated_cost_usd: string;
}

/** ChatRequest.mode literal type — mirrors the backend enum. */
export type ChatMode =
  | "baseline"
  | "promptwall_candidate_shadow"
  | "promptwall_enforced";

export interface SendChatMessageOptions {
  message: string;
  sessionId?: string;
  customerId?: number;
  /** Defaults to "baseline". The UI only offers baseline in Phase 2. */
  mode?: ChatMode;
  /** LLM model name. Defaults to "mock". */
  model?: string;
  /** Free-form metadata stamped onto the trace. Merged with frontend defaults. */
  metadata?: Record<string, unknown>;
  /** Convenience — placed under `metadata.use_case`. */
  useCase?: string;
}

// ---------------------------------------------------------------------------
// HTTP helper
// ---------------------------------------------------------------------------

class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, body: unknown, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  const text = await res.text();
  let parsed: unknown = undefined;
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }
  }
  if (!res.ok) {
    const detail =
      parsed &&
      typeof parsed === "object" &&
      "detail" in (parsed as Record<string, unknown>)
        ? String((parsed as Record<string, unknown>).detail)
        : res.statusText;
    throw new ApiError(res.status, parsed, `${path} → ${res.status} ${detail}`);
  }
  return parsed as T;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}

/** Hit GET /health. Resolves to `{ status: "ok" }` when the backend is up. */
export async function getHealth(): Promise<HealthResponse> {
  return http<HealthResponse>("/health");
}

/** List every tool registered in the backend tool registry. */
export async function getTools(): Promise<ListToolsResponse> {
  return http<ListToolsResponse>("/tools");
}

/** Fetch full details for a single tool (input + output JSON-Schema). */
export async function getToolDetail(name: string): Promise<ToolDescription> {
  return http<ToolDescription>(`/tools/${encodeURIComponent(name)}`);
}

/**
 * Send a user message to the DemoCorp chatbot.
 *
 * Defaults: ``mode="baseline"``, ``model="mock"``, ``metadata.use_case=
 * "demo_frontend"``. The UI only exposes the baseline mode — no PromptWall
 * routing/enforcement is invoked here. Callers can override any field.
 */
export async function sendChatMessage(
  opts: SendChatMessageOptions
): Promise<ChatResponse> {
  const payload: Record<string, unknown> = {
    mode: opts.mode ?? "baseline",
    model: opts.model ?? "mock",
    message: opts.message,
    session_id: opts.sessionId,
    customer_id: opts.customerId,
    metadata: {
      use_case: opts.useCase ?? "demo_frontend",
      channel: "web",
      ...(opts.metadata ?? {}),
    },
  };
  return http<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export { ApiError };
