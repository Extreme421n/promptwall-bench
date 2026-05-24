import { useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  ChatMode,
  ChatResponse,
  getApiBaseUrl,
  getHealth,
  sendChatMessage,
} from "./api";
import ToolsPanel from "./ToolsPanel";

type View = "chat" | "tools";

// ---------------------------------------------------------------------------
// State model
// ---------------------------------------------------------------------------

type HealthState = "checking" | "ok" | "down";

interface BaseMessage {
  id: string;
  ts: number; // epoch ms
}

interface UserMessage extends BaseMessage {
  role: "user";
  text: string;
}

interface AssistantMessage extends BaseMessage {
  role: "assistant";
  /** Set once the backend has replied. */
  response?: ChatResponse;
  /** Echoes the mode/model used when the user sent this turn. */
  requestedMode: ChatMode;
  requestedModel: string;
  /** When the request failed, the assistant bubble carries the error. */
  error?: string;
  /** While the request is in flight. */
  pending: boolean;
}

type Message = UserMessage | AssistantMessage;

interface ChatSettings {
  mode: ChatMode;
  model: string;
  customerId: string; // string so the input can be empty
  useCase: string;
}

const DEFAULT_SETTINGS: ChatSettings = {
  mode: "baseline",
  model: "gpt-4o-mini",
  customerId: "",
  useCase: "demo_frontend",
};

const USE_CASE_OPTIONS = [
  "demo_frontend",
  "manual_qa",
  "policy_review",
  "incident_triage",
];

// Ten curated demo prompts. They exercise the seven Phase-6B-4 text-retrieval
// tools and the original DemoCorp tool set. The list is fixed so a presenter
// can rehearse against the same questions every time.
const DEMO_PROMPTS: string[] = [
  "Can I return an opened electronic product?",
  "What is our cancellation policy?",
  "What happens if my flight is delayed more than 3 hours?",
  "Why was I charged overage?",
  "What warranty exclusions apply to SKU-000001?",
  "Where is my order?",
  "Is my invoice paid?",
  "What is the status of my support ticket?",
  "Can I get a refund?",
  "How many seats do I have left?",
];

function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  const [view, setView] = useState<View>("chat");
  const [health, setHealth] = useState<HealthState>("checking");
  const [healthError, setHealthError] = useState<string | null>(null);
  const [settings, setSettings] = useState<ChatSettings>(DEFAULT_SETTINGS);
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const sessionIdRef = useRef<string | undefined>(undefined);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Initial healthcheck + 30s heartbeat.
  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const h = await getHealth();
        if (cancelled) return;
        if (h.status === "ok") {
          setHealth("ok");
          setHealthError(null);
        } else {
          setHealth("down");
          setHealthError(`unexpected status: ${h.status}`);
        }
      } catch (e) {
        if (cancelled) return;
        setHealth("down");
        setHealthError(e instanceof Error ? e.message : String(e));
      }
    }
    void check();
    const t = window.setInterval(check, 30_000);
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, []);

  // Auto-scroll to bottom on new message.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages]);

  const customerIdParsed = useMemo(() => {
    const trimmed = settings.customerId.trim();
    if (!trimmed) return undefined;
    const n = Number(trimmed);
    return Number.isFinite(n) && n > 0 ? n : undefined;
  }, [settings.customerId]);

  const customerIdInvalid =
    settings.customerId.trim().length > 0 && customerIdParsed === undefined;

  const backendDown = health === "down";

  async function onSend(override?: string) {
    // Demo-prompt clicks pass the prompt text directly; the textarea path
    // uses whatever the user has typed.
    const text = (override ?? draft).trim();
    if (!text || sending || backendDown) return;

    const now = Date.now();
    const userMsg: UserMessage = {
      id: newId(),
      ts: now,
      role: "user",
      text,
    };
    const assistantId = newId();
    const placeholder: AssistantMessage = {
      id: assistantId,
      ts: now,
      role: "assistant",
      requestedMode: settings.mode,
      requestedModel: settings.model,
      pending: true,
    };

    setMessages((prev) => [...prev, userMsg, placeholder]);
    // Only clear the textarea when the text came from it — clicking a demo
    // prompt should leave whatever the user was already drafting alone.
    if (override === undefined) {
      setDraft("");
    }
    setSending(true);

    try {
      const r = await sendChatMessage({
        message: text,
        sessionId: sessionIdRef.current,
        customerId: customerIdParsed,
        mode: settings.mode,
        model: settings.model.trim() || "mock",
        useCase: settings.useCase.trim() || "demo_frontend",
      });
      sessionIdRef.current = r.session_id;
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId && m.role === "assistant"
            ? { ...m, pending: false, response: r }
            : m
        )
      );
    } catch (e) {
      const detail =
        e instanceof ApiError
          ? `${e.status} · ${e.message}`
          : e instanceof Error
          ? e.message
          : String(e);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId && m.role === "assistant"
            ? { ...m, pending: false, error: detail }
            : m
        )
      );
    } finally {
      setSending(false);
    }
  }

  function onResetChat() {
    setMessages([]);
    setDraft("");
    sessionIdRef.current = undefined;
  }

  function onKeyDown(ev: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) {
      ev.preventDefault();
      void onSend();
    }
  }

  const sendDisabled =
    sending || draft.trim().length === 0 || backendDown || customerIdInvalid;

  return (
    <div className="app-shell">
      <Sidebar
        settings={settings}
        onChange={setSettings}
        onReset={onResetChat}
        customerIdInvalid={customerIdInvalid}
        backendDown={backendDown}
        messageCount={messages.length}
        sessionId={sessionIdRef.current}
      />

      <div className="chat-area">
        <header className="header">
          <div className="header-row">
            <h1>DemoCorp AI Assistant</h1>
            <nav className="view-tabs" aria-label="View">
              <button
                type="button"
                className={`tab ${view === "chat" ? "tab-active" : ""}`}
                aria-pressed={view === "chat"}
                onClick={() => setView("chat")}
              >
                Chat
              </button>
              <button
                type="button"
                className={`tab ${view === "tools" ? "tab-active" : ""}`}
                aria-pressed={view === "tools"}
                onClick={() => setView("tools")}
              >
                Tools
              </button>
            </nav>
          </div>
          <div className="meta">
            <HealthBadge state={health} message={healthError} />
            <span className="api-base">api: {getApiBaseUrl()}</span>
          </div>
        </header>

        {backendDown && (
          <div className="banner banner-error" role="alert">
            <strong>Backend unavailable.</strong>{" "}
            {healthError ?? "The frontend can't reach the DemoCorp API."}
            <div className="banner-help">
              Start the backend with{" "}
              <code>uvicorn app.main:app --port 8000</code> or update{" "}
              <code>VITE_API_BASE_URL</code>.
            </div>
          </div>
        )}

        {view === "chat" ? (
          <main className="chat-main">
            <div ref={scrollRef} className="chat-history" aria-live="polite">
              {messages.length === 0 ? (
                <EmptyState />
              ) : (
                messages.map((m) =>
                  m.role === "user" ? (
                    <UserBubble key={m.id} message={m} />
                  ) : (
                    <AssistantBubble key={m.id} message={m} />
                  )
                )
              )}
            </div>

            <DemoPromptsPanel
              prompts={DEMO_PROMPTS}
              disabled={sending || backendDown || customerIdInvalid}
              defaultOpen={messages.length === 0}
              onPick={(p) => {
                void onSend(p);
              }}
              onPickToInput={(p) => setDraft(p)}
            />

            <section className="composer">
              <label htmlFor="message" className="visually-hidden">
                Your message
              </label>
              <textarea
                id="message"
                className="message-input"
                placeholder={
                  backendDown
                    ? "Backend offline — fix the connection to start chatting."
                    : "Ask DemoCorp anything — e.g. “What is our cancellation policy?”"
                }
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={onKeyDown}
                rows={3}
                disabled={sending || backendDown}
              />
              <div className="composer-row">
                <button
                  type="button"
                  className="send-button"
                  onClick={() => void onSend()}
                  disabled={sendDisabled}
                >
                  {sending ? "Sending…" : "Send"}
                </button>
                <span className="hint">
                  ⌘/Ctrl + Enter to send · mode <code>{settings.mode}</code> ·
                  model <code>{settings.model || "mock"}</code>
                </span>
              </div>
              {customerIdInvalid && (
                <div className="error" role="alert">
                  customer_id must be a positive integer (got{" "}
                  <code>{settings.customerId}</code>).
                </div>
              )}
            </section>
          </main>
        ) : (
          <main className="tools-main">
            <ToolsPanel enabled={!backendDown} />
          </main>
        )}

        <footer className="footer">
          DemoCorp simulation · baseline mode · mock LLM
        </footer>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------

interface SidebarProps {
  settings: ChatSettings;
  onChange: (s: ChatSettings) => void;
  onReset: () => void;
  customerIdInvalid: boolean;
  backendDown: boolean;
  messageCount: number;
  sessionId: string | undefined;
}

function Sidebar({
  settings,
  onChange,
  onReset,
  customerIdInvalid,
  backendDown,
  messageCount,
  sessionId,
}: SidebarProps) {
  return (
    <aside className="sidebar" aria-label="chat settings">
      <h2 className="sidebar-title">Session</h2>

      <div className="sidebar-field">
        <label htmlFor="mode">mode</label>
        {/* Phase 2 only exposes baseline. The backend also accepts the two
            PromptWall modes — those are intentionally NOT in the dropdown. */}
        <select
          id="mode"
          value={settings.mode}
          onChange={(e) =>
            onChange({ ...settings, mode: e.target.value as ChatMode })
          }
        >
          <option value="baseline">baseline</option>
        </select>
      </div>

      <div className="sidebar-field">
        <label htmlFor="model">model</label>
        <input
          id="model"
          list="model-suggestions"
          value={settings.model}
          onChange={(e) => onChange({ ...settings, model: e.target.value })}
          placeholder="mock"
        />
        <datalist id="model-suggestions">
          <option value="mock" />
          <option value="gpt-4o-mini" />
          <option value="gpt-4o" />
        </datalist>
      </div>

      <div className="sidebar-field">
        <label htmlFor="customer_id">customer_id</label>
        <input
          id="customer_id"
          type="number"
          min={1}
          step={1}
          value={settings.customerId}
          onChange={(e) =>
            onChange({ ...settings, customerId: e.target.value })
          }
          placeholder="(optional)"
          aria-invalid={customerIdInvalid || undefined}
        />
        <span className="field-hint">
          Optional — when set, the chatbot knows whose account it's helping.
        </span>
      </div>

      <div className="sidebar-field">
        <label htmlFor="use_case">use_case</label>
        <input
          id="use_case"
          list="use-case-suggestions"
          value={settings.useCase}
          onChange={(e) =>
            onChange({ ...settings, useCase: e.target.value })
          }
          placeholder="demo_frontend"
        />
        <datalist id="use-case-suggestions">
          {USE_CASE_OPTIONS.map((u) => (
            <option key={u} value={u} />
          ))}
        </datalist>
        <span className="field-hint">Sent under metadata.use_case.</span>
      </div>

      <button
        type="button"
        className="reset-button"
        onClick={onReset}
        disabled={messageCount === 0 && sessionId === undefined}
        title="Clear messages and start a new chat session"
      >
        Reset chat
      </button>

      <dl className="sidebar-stats">
        <div>
          <dt>messages</dt>
          <dd>{messageCount}</dd>
        </div>
        <div>
          <dt>session</dt>
          <dd className="mono" title={sessionId ?? ""}>
            {sessionId ? `${sessionId.slice(0, 8)}…` : "(new)"}
          </dd>
        </div>
        <div>
          <dt>backend</dt>
          <dd>{backendDown ? "offline" : "online"}</dd>
        </div>
      </dl>
    </aside>
  );
}

// ---------------------------------------------------------------------------
// Health badge + chat bubbles
// ---------------------------------------------------------------------------

function HealthBadge({
  state,
  message,
}: {
  state: HealthState;
  message: string | null;
}) {
  const label =
    state === "checking" ? "checking…" : state === "ok" ? "online" : "offline";
  return (
    <span
      className={`health-badge health-${state}`}
      title={message ?? undefined}
      aria-label={`backend health: ${label}`}
    >
      <span className="health-dot" aria-hidden="true" />
      backend: {label}
    </span>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <p>
        Start a conversation by picking a demo prompt below — or type your
        own.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Demo prompts panel
// ---------------------------------------------------------------------------

interface DemoPromptsPanelProps {
  prompts: string[];
  /** When true, clicks are inert (sending in flight, backend down, or
   *  customer_id invalid). */
  disabled: boolean;
  defaultOpen: boolean;
  /** Send the prompt directly. */
  onPick: (prompt: string) => void;
  /** Drop the prompt into the textarea instead of sending. */
  onPickToInput: (prompt: string) => void;
}

function DemoPromptsPanel({
  prompts,
  disabled,
  defaultOpen,
  onPick,
  onPickToInput,
}: DemoPromptsPanelProps) {
  return (
    <details className="demo-prompts" open={defaultOpen}>
      <summary>
        <span className="demo-prompts-title">Demo questions</span>
        <span className="demo-prompts-count">{prompts.length}</span>
        <span className="demo-prompts-hint">
          click to send · Shift-click to copy into the textarea
        </span>
      </summary>
      <ul className="demo-prompts-list">
        {prompts.map((p, i) => (
          <li key={i}>
            <button
              type="button"
              className="demo-prompt"
              disabled={disabled}
              onClick={(ev) => {
                if (ev.shiftKey) {
                  onPickToInput(p);
                } else {
                  onPick(p);
                }
              }}
              title={
                disabled
                  ? "Disabled while sending or backend is offline"
                  : "Click to send · Shift-click to fill the textarea"
              }
            >
              <span className="demo-prompt-index">{i + 1}.</span>
              <span className="demo-prompt-text">{p}</span>
            </button>
          </li>
        ))}
      </ul>
    </details>
  );
}

function fmtTime(ts: number): string {
  return new Date(ts).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function UserBubble({ message }: { message: UserMessage }) {
  return (
    <div className="bubble bubble-user">
      <div className="bubble-meta">
        <span className="bubble-role">you</span>
        <time dateTime={new Date(message.ts).toISOString()}>
          {fmtTime(message.ts)}
        </time>
      </div>
      <div className="bubble-body">{message.text}</div>
    </div>
  );
}

function AssistantBubble({ message }: { message: AssistantMessage }) {
  const { response, error, pending, requestedMode, requestedModel } = message;
  const cls = error
    ? "bubble bubble-assistant bubble-error"
    : "bubble bubble-assistant";

  return (
    <div className={cls}>
      <div className="bubble-meta">
        <span className="bubble-role">assistant</span>
        <time dateTime={new Date(message.ts).toISOString()}>
          {fmtTime(message.ts)}
        </time>
      </div>

      {pending ? (
        <div className="bubble-body bubble-pending">
          <span className="spinner" aria-hidden="true" />
          Thinking…
        </div>
      ) : error ? (
        <div className="bubble-body">
          <strong>Request failed.</strong>
          <pre className="error-detail">{error}</pre>
        </div>
      ) : response ? (
        <>
          <div className="bubble-body">{response.answer || "(no answer)"}</div>

          <dl className="trace-meta">
            <div>
              <dt>trace_id</dt>
              <dd className="trace-id-cell">
                {response.trace_id}
                <CopyButton
                  value={String(response.trace_id)}
                  label="copy trace_id"
                />
              </dd>
            </div>
            <div>
              <dt>mode</dt>
              <dd>{requestedMode}</dd>
            </div>
            <div>
              <dt>model</dt>
              <dd>{requestedModel}</dd>
            </div>
            <div>
              <dt>latency_ms</dt>
              <dd>{response.latency_ms}</dd>
            </div>
            <div>
              <dt>cost (USD)</dt>
              <dd>{response.estimated_cost_usd}</dd>
            </div>
            <div>
              <dt>session</dt>
              <dd className="mono" title={response.session_id}>
                {response.session_id.slice(0, 8)}…
              </dd>
            </div>
          </dl>

          {response.tools_called.length > 0 && (
            <details open className="tools-called">
              <summary>
                tools called ({response.tools_called.length})
              </summary>
              <ul>
                {response.tools_called.map((tc, i) => (
                  <li key={i} className={tc.success ? "tool ok" : "tool err"}>
                    <code>{tc.name}</code>
                    <span className="latency"> · {tc.latency_ms}ms</span>
                    {tc.evidence_id && (
                      <span className="evidence"> · {tc.evidence_id}</span>
                    )}
                    {!tc.success && (
                      <span className="tool-error">
                        {" "}· {tc.error_type}: {tc.error_message}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            </details>
          )}

          {response.evidence_ids.length > 0 && (
            <p className="evidence-summary">
              grounded in {response.evidence_ids.length} evidence record(s)
            </p>
          )}

          {/* Full raw /chat response — collapsed by default. Useful for live
              demos when an audience asks "what came back?". */}
          <details className="raw-json">
            <summary>raw response JSON</summary>
            <div className="raw-json-toolbar">
              <CopyButton
                value={JSON.stringify(response, null, 2)}
                label="copy JSON"
              />
            </div>
            <pre className="schema-json">
              {JSON.stringify(response, null, 2)}
            </pre>
          </details>
        </>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small reusable copy-to-clipboard button
// ---------------------------------------------------------------------------

function CopyButton({ value, label }: { value: string; label: string }) {
  const [state, setState] = useState<"idle" | "copied" | "error">("idle");

  async function onClick(ev: React.MouseEvent<HTMLButtonElement>) {
    ev.stopPropagation();
    ev.preventDefault();
    try {
      // navigator.clipboard requires a secure context (https or localhost).
      // Both are true for our dev setup; in unsupported envs we degrade to
      // an "error" badge instead of throwing.
      if (typeof navigator !== "undefined" && navigator.clipboard) {
        await navigator.clipboard.writeText(value);
        setState("copied");
      } else {
        setState("error");
      }
    } catch {
      setState("error");
    }
    window.setTimeout(() => setState("idle"), 1500);
  }

  const text =
    state === "copied" ? "copied" : state === "error" ? "copy?" : "copy";
  return (
    <button
      type="button"
      className={`copy-button copy-${state}`}
      onClick={onClick}
      aria-label={label}
      title={label}
    >
      {text}
    </button>
  );
}
