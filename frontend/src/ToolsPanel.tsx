import { useEffect, useMemo, useState } from "react";
import {
  ApiError,
  ToolDescription,
  ToolSummary,
  getToolDetail,
  getTools,
} from "./api";

type LoadState = "idle" | "loading" | "ready" | "error";

interface ToolsPanelProps {
  /** Re-fetch trigger (Phase 2 → 3 just uses backendOnline). */
  enabled: boolean;
}

export default function ToolsPanel({ enabled }: ToolsPanelProps) {
  const [state, setState] = useState<LoadState>("idle");
  const [tools, setTools] = useState<ToolSummary[]>([]);
  const [count, setCount] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string | null>(null);

  // Load tool list when the panel becomes visible (and backend is up).
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    setState("loading");
    setError(null);
    getTools()
      .then((r) => {
        if (cancelled) return;
        setTools(r.tools);
        setCount(r.count);
        setState("ready");
      })
      .catch((e) => {
        if (cancelled) return;
        const detail =
          e instanceof ApiError
            ? `${e.status} · ${e.message}`
            : e instanceof Error
            ? e.message
            : String(e);
        setError(detail);
        setState("error");
      });
    return () => {
      cancelled = true;
    };
  }, [enabled]);

  // Filter + group.
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return tools;
    return tools.filter(
      (t) =>
        t.name.toLowerCase().includes(q) ||
        t.domain.toLowerCase().includes(q) ||
        t.description.toLowerCase().includes(q)
    );
  }, [tools, query]);

  const grouped = useMemo(() => {
    const map = new Map<string, ToolSummary[]>();
    for (const t of filtered) {
      const key = t.domain || "(unknown)";
      const list = map.get(key);
      if (list) list.push(t);
      else map.set(key, [t]);
    }
    // Sort domains alphabetically; tools within a domain by name.
    return [...map.entries()]
      .map(([domain, list]) => ({
        domain,
        list: [...list].sort((a, b) => a.name.localeCompare(b.name)),
      }))
      .sort((a, b) => a.domain.localeCompare(b.domain));
  }, [filtered]);

  return (
    <div className="tools-panel">
      <header className="tools-header">
        <div className="tools-summary">
          <span className="tools-count">{count}</span>
          <span className="tools-count-label">
            tool{count === 1 ? "" : "s"} registered
          </span>
          {query && (
            <span className="tools-filtered">
              · showing {filtered.length}
            </span>
          )}
        </div>
        <input
          type="search"
          className="tools-search"
          placeholder="Filter by name, domain, or description…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Filter tools"
        />
      </header>

      {state === "loading" && (
        <div className="tools-state">
          <span className="spinner" aria-hidden="true" /> Loading tools…
        </div>
      )}

      {state === "error" && (
        <div className="error tools-state" role="alert">
          Failed to load tools: <code>{error}</code>
        </div>
      )}

      {state === "ready" && tools.length === 0 && (
        <div className="tools-state">No tools are registered.</div>
      )}

      {state === "ready" && tools.length > 0 && filtered.length === 0 && (
        <div className="tools-state">
          No tools match <code>{query}</code>.
        </div>
      )}

      {state === "ready" && filtered.length > 0 && (
        <div className="tools-groups">
          {grouped.map(({ domain, list }) => (
            <section key={domain} className="tools-group">
              <h3 className="tools-group-title">
                <span className="tools-domain-tag">{domain}</span>
                <span className="tools-group-count">{list.length}</span>
              </h3>
              <ul className="tools-list">
                {list.map((t) => (
                  <li key={t.name}>
                    <button
                      type="button"
                      className="tool-row"
                      onClick={() => setSelected(t.name)}
                    >
                      <div className="tool-row-head">
                        <code className="tool-name">{t.name}</code>
                        <RiskBadge level={t.risk_level} />
                        {t.read_only && (
                          <span className="badge badge-readonly" title="read-only">
                            read-only
                          </span>
                        )}
                      </div>
                      <p className="tool-row-desc">{t.description}</p>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      )}

      {selected && (
        <ToolDetailModal
          name={selected}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tool detail modal
// ---------------------------------------------------------------------------

function ToolDetailModal({
  name,
  onClose,
}: {
  name: string;
  onClose: () => void;
}) {
  const [detail, setDetail] = useState<ToolDescription | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setDetail(null);
    getToolDetail(name)
      .then((d) => {
        if (cancelled) return;
        setDetail(d);
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        const msg =
          e instanceof ApiError
            ? `${e.status} · ${e.message}`
            : e instanceof Error
            ? e.message
            : String(e);
        setError(msg);
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [name]);

  // Close on Escape.
  useEffect(() => {
    function onKey(ev: KeyboardEvent) {
      if (ev.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label={`Tool details: ${name}`}
      onClick={onClose}
    >
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <header className="modal-header">
          <code className="modal-title">{name}</code>
          <button
            type="button"
            className="modal-close"
            onClick={onClose}
            aria-label="Close"
          >
            ✕
          </button>
        </header>

        <div className="modal-body">
          {loading && (
            <div className="tools-state">
              <span className="spinner" aria-hidden="true" /> Loading details…
            </div>
          )}

          {error && (
            <div className="error" role="alert">
              {error}
            </div>
          )}

          {detail && (
            <>
              <p className="tool-row-desc">{detail.description}</p>

              <dl className="modal-meta">
                <div>
                  <dt>domain</dt>
                  <dd>{detail.domain}</dd>
                </div>
                <div>
                  <dt>risk_level</dt>
                  <dd>
                    <RiskBadge level={detail.risk_level} />
                  </dd>
                </div>
                <div>
                  <dt>read_only</dt>
                  <dd>{detail.read_only ? "yes" : "no"}</dd>
                </div>
              </dl>

              <SchemaBlock title="input_schema" schema={detail.input_schema} />
              <SchemaBlock title="output_schema" schema={detail.output_schema} />
            </>
          )}
        </div>

        <footer className="modal-footer">
          <span className="hint">Press Esc or click outside to close</span>
        </footer>
      </div>
    </div>
  );
}

function SchemaBlock({
  title,
  schema,
}: {
  title: string;
  schema: Record<string, unknown>;
}) {
  return (
    <details className="schema-block" open>
      <summary>{title}</summary>
      <pre className="schema-json">{JSON.stringify(schema, null, 2)}</pre>
    </details>
  );
}

// ---------------------------------------------------------------------------
// Risk badge — colored chip for low/medium/high
// ---------------------------------------------------------------------------

function RiskBadge({ level }: { level: string }) {
  const safe = level.toLowerCase();
  return (
    <span className={`badge badge-risk badge-risk-${safe}`} title={`risk: ${safe}`}>
      {safe}
    </span>
  );
}
