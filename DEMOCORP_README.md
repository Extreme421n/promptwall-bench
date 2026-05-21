# DemoCorp — Enterprise Customer-Support Simulation Lab

A realistic, deterministic synthetic enterprise: a customer-support business that
spans **airline**, **commerce**, **SaaS / billing**, **support**, and **CRM**
domains, with a real schema, real tools, real text knowledge (policies, return
rules, warranties, internal agent notes, operational incidents), and a baseline
chatbot that drives them through a typed tool registry.

> This is the **DemoCorp** simulation. It is the substrate that PromptWall
> (a separate project) will eventually evaluate against. **Nothing in this
> repository implements PromptWall routing or enforcement** — the modes are
> wired but the baseline path is what we use for everything below.

---

## 1. What DemoCorp is

DemoCorp is a stand-alone lab for **building, evaluating, and stress-testing
LLM-driven customer-support chatbots** against a realistic enterprise database.
It exists so a team can answer questions like:

- "Does our chatbot ground its answers in real company text, or does it make
  policy details up?"
- "How does it behave on customer-specific, multi-domain, or missing-context
  questions?"
- "Which tools does it confuse for which other tools?"
- "What does its trace look like end-to-end (LLM call → tool call → evidence
  → final answer)?"

To make those questions answerable, DemoCorp ships:

- **42 read-only tools** with typed Pydantic input/output schemas (airline,
  commerce, SaaS, support, CRM, KB / policies).
- **A textual knowledge layer** (policies, clauses, return rules, warranties,
  agent notes, operational incidents, resolution templates) with synonym-aware
  retrieval and explicit `match_score` / `match_reason` / `matched_fields`
  / `excerpt` on every tool result.
- **A deterministic seed** at `small` (~16 k rows), `medium` (~700 k rows),
  and `large` (~8.7 M rows / ~1.84 GB Postgres) scales.
- **1,309 eval cases** across realistic categories (booking, refund, policy,
  warranty, incident, …) with fairness flags (`missing_context_expected`,
  `clarification_acceptable`).
- **A benchmark runner + scorer** that records every trace, LLM call, tool
  invocation, and evidence_id, and emits per-domain / per-risk / per-category
  metrics + a tool-confusion matrix.
- **A data-quality report** that checks the DB is rich, consistent, and not
  broken (counts, body lengths, orphans, duplicate emails, stale flights, …).

The whole stack runs locally with a deterministic `MockLLMProvider` so no
API key is required to develop, test, or benchmark.

---

## 2. Architecture overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FastAPI HTTP layer                              │
│   GET /health · GET /tools · GET /tools/{name} ·                         │
│   POST /tools/{name}/execute · POST /chat                                │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ChatService (app/services/chat_service.py)                              │
│    • Builds [system, user] messages + tool spec list                     │
│    • Calls LLM provider; logs each turn into `llm_calls`                 │
│    • Executes requested tools via ToolExecutor                           │
│    • Persists trace + final answer; returns ChatResult                   │
│    • Stamps `prompt_version` into trace metadata                         │
└─────────────────────────────────────────────────────────────────────────┘
        │                          │                          │
        ▼                          ▼                          ▼
┌────────────────┐         ┌────────────────┐         ┌─────────────────┐
│  LLMProvider   │         │  ToolExecutor  │         │  TraceService   │
│  • mock (det.) │         │  • validates I │         │  • chat_sessions│
│  • openai-     │         │  • runs impl   │         │  • traces       │
│    compatible  │         │  • mints       │         │  • llm_calls    │
│                │         │    evidence_id │         │  • tool_invoc.  │
└────────────────┘         └────────────────┘         └─────────────────┘
        │                          │                          │
        └──────────────────────────┴──────────────────────────┘
                              │
                              ▼
                ┌─────────────────────────────┐
                │  SQLAlchemy 2.x ORM         │
                │  PostgreSQL 16/17 · SQLite  │
                └─────────────────────────────┘
```

Key directories:

| Path | What it holds |
|---|---|
| `app/models/` | 50+ SQLAlchemy models across `airline`, `commerce`, `crm`, `saas`, `support`, `kb`, `knowledge`, `observability`, `evaluation`, `promptwall` |
| `app/tools/` | 42 read-only tools; each is a `Tool(input_schema, output_schema, impl)` |
| `app/tools/_text_search.py` | Deterministic text retrieval helpers (normalize / expand_query / score_match / infer_policy_types) — no LLM |
| `app/services/` | `ChatService`, `ToolExecutor`, `TraceService` |
| `app/llm/` | `MockLLMProvider` (heuristic) + `OpenAICompatibleProvider` |
| `app/eval/` | Benchmark `RunSummary`, `Scorer`, `BenchmarkRunner` |
| `app/seed.py` | Deterministic seed (random.Random(42) + Faker.seed(42)) at small/medium/large |
| `backend/scripts/` | Operational CLI: `seed_db`, `db_size`, `text_knowledge_report`, `data_quality_report`, `generate_eval_cases`, `run_benchmark`, `report_benchmark`, `compare_runs` |
| `data/eval/` | `eval_cases.jsonl` (1,309 cases) + `eval_cases_large.jsonl` |
| `alembic/` | 7 migrations, head includes Phase 6B textual-knowledge schema |
| `reports/` | Benchmark report output (JSON + CSV) |
| `tests/` | 520 pytest tests, SQLite-only fixtures |

---

## 3. Domains included

| Domain | Example entities / tools | Example questions it answers |
|---|---|---|
| **Airline** | Flights, bookings, refunds, seats, baggage policy, loyalty | "What's my booking status?" "Baggage allowance in business?" "Why hasn't my refund cleared?" |
| **Commerce** | Products, categories, orders, shipments, returns, warranty terms, return rules | "Where's my order?" "Can I return an opened item?" "What's the warranty on SKU-000001?" |
| **SaaS / billing** | Organizations, subscriptions, invoices, plans, usage events, overage charges, seat allocations | "Why was I charged overage?" "Show my invoice." "How many seats are we using?" |
| **Support** | Support tickets + messages, escalation policy, resolution templates, operational incidents | "Any update on TKT-…?" "Open a ticket draft." "Any active outages affecting commerce?" |
| **CRM** | Customers, customer-organizations, segments, internal agent notes | "What segment is this customer in?" "Any internal notes on customer 100?" |
| **KB / policies** | Policy documents (versioned), policy clauses, KB articles | "What's our cancellation policy?" "List versions of the refund policy." "Show active baggage_policy." |

---

## 4. Database size and main tables

`small` and `medium` seeds run on SQLite or Postgres. `large` is meant for
Postgres (~1.84 GB; ~8.7 M rows). All counts below are the live large preset.

### Operational tables (highlights)

| Table | Rows | Size |
|---|---:|---:|
| `customers` | 200,000 | 46.6 MB |
| `bookings` | 500,000 | 80.6 MB |
| `support_tickets` | 200,000 | 38.5 MB |
| `support_messages` | 2,000,000 | 868.5 MB |
| `kb_articles` | 5,000 | 9.9 MB |
| `invoices` | 80,000 | 13.1 MB |
| `products` | 50,000 | 10.2 MB |
| `commerce_orders` | 300,000 | 46.7 MB |
| `shipments` | 280,000 | 47.2 MB |

### Text knowledge tables

| Table | Rows | What it stores |
|---|---:|---|
| `policy_documents` | 2,000 | Versioned policy docs across 5 domains × 10 policy_types |
| `policy_clauses` | 15,000 | Structured clauses inside each policy (clause_key, body, applies_to, exceptions, severity) |
| `product_return_rules` | 5,000 | Per-category return rules (window, restocking %, opened-item allowed, exceptions) |
| `product_warranty_terms` | 50,000 | Per-product warranty (type, duration, body, exclusions) |
| `internal_agent_notes` | 500,000 | Operator-facing notes on customers (vip_handling, exception_grant, refund_promised, …) |
| `operational_incidents` | 5,000 | Outage/disruption rows (title, body, domain, started_at, resolved_at, affected_entities_json) |
| `support_resolution_templates` | 2,000 | Per-category response templates (refund_delay, baggage_lost, …) |

### Total

**1.84 GB · 8,697,020 rows · 7 textual tables · 218 MB of text knowledge.**

---

## 5. Tools overview

42 read-only tools, registered into `app.tools.default_registry`. Every tool
has a typed Pydantic input schema (with `extra="forbid"` strict validation),
a typed output schema, and a `domain` / `risk_level` / `read_only` tag.

| Domain | # | Tools |
|---|---:|---|
| **airline** | 8 | `get_booking_details`, `get_flight_status`, `search_available_flights`, `get_refund_status`, `get_baggage_policy`, `search_available_seats`, `calculate_change_fee`, `search_change_options` |
| **commerce** | 10 | `search_products`, `get_product_details`, `check_product_inventory`, `get_commerce_order_status`, `get_commerce_refund_status`, `get_commerce_return_status`, `get_shipment_status`, `calculate_bundle_price`, `search_return_rules`, `get_product_warranty_terms` |
| **saas** | 6 | `get_subscription_status`, `get_plan_limits`, `get_invoice_status`, `calculate_usage_overage`, `get_api_usage_summary`, `get_saas_seat_allocation` |
| **support** | 8 | `get_support_ticket_status`, `search_support_tickets`, `get_escalation_policy`, `create_support_ticket_draft`, `get_customer_open_issues`, `search_internal_agent_notes`, `search_operational_incidents`, `get_support_resolution_template` |
| **crm** | 4 | `get_customer_profile`, `search_customer_records`, `get_loyalty_balance`, `get_customer_segment` |
| **kb / policies** | 6 | `search_kb_articles`, `search_policy_documents`, `get_policy_clause`, `get_latest_policy_version`, `list_policy_versions`, `get_active_policy` |

Every tool runs through `ToolExecutor`, which:
1. Validates the input against the Pydantic schema.
2. Calls the impl.
3. Mints an `evidence_id` (`ev_<uuid>`) on success.
4. Logs a `tool_invocations` row with full input/output JSON.
5. Returns a typed result the chat layer can hand back to the model.

---

## 6. Text knowledge system

DemoCorp's textual retrieval is built to be **lightweight, deterministic, and
useful at scale** — no embeddings, no LLM in the loop. It lives in
`app/tools/_text_search.py` and is used by the 7 text tools listed below.

### How it works

1. **Normalize** the query: lowercase, strip punctuation, collapse whitespace.
2. **Expand** via two hand-curated synonym layers:
   - **Phrase synonyms** (`damaged packaging` → opened box / packaging damage / broken seals; `delayed flight` → flight delay / late departure / IRROPS; `missing accessories`, `invoice dispute`, `overage charge`, `checked bag`, `cabin bag`, …)
   - **Word synonyms** (`opened` → open / used / unsealed; `electronic` → electronics / device; `cancellation` → cancel / cancelled / canceled; `baggage` → luggage / bag; `warranty` → coverage / repair / replacement / exclusion; …)
   - Safe singularization (`policies` → policy) with a protected-lemma list.
3. **Search** across multiple fields per table:
   - return rules: `rule_name`, `body`, `exceptions`, `product_category_name`
   - policy clauses: `title`, `body`, `clause_key`, `applies_to`, `exceptions`
   - policy documents: `title`, `body`, `policy_type`
   - incidents: `title`, `body`, `incident_type`
   - templates: `title`, `body`, `category`
4. **Re-rank** in Python via `score_match(terms, fields)` — weighted-substring scoring with field weights (title=3, clause_key/policy_type=2.5, applies_to/exceptions=1.5, body=1).
5. **Infer + boost**: `infer_policy_types(query)` maps "cancellation rules" → `cancellation_policy`, "overage charge" → `overage_policy`, etc. Rows whose `policy_type` matches the inferred slug get a ×2 score boost and a CASE-driven SQL re-ordering so they always survive the candidate window.
6. **Fall back** when free-text returns nothing: try the inferred policy_type, or the category name for return rules.
7. **Explain** every match: every text-tool item returns `match_score` (0–1), `match_reason` ("matched 'electronics' in rule_name; 'electronics' in body"), `matched_fields` (list), and `excerpt` (whitespace-normalized, ≤240 chars).

### The 7 text-retrieval tools

| Tool | Domain | What it returns |
|---|---|---|
| `search_policy_documents` | kb | Active policy docs ranked by score, with excerpts, inferred_policy_types, fallback_used |
| `get_policy_clause` | kb | Ranked clauses joined with their parent document |
| `get_active_policy` | kb | Single active policy doc by (domain, policy_type) |
| `list_policy_versions` | kb | Every version (active + historical) of a policy |
| `search_return_rules` | commerce | Ranked return rules with window, restocking %, opened-item flag |
| `get_product_warranty_terms` | commerce | Warranty rows (body + exclusions + excerpts) by sku or product_id |
| `search_internal_agent_notes` | support | Customer-scoped notes (optional free-text); newest first |
| `search_operational_incidents` | support | Incidents by domain / text / active-only |
| `get_support_resolution_template` | support | Resolution templates by category and/or query |

---

## 7. How to run locally

DemoCorp targets **Python 3.11+**. Postgres is optional for `small`/`medium`;
required for `large`.

```bash
# 1. Clone + virtualenv
git clone <this-repo> democorp
cd democorp
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install (editable + dev deps)
pip install -e ".[dev]"

# 3. Run the test suite (uses SQLite fixtures, no Docker, no Postgres)
python -m pytest
# expected: 520 passed
```

Optional `.env` (or export inline):

```bash
# SQLite default (no setup required)
export DATABASE_URL=sqlite:///./democorp.db

# OR local Postgres (matches docker-compose.yml)
# export DATABASE_URL=postgresql+psycopg://promptwall:promptwall@localhost:5544/promptwall_bench

# LLM provider — `mock` requires no API key
export LLM_PROVIDER=mock
export DEFAULT_MODEL=mock-1
```

---

## 8. How to seed small DB

```bash
# SQLite, fresh DB
python -m backend.scripts.seed_db --reset --small --db-url sqlite:///./democorp.db

# Output:
#   [seed] customers           1,000 rows  in   0.10s
#   [seed] flights               500 rows  in   0.04s
#   ...
#   [seed] knowledge/text     16,150 rows  in   2.30s
# Total: ~16,000 rows
```

Sanity check after seeding:

```bash
DATABASE_URL=sqlite:///./democorp.db python -m backend.scripts.db_size
DATABASE_URL=sqlite:///./democorp.db python -m backend.scripts.text_knowledge_report
DATABASE_URL=sqlite:///./democorp.db python -m backend.scripts.data_quality_report
```

---

## 9. How to seed large Postgres DB

Set up Postgres first. Two common paths:

**Option A — local Homebrew Postgres** (the layout this repo was verified against):

```bash
# macOS
brew install postgresql@17
brew services start postgresql@17
createdb promptwall_bench

export DATABASE_URL="postgresql+psycopg://$USER@localhost:5432/promptwall_bench"
```

**Option B — Docker Compose** (matches `docker-compose.yml`):

```bash
docker-compose up -d postgres
export DATABASE_URL="postgresql+psycopg://promptwall:promptwall@localhost:5544/promptwall_bench"
```

Then apply migrations + seed at `large` scale:

```bash
alembic upgrade head

# Full reseed (takes ~15-20 minutes; produces ~1.84 GB)
python -m backend.scripts.seed_db --reset --large --db-url "$DATABASE_URL"
```

Verify:

```bash
python -m backend.scripts.db_size --db-url "$DATABASE_URL"
# total size: 1.84 GB · total rows 8,697,020

python -m backend.scripts.text_knowledge_report --db-url "$DATABASE_URL"
# 2,000 policy_documents · 15,000 policy_clauses · 500,000 internal_agent_notes · …
```

> If the operational tables already exist (e.g. you're upgrading from a
> pre-Phase-6B seed) but the textual knowledge tables are empty, `alembic
> upgrade head` will create them and you can populate just those tables
> incrementally — see the `_seed_knowledge` helper in `app/seed.py`.

---

## 10. How to run the chatbot

The chatbot is a FastAPI app. It runs against any seeded DB.

```bash
# Start the API server
DATABASE_URL=sqlite:///./democorp.db  uvicorn app.main:app --reload --port 8000

# Healthcheck
curl http://localhost:8000/health
# {"status":"ok"}

# Send a chat turn (mock LLM provider; no API key needed)
curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"mode":"baseline","model":"mock","message":"What is our cancellation policy?"}' \
  | python -m json.tool

# Response includes:
#   trace_id, session_id, tools_called[], evidence_ids[],
#   answer, latency_ms, estimated_cost_usd
```

Provider switches (set in `.env` or inline):

```bash
# Default: mock (deterministic heuristics, no network)
export LLM_PROVIDER=mock  DEFAULT_MODEL=mock-1

# OpenAI-compatible (real LLM)
export LLM_PROVIDER=openai_compatible
export DEFAULT_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-...
# export OPENAI_BASE_URL=https://api.groq.com/openai/v1   # optional
```

The chat service stamps every trace's metadata with `prompt_version=
baseline_v2_text_knowledge` (the v2 prompt explicitly mentions the text-
retrieval tools and a "do not invent" rule for policy details, return windows,
warranty exclusions, SLA commitments, internal notes, etc.).

---

## 11. How to run tools manually

Every tool is reachable via the registry **and** via HTTP.

### From the API

```bash
# List all tools (count + name + description + domain + risk_level)
curl -s http://localhost:8000/tools | python -m json.tool

# Describe one tool (full input + output JSON-Schema)
curl -s http://localhost:8000/tools/search_return_rules | python -m json.tool

# Execute a tool directly (200 with success=false on validation/not-found,
# 404 only when the tool name doesn't exist in the registry)
curl -s -X POST http://localhost:8000/tools/search_return_rules/execute \
  -H 'Content-Type: application/json' \
  -d '{"input": {"query": "opened electronic product"}}' \
  | python -m json.tool
# → tool returns ranked rules with match_score / match_reason / matched_fields / excerpt

curl -s -X POST http://localhost:8000/tools/get_policy_clause/execute \
  -H 'Content-Type: application/json' \
  -d '{"input": {"query": "cancellation rules"}}' \
  | python -m json.tool

curl -s -X POST http://localhost:8000/tools/get_product_warranty_terms/execute \
  -H 'Content-Type: application/json' \
  -d '{"input": {"sku": "SKU-000001"}}' \
  | python -m json.tool
```

### From Python

```python
from sqlalchemy.orm import sessionmaker
from app.db import make_engine
from app.tools import get_policy_clause, search_return_rules

engine = make_engine("sqlite:///./democorp.db")
S = sessionmaker(bind=engine)
session = S()

out = get_policy_clause.call(session, {"query": "warranty exclusions"})
print(out["count"], "clauses")
for c in out["clauses"][:3]:
    print(c["policy_type"], "·", c["title"], "·", c["match_score"])
    print("  ", c["excerpt"])
```

---

## 12. How to run the benchmark

The benchmark replays `data/eval/eval_cases.jsonl` (1,309 cases) through the
chat service and scores every run.

```bash
# Small SQLite run (good for iteration, ~1.7 s for 100 cases)
python -m backend.scripts.run_benchmark \
    --mode baseline --model mock --limit 100 \
    --db-url sqlite:///./democorp.db \
    --name local_smoke

# Full Postgres large run
python -m backend.scripts.run_benchmark \
    --mode baseline --model mock \
    --cases data/eval/eval_cases_large.jsonl \
    --db-url "$DATABASE_URL" \
    --concurrency 4 \
    --name pg_large
```

Output (printed inline + persisted to `evaluation_runs` + `evaluation_results`):

```
tool_called_when_required_rate    78.6%
tool_skip_rate                    21.4%
expected_tool_hit_rate            68.0%
wrong_tool_rate                   11.7%
missing_evidence_rate             32.7%
clarification_rate                11.2%
suspicious_unsupported_claim_rate 10.2%
average_latency_ms                29.8 ms
p95_latency_ms                    90.3 ms
```

Eval-case shape (each line is a JSON object):

```json
{
  "id": "eval_001",
  "category": "booking",
  "message": "What is the status of booking S1FOI0?",
  "expected_tools": ["get_booking_details"],
  "must_use_tool": true,
  "expected_domain": "airline",
  "risk": "medium",
  "notes": "Booking PNR provided; expect a booking lookup.",
  "customer_id": null,
  "missing_context_expected": false,
  "clarification_acceptable": false
}
```

Regenerate the eval JSONL from the current DB (deterministic, seed=42):

```bash
python -m backend.scripts.generate_eval_cases \
    --db-url "$DATABASE_URL" \
    --output data/eval/eval_cases_large.jsonl
```

---

## 13. How to generate reports

### Benchmark report (per run)

```bash
python -m backend.scripts.report_benchmark \
    --run-id 1 \
    --db-url "$DATABASE_URL" \
    --output-dir reports/local

# Writes:
#   reports/local/1_summary.json         — top-level metrics + per-domain
#   reports/local/1_failures.csv         — every failed case with expected/actual
#   reports/local/1_tool_confusion.csv   — top mis-routings
#   reports/local/1_domain_metrics.json  — per-domain breakdown
```

### Compare two runs (baseline vs. anything else)

```bash
python -m backend.scripts.compare_runs --run-a 1 --run-b 2 --db-url "$DATABASE_URL"
```

### Text-knowledge report

Inventory + sample excerpts from every text table.

```bash
python -m backend.scripts.text_knowledge_report --db-url "$DATABASE_URL"
python -m backend.scripts.text_knowledge_report --db-url "$DATABASE_URL" --json
```

### Data-quality report

28 metrics across text-knowledge / relationship integrity / operational
consistency, plus threshold-driven warnings.

```bash
python -m backend.scripts.data_quality_report --db-url "$DATABASE_URL"
python -m backend.scripts.data_quality_report --db-url "$DATABASE_URL" --json
```

Exit codes: `0` when only `warning` severity (or none); `1` when any
`error`-severity issue (orphans, missing FKs, …). Useful in CI.

### DB size report

```bash
python -m backend.scripts.db_size --db-url "$DATABASE_URL"
```

---

## 14. How to inspect traces

Every chat turn writes a row to `chat_sessions`, `traces`, `llm_calls`, and
`tool_invocations`. Inspect them directly:

```sql
-- The last 10 traces with their final answer + latency
SELECT t.id, t.mode, t.user_message, t.final_answer, t.latency_ms,
       t.extra_metadata AS metadata
FROM   traces t
ORDER  BY t.id DESC
LIMIT  10;

-- Every tool the chatbot called for trace 42
SELECT ti.tool_name, ti.success, ti.evidence_id, ti.latency_ms,
       ti.error_type, ti.error_message
FROM   tool_invocations ti
WHERE  ti.trace_id = 42
ORDER  BY ti.id;

-- The LLM round-trips for the same trace
SELECT lc.id, lc.provider, lc.model,
       lc.prompt_tokens, lc.completion_tokens, lc.estimated_cost_usd,
       lc.output_message, lc.tool_calls_requested
FROM   llm_calls lc
WHERE  lc.trace_id = 42
ORDER  BY lc.id;

-- A specific evidence_id back to the tool output it came from
SELECT tool_name, input_json, output_json
FROM   tool_invocations
WHERE  evidence_id = 'ev_3b1e635795814ac98e15adb38ca9d335';
```

The prompt version that drove each trace lives in `traces.metadata`:

```sql
SELECT extra_metadata->>'prompt_version' AS prompt_version, COUNT(*)
FROM   traces
GROUP  BY 1;
-- baseline_v2_text_knowledge | 504
-- baseline_v1                |   0   ← legacy runs; kept as a constant for A/B
```

For ad-hoc Python inspection:

```python
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker
from app.db import make_engine
from app.models import Trace, ToolInvocation, LLMCall

engine = make_engine("sqlite:///./democorp.db")
S = sessionmaker(bind=engine); s = S()

trace = s.execute(select(Trace).order_by(Trace.id.desc()).limit(1)).scalar_one()
print(trace.id, trace.mode, trace.final_answer)
print("metadata:", trace.extra_metadata)
for inv in trace.tool_invocations:
    print(" ", inv.tool_name, "→", inv.evidence_id, "ok=", inv.success)
```

---

## 15. Known limitations

DemoCorp is a **simulation lab**. We document the rough edges so they don't
surprise you mid-demo.

1. **The mock LLM is heuristic, not adaptive.** It routes accurately on the
   ~30 phrasings the heuristics know about and asks for clarification
   otherwise. Two specific phrasings still under-route:
   - *"What happens if my flight is delayed more than 3 hours?"* — the mock
     hits the flight-status heuristic and asks for a flight number. The
     `get_policy_clause` tool answers correctly via direct API call.
   - *"Why was I charged overage?"* — no narrow heuristic; falls through to
     generic clarification. The `get_policy_clause` tool answers correctly
     via direct API call (overage_policy boost).
2. **Mock summarization is terse.** Some tool results (warranty exclusions,
   incident bodies) come back with full text in `body` / `excerpt` but the
   mock's `_summarize_payload` only renders type+duration. The data is
   grounded; the visible string is just short.
3. **Text retrieval is ILIKE-based** with synonym expansion. No `tsvector` /
   GIN index yet. Latency stays under 100 ms p95 at 15 k clauses, but a
   full-text index would help if clauses grow past 100 k.
4. **`closed_ticket_open_refund_count`** in the data-quality report is
   permanently `"not_available"` until `support_tickets` gains a
   `related_booking_reference` column. The check is wired but inert.
5. **Faker-padded tails** appear in the medium/large seed bodies for
   `internal_agent_notes`, `product_warranty_terms`, and `product_return_rules`
   (e.g. "…well feeling realize town before discussion forward"). The first
   sentence is real and coherent — the tail is decorative.
6. **Operational counters drift with calendar time.** `stale_flight_status_count`
   and `pending_refunds_past_due_count` will grow as the seed's mid-2025 dates
   recede further into the past. Re-seed or run an aging job if the demo lives
   long-term.
7. **No real-time data.** The seed is deterministic (`Random(42) +
   Faker.seed(42)`). Nothing updates on its own.
8. **`get_product_warranty_terms` accepts only sku or product_id** — there's no
   free-text "warranty exclusions for laptops" search yet. The clause-based
   answer comes via `get_policy_clause` instead.

---

## 16. Recommended next steps before installing PromptWall

Before layering a router / enforcement system on top of this lab, harden
DemoCorp itself. None of these require touching PromptWall code.

1. **Tighten short excerpts.** Add a curated 1–2 sentence `summary` column to
   `policy_documents`, `policy_clauses`, `product_return_rules`,
   `product_warranty_terms`, and `operational_incidents`. Tools return that
   instead of a runtime-truncated body — removes the faker-tail visibility.
2. **Add a real full-text-search index** (`tsvector` + GIN on `body` columns
   in Postgres). Keep the SQLite ILIKE path as a fallback.
3. **Close the mock-LLM heuristic gaps** for "delayed flight" and "overage
   charge" so the baseline benchmark numbers reflect actual retrieval
   capability, not heuristic gaps.
4. **Enrich the mock's `_summarize_payload`** for the warranty / incident
   tools so the user-visible answer includes exclusions / impact / template
   subject. The data is already in the tool result.
5. **Add a `search_warranty_terms` tool** (free-text across body + exclusions
   + product name) so "warranty exclusions for laptops" routes natively.
6. **Cross-link incidents to bookings / orders / customers** via a join table
   or an `affected_entity_ids` column. Today `affected_entities_json` carries
   only a count.
7. **Versioned policy diffs.** `policy_documents.version` increments, but
   every version reuses the same text. Make a fraction of versions carry
   intentional edits (window 14 → 30 days, fee 15 % → 10 %, …) so
   `list_policy_versions` shows real change history.
8. **Auto-aging the seed.** A nightly job that nudges
   `Refund.expected_resolution_date` and `Flight.scheduled_departure` forward
   keeps operational counters in a realistic band over real-world time.
9. **`seed --refresh-knowledge-only`** subcommand to repopulate just the
   Phase-6B text tables in place. Useful when iterating on policy text without
   losing the 1.6 GB of customer / booking / commerce data.
10. **CI gate on data-quality.** `python -m backend.scripts.data_quality_report
    --json` already exits 1 on error-severity issues; add a `--strict` flag
    that also fails on warnings, and wire it into CI so the simulation can't
    regress between phases.
11. **Schema documentation in OpenAPI.** The FastAPI app already exposes
    `/docs` and `/openapi.json`. Add per-tool example invocations to the
    OpenAPI schema so a downstream consumer can browse them with realistic
    inputs.
12. **Frontend demo.** A tiny static page that hits `/chat` and renders the
    trace tree (LLM call → tool call → evidence → final answer) makes the
    lab presentable to non-engineers.

Once the above are in, DemoCorp is a solid substrate for any external
evaluator — PromptWall or otherwise.

---

## License & layout reminder

Everything under this directory is the DemoCorp simulation. PromptWall code
that *does* live alongside (`app/promptwall/`, the `promptwall_*` mode
strings, the `promptwall_candidate_decisions` table) is **not part of
DemoCorp** — it is a separate evaluator that is wired into `ChatService`
behind opt-in mode flags. The baseline path used by every command in this
README never invokes it.
