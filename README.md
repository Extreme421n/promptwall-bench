# PromptWall Benchmark — Airline + Customer Support

A realistic baseline-vs-PromptWall benchmark for a chatbot wired to 15 typed
tools over an airline + customer support schema. Everything runs locally
with a mock LLM provider; an OpenAI-compatible provider is available for
real-LLM runs.

## What's in the box

| Layer                  | What it does                                                                       |
|------------------------|------------------------------------------------------------------------------------|
| Schema (18 tables)     | CRM, airline, support, KB, observability, evaluation, PromptWall                   |
| Seed (`small/medium/large`) | Deterministic synthetic data with deliberate, *realistic* ambiguity            |
| Tool registry          | 15 read-only tools with typed Pydantic schemas + JSON-Schema export                |
| Tool executor          | Trace-logged invocation with `evidence_id` per successful call                     |
| LLM abstraction        | `MockLLMProvider` + `OpenAICompatibleProvider` behind a single interface           |
| Chat service           | `baseline`, `promptwall_candidate_shadow`, `promptwall_enforced` modes             |
| Eval generator         | 193 cases across 19 categories (booking, refund, baggage, …, adversarial)          |
| Benchmark runner       | Concurrent, deterministic; writes `evaluation_runs` + `evaluation_results`         |
| Report + comparison    | Human readable + JSON + CSV; baseline-vs-candidate diff with router metrics        |
| PromptWall             | Pure rule-based shadow analyzer + enforcement router (no LLM, no embeddings)       |

## Quickstart

```bash
# 1. Install
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Provision the DB (SQLite by default — no Docker required)
export DATABASE_URL=sqlite:///./bench.db
alembic upgrade head
python backend/scripts/seed_db.py --reset --small

# 3. Generate eval cases (deterministic, seed=42)
python backend/scripts/generate_eval_cases.py --output data/eval/eval_cases.jsonl

# 4. Run benchmarks across all three modes
python backend/scripts/run_benchmark.py --mode baseline                    --cases data/eval/eval_cases.jsonl --concurrency 5
python backend/scripts/run_benchmark.py --mode promptwall_candidate_shadow --cases data/eval/eval_cases.jsonl --concurrency 5
python backend/scripts/run_benchmark.py --mode promptwall_enforced         --cases data/eval/eval_cases.jsonl --concurrency 5

# 5. Reports and comparisons
python backend/scripts/report_benchmark.py --run-id 1
python backend/scripts/compare_runs.py --baseline-run-id 1 --candidate-run-id 2   # shadow
python backend/scripts/compare_runs.py --baseline-run-id 1 --candidate-run-id 3   # enforced

# 6. Tests
pytest
```

## Postgres (optional)

The small preset runs fine on SQLite. The `medium` preset (~85 MB) is OK on
SQLite. The `large` preset (~1.5 GB) is best on Postgres.

```bash
docker compose up -d                     # postgres:16-alpine on :5544
export DATABASE_URL=postgresql+psycopg://promptwall:promptwall@localhost:5544/promptwall_bench
alembic upgrade head
python backend/scripts/seed_db.py --reset --large
python backend/scripts/db_size.py
```

## Real LLM (OpenAI / OpenAI-compatible)

```bash
export OPENAI_API_KEY=sk-...
export DEFAULT_MODEL=gpt-4o-mini
# optional, for non-OpenAI providers:
# export OPENAI_BASE_URL=https://api.together.xyz/v1

# Single chat
uvicorn app.main:app --reload
curl -s localhost:8000/chat -X POST -H 'Content-Type: application/json' -d '{
  "mode": "baseline",
  "model": "gpt-4o-mini",
  "message": "What is the baggage allowance on business class?"
}'

# Full benchmark with the real model
python backend/scripts/run_benchmark.py --mode baseline --model gpt-4o-mini \
    --cases data/eval/eval_cases.jsonl --concurrency 3
```

The mock provider is the default and requires no API key. The factory in
`app.llm.get_provider` routes `model="mock"` (or any `mock*`) to
`MockLLMProvider`; everything else goes to `OpenAICompatibleProvider`.

## Layout

```
demo company/
├── app/
│   ├── main.py                  # FastAPI app: /health, /tools, /tools/{name}, /tools/{name}/execute, /chat
│   ├── config.py                # pydantic-settings (DB_URL, LLM_PROVIDER, OPENAI_*)
│   ├── db.py                    # engine + session factory
│   ├── seed.py                  # batched-insert seed for small/medium/large
│   ├── models/                  # 18 SQLAlchemy 2.x models (Base.metadata)
│   ├── tools/                   # 15 read-only tools + registry + InvocationResult
│   ├── llm/                     # ChatMessage, LLMProvider; mock + openai_compatible
│   ├── services/                # TraceService, ToolExecutor, ChatService
│   ├── promptwall/              # CandidateAnalyzer (shadow) + Router (enforcement)
│   └── eval/                    # scorer + runner + report + compare
├── backend/scripts/             # CLI: seed_db, generate_eval_cases, run_benchmark,
│                                #      report_benchmark, compare_runs, db_size
├── alembic/                     # migrations 0001 → 0003
├── tests/                       # 250 pytest cases
├── data/eval/                   # generated eval cases
├── reports/                     # generated benchmark reports + comparisons
└── docker-compose.yml           # postgres on :5544
```

## End-to-end flow

```
seed_db.py → generate_eval_cases.py → run_benchmark.py → report_benchmark.py
                                                ↓
                                       compare_runs.py (when comparing two runs)
```

Each chat call produces a `traces` row and (typically) one `llm_calls` row +
N `tool_invocations` rows. In `promptwall_candidate_shadow` mode an extra
`promptwall_candidate_decisions` row is written. In `promptwall_enforced`
mode the analyzer runs *and* the router pre-executes a tool, adds a verified
evidence block to the LLM input, and removes the forced tool from the
offered specs.

## Phase index

| Phase | Theme                                                            |
|------:|------------------------------------------------------------------|
| 1B    | Schema + Alembic migration 0001 (15 core tables)                 |
| 1C    | Deterministic seed `small`                                       |
| 1D    | 8 tools + registry + `InvocationResult`                          |
| 1E    | `TraceService` + `ToolExecutor` + FastAPI tool endpoints         |
| 1F    | LLM abstraction + `MockLLMProvider` + `llm_calls` logging        |
| 1G    | `/chat` endpoint + baseline loop                                 |
| 1H    | `OpenAICompatibleProvider` + token-aware cost                    |
| 2A    | Eval generator → 150 cases JSONL                                 |
| 2B    | Benchmark runner + scorer + 0002 migration (`evaluation_*`)      |
| 2C    | Report generator (JSON + CSV)                                    |
| 2D    | `medium` + `large` seed presets + `db_size`                      |
| 2E    | 7 additional overlapping tools (15 total)                        |
| 3A    | PromptWall candidate analyzer (shadow) + 0003 migration          |
| 3B    | Baseline vs candidate comparison                                 |
| 4A    | PromptWall enforcement (pre-execute + evidence block)            |

## Migrations

```bash
alembic upgrade head      # 0001 → 0002 → 0003

# revision: short reason
# 0001 — initial schema (15 tables)
# 0002 — evaluation_runs + evaluation_results
# 0003 — promptwall_candidate_decisions
```

A fresh DB picks up every table via `Base.metadata.create_all` in 0001, so
new tables added later are also created via `checkfirst=True` migrations.

## CLI reference

```bash
# seed
backend/scripts/seed_db.py --reset {--small | --medium | --large} [--db-url URL]

# eval set
backend/scripts/generate_eval_cases.py --output PATH [--db-url URL] [--seed N]

# benchmark
backend/scripts/run_benchmark.py --mode {baseline|promptwall_candidate_shadow|promptwall_enforced} \
    --model {mock|gpt-4o-mini|...} --cases data/eval/eval_cases.jsonl \
    [--limit N] [--concurrency N] [--db-url URL] [--name NAME]

# report
backend/scripts/report_benchmark.py --run-id N [--output-dir reports] [--top N] [--db-url URL]

# compare
backend/scripts/compare_runs.py --baseline-run-id A --candidate-run-id B \
    [--output-dir reports] [--top N] [--db-url URL]

# DB size + per-table row counts
backend/scripts/db_size.py [--db-url URL]
```

## Modes

| Mode                          | What changes                                                                                |
|-------------------------------|---------------------------------------------------------------------------------------------|
| `baseline`                    | LLM is offered all 15 tool specs; multi-turn loop driven by the provider.                   |
| `promptwall_candidate_shadow` | Identical behaviour to baseline; PromptWall runs and writes a `candidate_decision` row.     |
| `promptwall_enforced`         | For 5 high-confidence patterns, PromptWall pre-executes the tool and injects evidence.      |

Behaviour neutrality of shadow mode is verified by `test_promptwall_shadow.py` and by the
`behavior_drift_count` in the comparison report.

## Tests

```bash
pytest                # 250 tests, ~2s
pytest -v             # verbose
pytest tests/test_promptwall_enforced.py
```

Tests use a session-scoped seeded SQLite fixture (`seeded_engine`) so the
~10k-row small seed is generated once and reused.

## Known limitations

- **Mock provider is heuristic.** Designed to be reasonable, not optimal —
  it intentionally fails on hard cases (e.g. lone PNRs, "non-refundable"
  substring confusion) so the benchmark has meaningful headroom.
- **PromptWall enforcement covers 5 patterns.** Phase 4A is intentionally
  scoped; the remaining 10 tools (customer search, loyalty by external id,
  change options, etc.) are not yet enforced.
- **No embedding-based KB search.** `search_kb_articles` and
  `get_policy_clause` use SQL `ILIKE`. Good enough for the demo; semantic
  search would help in real deployments.
- **No authentication on /chat or /tools.** This is a benchmark, not a
  service. Don't expose without an auth layer.
- **SQLite for `large` seed is supported but Postgres is recommended.** 1.5 GB
  on SQLite works; concurrent writes get serialised on the global write lock.
- **Single-airline domain.** All ambiguity comes from the airline + support
  schema. Multi-tenancy/multi-domain is not modelled.
- **Cost numbers are best-effort.** The OpenAI-compatible provider's pricing
  table has three entries (`gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`). Other
  models report `None` for `estimated_cost_usd`.

## Next recommended improvements

1. **Extend enforcement coverage** — add rules for the 10 unenforced tools,
   starting with `get_customer_profile` by external id and `search_change_options`
   with explicit date ranges. The router's precision (currently 92.4% across 193
   cases on the full set, 100% on the 50-case slice) should hold.
2. **Per-category benchmark slicing in the compare CLI** — `case_rows` already
   carries `category`; aggregating per category surfaces where the router is
   weakest (currently: booking phrasings like "Could you tell me the details of
   <PNR>?" — see `reports/1_failures.csv`).
3. **Anthropic provider** on the same `LLMProvider` interface for `claude-*`
   models, for parity with OpenAI.
4. **Eval expansion** to cover seat-availability, change-options, and
   open-issues edge cases with multiple competing identifiers (e.g. PNR +
   TKT- + flight number in one message).
5. **Embedding-based KB** for `search_kb_articles` and `get_policy_clause`,
   so the benchmark exercises semantic retrieval failures, not just ILIKE
   misses.
6. **Dashboard** that reads `evaluation_runs` + `evaluation_results` +
   `promptwall_candidate_decisions` and renders the comparison report
   (currently CLI-only).
7. **Stricter token accounting for the mock** — currently `len(text) / 4`;
   making it match `tiktoken` would let benchmark cost numbers track real
   models more closely.
