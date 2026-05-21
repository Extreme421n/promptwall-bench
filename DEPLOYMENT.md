# DemoCorp Deployment Guide

This guide walks through running DemoCorp **locally**, packaging it with
**Docker**, and deploying the backend + frontend to common PaaS hosts
(Render, Railway, Fly, Vercel). It assumes you've already cloned the repo
and skimmed [`DEMOCORP_README.md`](./DEMOCORP_README.md).

> Reminder: **DemoCorp is the simulation lab**. Nothing in this guide
> deploys PromptWall — the chatbot runs in `mode: "baseline"` everywhere.

---

## 1. Local — bare metal (no Docker)

The most common dev loop: Python venv for the backend, `npm run dev` for the
frontend, and Postgres (either Homebrew or Docker) for the DB.

```bash
# Backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Postgres on port 5544 (matches docker-compose.yml)
docker compose up -d postgres                    # or use Homebrew Postgres
alembic upgrade head
python -m backend.scripts.seed_db --reset --small --db-url \
    "postgresql+psycopg://promptwall:promptwall@localhost:5544/promptwall_bench"

# Backend on :8000
DATABASE_URL=postgresql+psycopg://promptwall:promptwall@localhost:5544/promptwall_bench \
CORS_ORIGINS=http://localhost:5173 \
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend on :5173 (in a second terminal)
cd frontend
npm install
echo "VITE_API_BASE_URL=http://localhost:8000" > .env.local
npm run dev
```

Visit <http://localhost:5173>.

Smoke checks:

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl http://localhost:8000/tools | python -m json.tool | head -5
# "count": 42, ...
```

---

## 2. Local — full Docker stack

The included `docker-compose.yml` has a `fullstack` profile that builds and
runs Postgres + backend + frontend together.

```bash
docker compose --profile fullstack up --build

# → Postgres   :5544
# → Backend    http://localhost:8000   (uvicorn, /health, /tools, /chat)
# → Frontend   http://localhost:8080   (nginx serving the built bundle)
```

Seed the DB once Postgres is healthy (the backend container does NOT run the
seed for you):

```bash
docker compose exec backend \
  python -m backend.scripts.seed_db --reset --small \
  --db-url "postgresql+psycopg://promptwall:promptwall@postgres:5432/promptwall_bench"
```

Tear down:

```bash
docker compose --profile fullstack down            # keeps the postgres volume
docker compose --profile fullstack down -v          # also drops the volume
```

---

## 3. Environment variables

### Backend (`app/config.py` → `Settings`)

| Var | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg://promptwall:promptwall@localhost:5544/promptwall_bench` | SQLAlchemy URL. **Must use the `+psycopg` driver** (psycopg 3). Supports SQLite for dev / small-scale. |
| `TEST_DATABASE_URL` | `sqlite:///./test.db` | Used only by `pytest`. |
| `LLM_PROVIDER` | `mock` | `mock` (deterministic, no API) or `openai_compatible` (real LLM). |
| `DEFAULT_MODEL` | `mock-1` | Default model name; clients can override per-request. |
| `OPENAI_API_KEY` | unset | Required only for `openai_compatible`. |
| `OPENAI_BASE_URL` | unset | Optional — point at Groq / Together / vLLM / etc. |
| `CORS_ORIGINS` | `http://localhost:3000,…:5173,…:8080` (and 127.0.0.1 variants) | Comma-separated origins. Use `*` for any (development only). |
| `PORT` | `8000` (Docker `CMD`) | Honoured by the bundled Dockerfile entrypoint. |

### Frontend (`frontend/src/api.ts`)

| Var | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | Backend base URL. Baked into the bundle at **build time** — set it before `npm run build` / `docker build`. Only `VITE_`-prefixed vars are exposed to the browser. |

> Frontend env vars are evaluated at **build time**, not runtime. If you
> change `VITE_API_BASE_URL`, rebuild and redeploy.

---

## 4. Backend on Render / Railway / Fly

The backend Dockerfile is platform-agnostic. All three accept it and
inject `$PORT`; the entrypoint honours that.

### Common steps

1. **Provision managed Postgres** (Render Postgres, Railway Postgres, Fly
   Postgres, Supabase, or Neon). Note its connection string.
2. **Apply migrations** once, locally, pointing at the managed DB:
   ```bash
   DATABASE_URL='postgresql+psycopg://…' alembic upgrade head
   ```
3. **Seed** — see §6 for scale recommendations:
   ```bash
   DATABASE_URL='postgresql+psycopg://…' \
       python -m backend.scripts.seed_db --reset --small
   ```
4. **Deploy the backend** with the env vars from §3.
5. **Smoke** the deploy with `curl https://YOUR-BACKEND/health` and
   `curl https://YOUR-BACKEND/tools`.

### Render — using the included Dockerfile

```yaml
# render.yaml (or use the dashboard)
services:
  - type: web
    name: democorp-backend
    runtime: docker
    plan: starter
    healthCheckPath: /health
    envVars:
      - key: DATABASE_URL
        sync: false           # paste the managed-Postgres URL here
      - key: LLM_PROVIDER
        value: mock
      - key: DEFAULT_MODEL
        value: mock-1
      - key: CORS_ORIGINS
        value: https://your-frontend.vercel.app
      - key: PORT
        value: 10000          # Render's default; the Dockerfile CMD honours it
```

### Railway

* New project → "Deploy from GitHub" → select this repo.
* Service settings → Source → "Dockerfile" → root `/Dockerfile`.
* Variables → paste from §3.
* Connect a Railway Postgres add-on and reference its `DATABASE_URL`.
* Healthcheck path: `/health`.

### Fly.io

```bash
fly launch --no-deploy           # generates fly.toml from the Dockerfile
# edit fly.toml — set internal_port = 8000 and healthcheck path /health
fly secrets set DATABASE_URL='postgresql+psycopg://…' \
                CORS_ORIGINS='https://your-frontend.example.com'
fly deploy
```

---

## 5. Frontend on Vercel (recommended) or static-site hosts

The bundle is a vanilla Vite SPA; any static host works. Vercel auto-detects
Vite and needs no config beyond the env var.

### Vercel

1. **Connect** the repo. **Root Directory: `frontend`** (so Vercel runs
   `npm install && npm run build` inside `frontend/`).
2. **Environment Variables** → `VITE_API_BASE_URL=https://YOUR-BACKEND.example.com`.
3. Set the same value for **Preview** and **Production** environments (so
   PR previews talk to the right backend).
4. After the first deploy, copy the production URL and **add it to the
   backend's `CORS_ORIGINS`** — otherwise browser CORS will block requests.
5. **Smoke**: open the deploy, watch the network tab — `/health` should
   return `{"status":"ok"}` and the demo prompts should produce answers.

### Render Static Site / Netlify / Cloudflare Pages

Identical pattern:

* Build command: `npm install && npm run build`
* Publish directory: `frontend/dist`
* Env var: `VITE_API_BASE_URL=https://YOUR-BACKEND.example.com`

### Docker

```bash
docker build \
    --build-arg VITE_API_BASE_URL=https://YOUR-BACKEND.example.com \
    -t democorp-frontend ./frontend

docker run --rm -p 8080:8080 democorp-frontend
# → http://localhost:8080
```

---

## 6. Managed Postgres recommendation

| Provider | Free tier | Good for |
|---|---|---|
| **Neon** | 0.5 GB · always-on branches | small / medium scale; great DX |
| **Supabase** | 500 MB · 2 free projects | small / medium |
| **Railway Postgres** | included with backend hobby plan | medium |
| **Render Postgres** | none — 90-day free trial only | medium / large |
| **Fly Postgres** | 1× 1GB volume free | medium |

For DemoCorp's scale presets:

| Scale | Rows | DB size | Recommended hosting |
|---|---:|---:|---|
| `small`   | ~16 k | ~30 MB | any free tier |
| `medium`  | ~700 k | ~300 MB | Neon/Supabase free |
| `large`   | ~8.7 M | **~1.8 GB** | paid tier, ≥4 GB storage |

### ⚠️ Large-seed warning

The `large` preset writes **~8.7 M rows / 1.8 GB** and takes **15–25 minutes**
to seed on a typical laptop. Don't accidentally trigger it on a managed
Postgres free tier — most will rate-limit or hard-fail mid-seed. Either:

1. Seed against a **local** Postgres, then `pg_dump` + restore to the managed
   DB; or
2. Use `medium` everywhere except on machines you control end-to-end.

---

## 7. Smoke tests after deploy

Run these immediately after each deploy. They use only the public surface.

```bash
BACKEND=https://YOUR-BACKEND.example.com
FRONTEND=https://YOUR-FRONTEND.example.com

# 1. Backend liveness
curl -s "$BACKEND/health"
# {"status":"ok"}

# 2. Tool registry
curl -s "$BACKEND/tools" | python -m json.tool | head -10
# "count": 42, ...

# 3. Chat round-trip
curl -s -X POST "$BACKEND/chat" \
    -H 'Content-Type: application/json' \
    -d '{
      "mode": "baseline",
      "model": "mock",
      "message": "What is our cancellation policy?",
      "metadata": {"use_case": "post_deploy_smoke", "channel": "web"}
    }' | python -m json.tool

# 4. CORS preflight from the deployed frontend's origin
curl -s -i -X OPTIONS "$BACKEND/chat" \
    -H "Origin: $FRONTEND" \
    -H 'Access-Control-Request-Method: POST' \
    -H 'Access-Control-Request-Headers: content-type' \
  | grep -i 'access-control\|^HTTP/'
# HTTP/1.1 200 OK
# access-control-allow-origin: $FRONTEND
# access-control-allow-methods: ..., POST, ...

# 5. Frontend serves
curl -sI "$FRONTEND" | head -1
# HTTP/2 200
```

### Backend self-test suite

```bash
# Run the full pytest against the deployed backend's *test* config
# (local SQLite — does not touch the deployed Postgres).
python -m pytest
# 530 passed
```

### Production data-quality

```bash
DATABASE_URL='postgresql+psycopg://…' \
    python -m backend.scripts.data_quality_report
# Should report 0 orphans, 0 empty bodies, and 0 warnings.
```

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Frontend reaches the backend but every POST is blocked | Backend's `CORS_ORIGINS` doesn't include the frontend URL | Set `CORS_ORIGINS=https://your-frontend.example.com` (no trailing slash) on the backend; redeploy |
| `psycopg.errors.UndefinedTable: relation "policy_documents" does not exist` | Skipped `alembic upgrade head` after Phase 6B | Run `alembic upgrade head` against the managed DB |
| Frontend bundle still hits `http://localhost:8000` | `VITE_API_BASE_URL` wasn't set at **build** time | Rebuild with the env var; redeploy |
| `/chat` returns 400 "OPENAI_API_KEY is not configured" | Frontend or test asked for a non-mock model | Either pass `model: "mock"` or set `OPENAI_API_KEY` on the backend |
| Healthcheck flaps | Container's CMD started before Postgres was reachable | Make sure the platform respects `depends_on.condition: service_healthy` (Compose) or set a healthcheck grace period |
| 502 immediately after deploy on Render | Render expects the app to bind to `$PORT` | The bundled Dockerfile already does this; verify nothing overrides `PORT` |

---

## 9. What's NOT in this guide

- **PromptWall**. DemoCorp ships modes for it (`promptwall_candidate_shadow`,
  `promptwall_enforced`) and a `promptwall_candidate_decisions` table — but
  no router / enforcement / SDK lives in this repo. Treat PromptWall as a
  separate deployment that will eventually layer on top of this backend.
- **Auth.** The `/chat`, `/tools`, and `/health` endpoints are public. For a
  real deployment behind the public internet, put the backend behind an
  auth proxy (Cloudflare Access, Tailscale Funnel, an internal nginx with
  basic auth, …) or add an `Authorization` middleware before going live.
- **Observability beyond traces.** The DB already records every chat session,
  trace, LLM call, and tool invocation. Forward those to a logging stack
  (Datadog / OpenTelemetry) only if you need to.
