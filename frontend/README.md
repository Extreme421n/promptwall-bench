# DemoCorp Frontend

Minimal React + Vite + TypeScript chat UI for the DemoCorp simulation
backend. Talks to `POST /chat` in baseline mode against the mock LLM —
**no PromptWall logic** lives here.

## Prerequisites

- Node 18+ (verified on 24.x)
- The DemoCorp backend running on a reachable URL (default
  `http://localhost:8000`). The backend's CORS allow-list already includes
  `http://localhost:5173`, so no extra config is needed for the Vite dev
  server.

## Setup

```bash
cd frontend
npm install

# Optional: point at a non-default backend
cp .env.example .env.local
# edit .env.local → VITE_API_BASE_URL=http://localhost:8000
```

## Run dev server

```bash
npm run dev
# → http://localhost:5173
```

The page shows:

- **DemoCorp AI Assistant** title.
- A health badge (green when `GET /health` returns `{"status":"ok"}`,
  re-checked every 30 seconds).
- A text input + Send button (`⌘/Ctrl + Enter` also sends).
- The chatbot's answer plus `trace_id`, `session_id`, `latency_ms`,
  estimated cost, the list of tools called with their `evidence_id`s,
  and a count of evidence records.

## Build for production

```bash
npm run build      # type-check + bundle into ./dist
npm run preview    # serve the production bundle locally
```

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | Base URL of the DemoCorp backend. |

Only `VITE_`-prefixed env vars are exposed to the browser bundle. Never put
secrets in `.env.local` — this is a public client.
