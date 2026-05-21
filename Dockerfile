# syntax=docker/dockerfile:1.6
#
# DemoCorp backend image.
#
# Build:
#     docker build -t democorp-backend .
#
# Run (local Postgres on the host; macOS / Linux):
#     docker run --rm -p 8000:8000 \
#         -e DATABASE_URL='postgresql+psycopg://user:pass@host.docker.internal:5432/democorp' \
#         -e CORS_ORIGINS='https://your-frontend.example.com' \
#         democorp-backend
#
# The image listens on $PORT (default 8000) — convenient for Render / Railway
# / Fly / any platform that injects PORT.

FROM python:3.11-slim AS base

# Build deps for psycopg + a few common build tools. We keep them in the
# final image because the slim base lacks them and they're tiny.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so the layer caches when only source changes.
COPY pyproject.toml ./
RUN pip install --upgrade pip \
 && pip install -e .

# Copy application source. (Heavy seed/test/report data is excluded via
# .dockerignore so the image stays small.)
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
COPY backend ./backend

# Non-root user — pure hygiene.
RUN groupadd --system --gid 1001 democorp \
 && useradd  --system --uid 1001 --gid democorp --no-create-home democorp \
 && chown -R democorp:democorp /app
USER democorp

ENV PORT=8000
EXPOSE 8000

# Lightweight container-level healthcheck that uses /health (no external deps).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail --silent "http://127.0.0.1:${PORT}/health" || exit 1

# Single-process uvicorn by default. For higher throughput, override the CMD
# with `--workers 2` (Render free tier won't benefit from more than 1).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
