# syntax=docker/dockerfile:1.6
#
# DemoCorp frontend image (multi-stage: node-builder → nginx).
#
# The Vite bundle is built at image-build time, so the backend URL has to be
# baked in via a build-arg:
#
#     docker build \
#         --build-arg VITE_API_BASE_URL=https://api.example.com \
#         -t democorp-frontend ./frontend
#
# The nginx stage listens on $PORT (default 8080) so the same image runs on
# Render / Railway / Fly / Vercel-Static without modification.

# ---------------------------------------------------------------------------
# Stage 1 — build
# ---------------------------------------------------------------------------
FROM node:20-alpine AS builder

WORKDIR /app

# Cache deps separately from source.
COPY package.json package-lock.json* ./
RUN npm ci

# Bake the API base URL into the bundle. Empty string is allowed — the
# client falls back to http://localhost:8000 in that case.
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

COPY . .
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 — serve
# ---------------------------------------------------------------------------
FROM nginx:1.27-alpine AS runner

# SPA-friendly nginx config (try_files index.html fallback + gzip + cache).
COPY nginx.conf /etc/nginx/templates/default.conf.template

# Drop the built bundle in.
COPY --from=builder /app/dist /usr/share/nginx/html

# nginx:alpine's image honours $PORT via the official entrypoint when we use
# a .template file with the right variable. We still set a safe default.
ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider "http://127.0.0.1:${PORT}/" || exit 1
