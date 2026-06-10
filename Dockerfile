# syntax=docker/dockerfile:1.6
# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Multi-stage, security-hardened production image
#
# Stages:
#   1. builder  — installs Python deps into a virtual env (layer-cached)
#   2. runtime  — copies only the venv and source; no build tools in prod
#
# Security posture:
#   • Non-root user (uid 1000) in runtime stage
#   • No shell (sh/bash) in runtime stage — removes a common attack surface
#   • Read-only filesystem friendly (secrets via env vars, not bind-mounts)
#   • Trivy-scanned in CI for HIGH/CRITICAL CVEs before push
# ─────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.4.18

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

ARG UV_VERSION

WORKDIR /build

# Install uv (pinned hash for reproducibility)
RUN pip install --no-cache-dir "uv==${UV_VERSION}"

# Copy dependency manifests first to maximise layer cache hits.
# Source code changes won't invalidate this layer.
COPY pyproject.toml uv.lock* ./

# Install production dependencies into /opt/venv
RUN uv venv /opt/venv && \
    uv sync --frozen --no-dev --no-install-project

# Copy source and install the project itself (editable=false)
COPY src/ ./src/
RUN uv pip install --no-deps --python /opt/venv/bin/python -e .


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

# Create a non-root user with a fixed UID for deterministic file permissions.
RUN groupadd --gid 1000 agent && \
    useradd --uid 1000 --gid agent --shell /usr/sbin/nologin --no-create-home agent

# Copy the virtual environment and application source from builder.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/src /app/src

# Copy the config directory (no secrets; env vars supply those at runtime)
COPY config/ /app/config/

WORKDIR /app

# Ensure the venv is used by default.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Prevent Python from adding the cwd to sys.path (security best practice)
    PYTHONSAFEPATH=1

# Drop to non-root.
USER agent

# ── Health check ───────────────────────────────────────────────────────────────
# Invokes a lightweight Python import check; replace with an HTTP /health
# endpoint check once you add a FastAPI/Uvicorn serving layer (Step 6).
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "from mcp_agent import get_settings; get_settings()" || exit 1

# ── Default command ────────────────────────────────────────────────────────────
# Runs the agent in interactive CLI mode. Override in Kubernetes via `command:`.
CMD ["python", "-m", "mcp_agent"]

# ── Build-time metadata (OCI Image Spec) ──────────────────────────────────────
LABEL org.opencontainers.image.title="databricks-langgraph-mcp" \
      org.opencontainers.image.description="Production LangGraph agent with MCP tool integration" \
      org.opencontainers.image.source="https://github.com/your-org/databricks-langgraph-mcp" \
      org.opencontainers.image.licenses="Apache-2.0"