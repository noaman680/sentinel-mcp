# databricks-langgraph-mcp

> **Production-grade LangGraph agent** with Databricks foundation model inference and multi-server MCP tool integration.

[![CI](https://github.com/your-org/databricks-langgraph-mcp/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/your-org/databricks-langgraph-mcp/actions/workflows/ci-cd.yml)
[![Eval Pipeline](https://github.com/your-org/databricks-langgraph-mcp/actions/workflows/eval-pipeline.yml/badge.svg)](https://github.com/your-org/databricks-langgraph-mcp/actions/workflows/eval-pipeline.yml)
[![Coverage](https://codecov.io/gh/your-org/databricks-langgraph-mcp/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/databricks-langgraph-mcp)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        StateGraph (LangGraph)                    │
│                                                                   │
│  START → tool_discovery → agent ──┬──→ tool_executor → agent    │
│                                   └──→ respond → END            │
│                                                                   │
│  Routing:  edges.should_continue()    after_tool_execution()     │
│  State:    AgentState (TypedDict)     GraphConfig (RunnableConfig│
└─────────────────────────────────────────────────────────────────┘
           │                    │
           ▼                    ▼
   ChatDatabricks          MCPClientManager
   (LLM inference)     (SSE connections, circuit-breaker,
                         schema validation, audit log)
           │                    │
           ▼                    ▼
   Databricks FMAPI      MCP Server 1 … N
```

### Key design decisions

| Concern | Approach | Why |
|---------|----------|-----|
| **Type safety** | Pydantic v2 `BaseSettings` + `TypedDict` state | Fail-fast at startup, not at request time |
| **State reducers** | `Annotated[list, add_messages]` / `lambda a, b: a + b` | Correct concurrent branch merging |
| **Error isolation** | Per-server circuit-breaker in `MCPClientManager` | One bad MCP server ≠ agent failure |
| **Secret handling** | `SecretStr` fields; no secrets in YAML/code | Prevents accidental logging of credentials |
| **Evaluation** | MLflow GenAI metrics + JUnit XML in CI | Quantified, comparable quality gates |
| **Container security** | Non-root UID, no shell in runtime stage, Trivy scan | FAANG-grade supply-chain posture |

---

## Repository layout

```
databricks-langgraph-mcp/
├── .github/workflows/
│   ├── ci-cd.yml           # Lint → test → build → vulnerability scan
│   └── eval-pipeline.yml   # LLM quality gate on every PR touching src/
├── src/mcp_agent/
│   ├── config.py           # Pydantic settings; validated at import time
│   ├── state.py            # TypedDict AgentState + ToolCallRecord
│   ├── nodes.py            # Pure async node functions (no topology knowledge)
│   ├── edges.py            # Pure routing functions (no side effects)
│   ├── agent.py            # Graph compilation + public run_agent() / stream_agent()
│   └── tools/
│       └── mcp_client.py   # Multi-server discovery, circuit-breaker, schema validation
├── tests/
│   ├── unit/               # Zero-network, < 5s total
│   ├── integration/        # Full graph trajectories, mocked boundaries
│   └── evaluation/         # MLflow GenAI metrics against golden dataset
├── config/agent_config.yaml
├── Dockerfile              # Multi-stage, non-root, OCI-labelled
└── pyproject.toml          # uv / hatchling; ruff + mypy + pytest configured
```

---

## Quick start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (`pip install uv`)
- A Databricks workspace with a served foundation model endpoint
- (Optional) One or more MCP servers

### 1. Clone and install

```bash
git clone https://github.com/your-org/databricks-langgraph-mcp.git
cd databricks-langgraph-mcp
uv sync --extra dev
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — minimum required:
# MCP_AGENT__DATABRICKS_HOST=https://adb-<id>.azuredatabricks.net
# MCP_AGENT__DATABRICKS_TOKEN=dapi...
# MCP_AGENT__MCP_SERVER_URLS=https://your-mcp-server.example.com/sse
```

### 3. Run

```python
import asyncio
from mcp_agent import run_agent

result = asyncio.run(run_agent("List all tables in the analytics catalog."))
print(result["final_answer"])
```

### 4. Stream

```python
import asyncio
from mcp_agent import stream_agent

async def main():
    async for update in stream_agent("What is the P95 query latency?"):
        print(update)

asyncio.run(main())
```

---

## Development

```bash
# Lint + format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type-check
uv run mypy src/mcp_agent --strict

# Unit tests (fast, no credentials needed)
uv run pytest tests/unit/ -v --cov=src/mcp_agent --cov-fail-under=85

# Integration tests (mocked boundaries)
uv run pytest tests/integration/ -v

# Evaluation suite (requires live Databricks + MLflow)
MLFLOW_TRACKING_URI=http://localhost:5000 uv run pytest tests/evaluation/ -v
```

---

## CI / CD

| Workflow | Trigger | Gates |
|----------|---------|-------|
| `ci-cd.yml` | Push / PR to main | ruff lint, mypy strict, pytest ≥85% coverage, Trivy HIGH/CRIT |
| `eval-pipeline.yml` | PR touching `src/` or `config/` | Tool precision ≥0.80, recall ≥0.80, keyword coverage ≥0.70, P95 latency ≤10s |

Container images are published to `ghcr.io/your-org/databricks-langgraph-mcp` on every push to `main`.

---

## Configuration reference

All settings are sourced from environment variables with prefix `MCP_AGENT__`.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MCP_AGENT__MODEL_PROVIDER` | enum | `databricks` | Inference provider |
| `MCP_AGENT__MODEL_NAME` | str | `databricks-meta-llama-3-1-70b-instruct` | Model endpoint name |
| `MCP_AGENT__DATABRICKS_HOST` | URL | — | Workspace URL (**required**) |
| `MCP_AGENT__DATABRICKS_TOKEN` | secret | — | PAT or M2M token (**required**) |
| `MCP_AGENT__MCP_SERVER_URLS` | list | `[]` | Comma-separated SSE server URLs |
| `MCP_AGENT__MAX_ITERATIONS` | int | `20` | ReAct loop ceiling |
| `MCP_AGENT__LOG_LEVEL` | enum | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `MCP_AGENT__MLFLOW_TRACKING_URI` | str | — | MLflow server URI |
| `MCP_AGENT__ENABLE_LANGSMITH_TRACING` | bool | `false` | Enable LangSmith traces |

---

## Security

- **Secrets**: never committed; loaded exclusively from environment variables via `pydantic-settings`.
- **Container**: non-root user (uid 1000), no shell binary in runtime stage, Trivy-scanned on every build.
- **MCP validation**: tool call arguments are JSON-Schema validated before any network round-trip.
- **Audit log**: every tool call produces a tamper-evident `ToolCallRecord` appended to `AgentState.tool_call_log`.

To report a vulnerability, email `security@your-org.example.com` — do not open a public issue.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).