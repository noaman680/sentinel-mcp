"""
config.py — Strictly typed, environment-aware settings via Pydantic v2.

All runtime configuration is loaded once at import time and validated against
strict type contracts. Secrets are sourced exclusively from the environment;
no plaintext credentials are ever written to disk or committed to version control.
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from typing import Annotated

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelProvider(str, Enum):
    DATABRICKS = "databricks"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class AgentSettings(BaseSettings):
    """
    Central settings object.  Every field maps 1-to-1 to an environment variable
    (prefix: MCP_AGENT__).  Pydantic validates types and constraints at startup,
    so misconfigured deployments fail loudly before serving a single request.
    """

    model_config = SettingsConfigDict(
        env_prefix="MCP_AGENT__",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",          # reject unknown env vars — prevents silent misconfiguration
        validate_default=True,
    )

    # ── Model / Provider ──────────────────────────────────────────────────────
    model_provider: ModelProvider = Field(
        default=ModelProvider.DATABRICKS,
        description="Inference provider backing the agent.",
    )
    model_name: str = Field(
        default="databricks-meta-llama-3-1-70b-instruct",
        description="Fully qualified model identifier as registered in the provider.",
    )
    model_temperature: Annotated[float, Field(ge=0.0, le=2.0)] = Field(
        default=0.0,
        description="Sampling temperature; 0.0 = deterministic, ideal for tool-calling agents.",
    )
    model_max_tokens: Annotated[int, Field(gt=0, le=128_000)] = Field(
        default=4_096,
        description="Maximum tokens to generate per LLM call.",
    )

    # ── Databricks ────────────────────────────────────────────────────────────
    databricks_host: AnyHttpUrl | None = Field(
        default=None,
        description="Databricks workspace URL, e.g. https://adb-<id>.azuredatabricks.net",
    )
    databricks_token: SecretStr | None = Field(
        default=None,
        description="Personal access token or M2M OAuth token for the workspace.",
    )

    # ── MCP Servers ───────────────────────────────────────────────────────────
    mcp_server_urls: list[AnyHttpUrl] = Field(
        default_factory=list,
        description="SSE-based MCP server endpoints the agent may discover tools from.",
    )
    mcp_connection_timeout_s: Annotated[float, Field(gt=0)] = Field(
        default=10.0,
        description="Seconds to wait when establishing a connection to an MCP server.",
    )
    mcp_tool_call_timeout_s: Annotated[float, Field(gt=0)] = Field(
        default=30.0,
        description="Seconds to wait for an individual tool call to complete.",
    )

    # ── Graph Execution ───────────────────────────────────────────────────────
    max_iterations: Annotated[int, Field(gt=0, le=100)] = Field(
        default=20,
        description="Hard ceiling on ReAct iterations to prevent runaway loops.",
    )
    recursion_limit: Annotated[int, Field(gt=0, le=500)] = Field(
        default=50,
        description="LangGraph StateGraph recursion limit.",
    )

    # ── Observability ─────────────────────────────────────────────────────────
    log_level: LogLevel = Field(default=LogLevel.INFO)
    mlflow_tracking_uri: str | None = Field(
        default=None,
        description="MLflow tracking server URI for experiment logging and eval metrics.",
    )
    mlflow_experiment_name: str = Field(
        default="mcp-agent-evals",
        description="MLflow experiment to log evaluation runs under.",
    )
    enable_langsmith_tracing: bool = Field(
        default=False,
        description="Pipe LangChain traces to LangSmith for debugging.",
    )
    langsmith_api_key: SecretStr | None = Field(default=None)
    langsmith_project: str = Field(default="mcp-agent")

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("mcp_server_urls", mode="before")
    @classmethod
    def parse_server_urls(cls, v: object) -> list[str]:
        """Accept both a JSON list and a comma-separated string from env vars."""
        if isinstance(v, str):
            return [u.strip() for u in v.split(",") if u.strip()]
        return v  # type: ignore[return-value]

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> "AgentSettings":
        if self.model_provider == ModelProvider.DATABRICKS:
            if not self.databricks_host or not self.databricks_token:
                raise ValueError(
                    "MCP_AGENT__DATABRICKS_HOST and MCP_AGENT__DATABRICKS_TOKEN "
                    "are required when model_provider='databricks'."
                )
        return self

    @model_validator(mode="after")
    def warn_tracing_without_key(self) -> "AgentSettings":
        if self.enable_langsmith_tracing and not self.langsmith_api_key:
            logger.warning(
                "enable_langsmith_tracing=True but no langsmith_api_key provided; "
                "tracing will be silently disabled by the LangChain SDK."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> AgentSettings:
    """
    Return the singleton settings instance.

    Using @lru_cache guarantees that environment validation runs exactly once
    per process, avoiding repeated I/O and providing a stable object to inject
    into tests via dependency-overriding.
    """
    settings = AgentSettings()
    logger.info(
        "AgentSettings loaded | provider=%s model=%s iterations=%d",
        settings.model_provider,
        settings.model_name,
        settings.max_iterations,
    )
    return settings