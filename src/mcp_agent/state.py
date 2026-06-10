"""
state.py — Canonical state definitions for the LangGraph StateGraph.

Design principles:
  • Every field is explicitly typed; no untyped dicts anywhere in the graph.
  • Reducer annotations on list fields prevent accidental full-list replacement
    when parallel branches write concurrently.
  • Immutable sentinel values (MISSING) allow nodes to distinguish "not yet set"
    from None without nullable gymnastics.
  • ToolCallRecord provides a tamper-evident audit trail; each entry is appended,
    never mutated, so replay and forensic debugging are always possible.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ── Enumerations ──────────────────────────────────────────────────────────────

class AgentStatus(str, Enum):
    """Lifecycle state of a single agent invocation."""
    IDLE = "idle"
    RUNNING = "running"
    AWAITING_TOOL = "awaiting_tool"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"          # hit max_iterations ceiling


class ToolCallStatus(str, Enum):
    """Outcome of a single MCP tool invocation."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    REJECTED = "rejected"        # schema validation failed before the call was made


# ── Immutable audit record ─────────────────────────────────────────────────────

class ToolCallRecord(TypedDict):
    """
    Append-only audit record written after every tool call attempt.

    Using TypedDict (not dataclass) keeps this serialisable to JSON without
    any custom encoder, which is required for MLflow and LangSmith logging.
    """
    call_id: str                 # uuid4; correlates with LangChain ToolMessage.tool_call_id
    tool_name: str
    server_url: str              # which MCP server handled the call
    arguments: dict[str, Any]
    status: ToolCallStatus
    result_summary: str          # ≤ 200 chars; never raw output (PII risk)
    latency_ms: float
    timestamp_utc: str           # ISO-8601


def make_tool_call_record(
    *,
    call_id: str,
    tool_name: str,
    server_url: str,
    arguments: dict[str, Any],
    status: ToolCallStatus,
    result_summary: str,
    latency_ms: float,
) -> ToolCallRecord:
    """Factory that stamps the UTC timestamp and enforces the 200-char summary cap."""
    return ToolCallRecord(
        call_id=call_id,
        tool_name=tool_name,
        server_url=server_url,
        arguments=arguments,
        status=status,
        result_summary=result_summary[:200],
        latency_ms=round(latency_ms, 2),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


# ── Primary graph state ────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The single source of truth passed between every node in the StateGraph.

    LangGraph merges partial updates returned by each node into this dict.
    The `Annotated[list, add_messages]` reducer on `messages` is critical:
    it *appends* incoming messages rather than replacing the whole list,
    which is the correct behaviour for streaming multi-turn conversations.
    """

    # ── Conversation ──────────────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
    """Full conversation history, managed by the add_messages reducer."""

    session_id: str
    """Stable identifier for the user session; set once at graph entry."""

    # ── Execution bookkeeping ─────────────────────────────────────────────────
    status: AgentStatus
    iteration: int
    """Current ReAct iteration count; checked against max_iterations each cycle."""

    # ── Tool-call audit log ───────────────────────────────────────────────────
    tool_call_log: Annotated[list[ToolCallRecord], lambda a, b: a + b]
    """Append-only log; the reducer concatenates branch writes automatically."""

    # ── MCP discovery cache ───────────────────────────────────────────────────
    discovered_tools: list[dict[str, Any]]
    """
    Snapshot of tools discovered from all MCP servers at session start.
    Refreshed on reconnection; never mutated in-place by nodes.
    """

    # ── Error propagation ─────────────────────────────────────────────────────
    last_error: str | None
    """Human-readable error from the most recent failed operation, or None."""

    error_count: int
    """Cumulative errors this session; used by edges for circuit-breaking."""

    # ── Final output ──────────────────────────────────────────────────────────
    final_answer: str | None
    """
    Set by the respond node when the agent concludes.
    Downstream consumers should read this field rather than parsing messages.
    """


def initial_state(*, user_message: str, session_id: str | None = None) -> AgentState:
    """
    Construct a clean AgentState for a new invocation.

    Centralising initialisation here prevents individual nodes from
    accidentally omitting required fields and producing KeyErrors mid-graph.
    """
    from langchain_core.messages import HumanMessage

    return AgentState(
        messages=[HumanMessage(content=user_message)],
        session_id=session_id or str(uuid.uuid4()),
        status=AgentStatus.RUNNING,
        iteration=0,
        tool_call_log=[],
        discovered_tools=[],
        last_error=None,
        error_count=0,
        final_answer=None,
    )


# ── Config schema (passed via RunnableConfig, not stored in state) ─────────────

class GraphConfig(TypedDict, total=False):
    """
    Immutable per-invocation configuration injected via RunnableConfig["configurable"].

    Separating execution parameters from mutable state is a LangGraph best
    practice: it prevents nodes from accidentally mutating their own config
    and keeps the state diff clean for reproducibility.
    """
    max_iterations: int
    recursion_limit: int
    user_id: str              # for multi-tenant rate-limiting
    trace_id: str             # external correlation ID (e.g. from an API gateway)
    dry_run: Literal[True]    # if present, suppress all side-effecting tool calls