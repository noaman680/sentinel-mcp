"""
edges.py — Routing functions and conditional logic for the LangGraph StateGraph.

Edges encode the *topology* of the agent loop.  They must be:
  • Pure functions (no side effects, no I/O).
  • Exhaustive — every possible state must map to a named destination node.
  • Tested independently of nodes, because topology bugs are the hardest to trace.

The routing functions in this module return string node names rather than
Enum values to keep LangGraph's conditional_edge API happy without extra
adaptation boilerplate.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from mcp_agent.config import get_settings
from mcp_agent.state import AgentState, AgentStatus

logger = logging.getLogger(__name__)

# Node name constants — kept in sync with agent.py to avoid magic strings.
NODE_AGENT = "agent"
NODE_TOOL_EXECUTOR = "tool_executor"
NODE_RESPOND = "respond"

RouteToTool = Literal["tool_executor"]
RouteToRespond = Literal["respond"]
RouteToAgent = Literal["agent"]


def should_continue(
    state: AgentState,
    config: RunnableConfig,
) -> RouteToTool | RouteToRespond:
    """
    Primary routing edge called after every agent node execution.

    Decision matrix:
    ┌──────────────────────────────────┬───────────────┐
    │ Condition                        │ Route         │
    ├──────────────────────────────────┼───────────────┤
    │ Status is terminal (SUCCEEDED,   │ → respond     │
    │   FAILED, ABORTED)               │               │
    │ Error count exceeds threshold    │ → respond     │
    │ Iteration ceiling reached        │ → respond     │
    │ Last message has tool_calls      │ → tool_exec   │
    │ Last message has no tool_calls   │ → respond     │
    └──────────────────────────────────┴───────────────┘
    """
    settings = get_settings()
    status = state.get("status")

    # ── Terminal status guard ─────────────────────────────────────────────────
    if status in (AgentStatus.SUCCEEDED, AgentStatus.FAILED, AgentStatus.ABORTED):
        logger.debug("Routing to respond — terminal status: %s", status)
        return NODE_RESPOND

    # ── Error circuit-breaker ─────────────────────────────────────────────────
    error_count = state.get("error_count", 0)
    if error_count >= 3:
        logger.warning(
            "Routing to respond — error circuit breaker tripped (error_count=%d)",
            error_count,
        )
        return NODE_RESPOND

    # ── Iteration ceiling ─────────────────────────────────────────────────────
    if state.get("iteration", 0) >= settings.max_iterations:
        logger.warning(
            "Routing to respond — iteration ceiling reached (%d)",
            settings.max_iterations,
        )
        return NODE_RESPOND

    # ── Tool call detection ───────────────────────────────────────────────────
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            logger.debug(
                "Routing to tool_executor — %d tool call(s) pending",
                len(last.tool_calls),
            )
            return NODE_TOOL_EXECUTOR

    logger.debug("Routing to respond — no tool calls in last message")
    return NODE_RESPOND


def after_tool_execution(
    state: AgentState,
    config: RunnableConfig,
) -> RouteToAgent | RouteToRespond:
    """
    Routing edge called after every tool_executor node execution.

    After tools run, we generally loop back to the agent so it can reason
    over the tool results.  The exceptions are hard failure conditions where
    looping would be harmful.
    """
    # If too many errors accumulated during this tool batch, abort.
    error_count = state.get("error_count", 0)
    if error_count >= 3:
        logger.warning(
            "Routing to respond after tool execution — error threshold reached (count=%d)",
            error_count,
        )
        return NODE_RESPOND

    if state.get("status") in (AgentStatus.FAILED, AgentStatus.ABORTED):
        return NODE_RESPOND

    logger.debug("Routing back to agent after tool execution")
    return NODE_AGENT