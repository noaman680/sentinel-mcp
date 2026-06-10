"""
agent.py — Compiles the LangGraph StateGraph and exposes the public agent API.

This module is the single composition root.  It wires together:
  • Nodes (nodes.py)   — what each step does
  • Edges (edges.py)   — how steps connect
  • State (state.py)   — the shared data contract
  • Config (config.py) — runtime parameters

Nothing in this module contains business logic; it is a pure wiring layer.
That separation means the graph topology can be visualised, tested, and
modified without touching any application code.

Public surface:
  • build_graph()  — returns a compiled CompiledStateGraph (for reuse / testing)
  • run_agent()    — convenience coroutine for single invocations
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from mcp_agent.config import GraphConfig, get_settings
from mcp_agent.edges import (
    NODE_AGENT,
    NODE_RESPOND,
    NODE_TOOL_EXECUTOR,
    after_tool_execution,
    should_continue,
)
from mcp_agent.nodes import agent_node, respond_node, tool_discovery_node, tool_executor_node
from mcp_agent.state import AgentState, initial_state

logger = logging.getLogger(__name__)

# ── Graph compilation ──────────────────────────────────────────────────────────

def build_graph():
    """
    Compile and return the StateGraph.

    Calling build_graph() is intentionally cheap — it performs no I/O.
    MCP connections are established lazily when nodes execute, so the compiled
    graph can be created at module import time and reused across requests.

    Returns:
        CompiledStateGraph — a LangGraph compiled graph ready for ainvoke / astream.
    """
    settings = get_settings()

    graph = StateGraph(AgentState, config_schema=GraphConfig)

    # ── Register nodes ─────────────────────────────────────────────────────────
    graph.add_node("tool_discovery", tool_discovery_node)
    graph.add_node(NODE_AGENT, agent_node)
    graph.add_node(NODE_TOOL_EXECUTOR, tool_executor_node)
    graph.add_node(NODE_RESPOND, respond_node)

    # ── Entry point ────────────────────────────────────────────────────────────
    graph.add_edge(START, "tool_discovery")
    graph.add_edge("tool_discovery", NODE_AGENT)

    # ── Conditional routing after agent ───────────────────────────────────────
    graph.add_conditional_edges(
        NODE_AGENT,
        should_continue,
        {NODE_TOOL_EXECUTOR: NODE_TOOL_EXECUTOR, NODE_RESPOND: NODE_RESPOND},
    )

    # ── Conditional routing after tool execution ───────────────────────────────
    graph.add_conditional_edges(
        NODE_TOOL_EXECUTOR,
        after_tool_execution,
        {NODE_AGENT: NODE_AGENT, NODE_RESPOND: NODE_RESPOND},
    )

    # ── Terminal node ──────────────────────────────────────────────────────────
    graph.add_edge(NODE_RESPOND, END)

    compiled = graph.compile()
    compiled.name = "databricks-langgraph-mcp"

    logger.info(
        "Graph compiled | nodes=%s recursion_limit=%d",
        list(graph.nodes.keys()),
        settings.recursion_limit,
    )
    return compiled


# Module-level singleton — built once, reused across requests.
_graph = None


def get_graph():
    """Return the module-level compiled graph, building it on first access."""
    global _graph  # noqa: PLW0603
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Public API ─────────────────────────────────────────────────────────────────

async def run_agent(
    user_message: str,
    *,
    session_id: str | None = None,
    configurable: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run the agent for a single user message and return the final state.

    Args:
        user_message:  The user's input text.
        session_id:    Optional stable session identifier for tracing.
        configurable:  Optional GraphConfig overrides (e.g. max_iterations).

    Returns:
        The final AgentState dict, including final_answer, tool_call_log, etc.

    Example:
        result = await run_agent("What is the average latency of my SQL endpoint?")
        print(result["final_answer"])
    """
    settings = get_settings()
    state = initial_state(user_message=user_message, session_id=session_id)

    config = RunnableConfig(
        recursion_limit=settings.recursion_limit,
        configurable=configurable or {},
    )

    graph = get_graph()
    final_state: dict[str, Any] = await graph.ainvoke(state, config=config)

    logger.info(
        "Agent run complete | session=%s status=%s iterations=%d errors=%d",
        final_state.get("session_id"),
        final_state.get("status"),
        final_state.get("iteration", 0),
        final_state.get("error_count", 0),
    )
    return final_state


async def stream_agent(
    user_message: str,
    *,
    session_id: str | None = None,
    configurable: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    Stream intermediate state updates from the agent.

    Yields one partial state dict per node execution, suitable for
    server-sent events or WebSocket streaming to a UI.

    Example:
        async for update in stream_agent("Summarise my open Jira tickets"):
            print(update)
    """
    settings = get_settings()
    state = initial_state(user_message=user_message, session_id=session_id)

    config = RunnableConfig(
        recursion_limit=settings.recursion_limit,
        configurable=configurable or {},
    )

    graph = get_graph()
    async for update in graph.astream(state, config=config, stream_mode="updates"):
        yield update