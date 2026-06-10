"""delegate tool — send tasks to sub-agents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from orchestrator.core.session_types import DelegationExchange
from orchestrator.shared.constants import (
    DEFAULT_THREAD_ID,
    ORCHESTRATOR_DELEGATION_SOURCE,
    STEP_FAILED_PREFIX,
)

if TYPE_CHECKING:
    from orchestrator.core.session_types import OrchestratorSession
    from orchestrator.instrumentation.trajectory import AgentCallTracker
    from orchestrator.registration.client import AgentClient

logger = logging.getLogger(__name__)


def make_delegate(session: "OrchestratorSession"):
    """Build the delegate tool closed over *session*."""
    ctx = session.ctx
    registry = ctx.registry
    call_tracker = ctx.call_tracker

    @tool("delegate")
    def delegate(
        agent: str,
        instruction: str,
        reasoning: str,
        cited_bullets: str = "",
    ) -> str:
        """Delegate a task to an agent and return its response.

        Args:
            agent: Agent name (must have been discovered via gather_context).
            instruction: Self-contained instruction — agents are stateless.
            reasoning: WHY this agent with this instruction. Stored for learning.
            cited_bullets: Comma-separated playbook bullet IDs that motivated this delegation (e.g. 'agent-1,agent-3').
        """
        connection = registry.get_connection(agent)
        if connection is None:
            return f"Error: Agent '{agent}' not found."

        logger.info(
            "[delegate] agent=%s instruction=%s",
            agent, instruction[:80],
        )

        try:
            output = _send_to_agent(
                connection,
                agent,
                instruction,
                call_tracker=call_tracker,
                thread_id=session.thread_id,
                enable_subagent_memory=ctx.enable_subagent_memory,
            )
        except Exception as exc:
            output = f"{STEP_FAILED_PREFIX}: Agent delegation error: {type(exc).__name__}: {exc}"
            logger.error("[delegate] Delegation crashed for %s: %s", agent, exc, exc_info=True)

        session.timeline.append(DelegationExchange(
            step_number=len(session.history) + 1,
            agent=agent,
            instruction=instruction,
            actual_output=output,
            reasoning=reasoning,
            success=not output.startswith(STEP_FAILED_PREFIX),
            feedback="",
            cited_bullets=[b.strip() for b in cited_bullets.split(",") if b.strip()],
        ))

        return output

    return delegate


def _send_to_agent(
    connection: "AgentClient",
    agent_name: str,
    message: str,
    *,
    call_tracker: "AgentCallTracker | None" = None,
    thread_id: str = DEFAULT_THREAD_ID,
    enable_subagent_memory: bool = True,
    source: str = ORCHESTRATOR_DELEGATION_SOURCE,
) -> str:
    """Send a message to an agent and return its response."""
    try:
        if enable_subagent_memory:
            agent_tid = f"{thread_id}::agent::{agent_name}"
            response = connection.send_message(text=message, thread_id=agent_tid)
        else:
            response = connection.send_message(text=message)
    except Exception as exc:
        logger.exception("Agent '%s' failed", agent_name)
        response = f"{STEP_FAILED_PREFIX}: {type(exc).__name__}: {exc}"

    if call_tracker is not None:
        call_tracker.record(agent_name, message=message, response=response, source=source)

    return response
