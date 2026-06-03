"""gather_context tool — agent discovery and delegation blueprints."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from orchestrator.core.retrieval.retrieve_context import construct_context
from orchestrator.core.session_types import AgentContext, PlanContext, ToolCallRecord
from orchestrator.instrumentation.recorder import EpisodeMetricsRecorder

if TYPE_CHECKING:
    from orchestrator.core.session_types import OrchestratorSession

logger = logging.getLogger(__name__)


def _render_context(context: PlanContext) -> str:
    sections: list[str] = [
        context.blueprint_text or "No similar delegation blueprints found."
    ]

    if context.agents:
        sections.append("# Available Agents")
        sections.extend(_render_agent(agent) for agent in context.agents)

    sections.extend(
        _render_unmatched_capability(capability)
        for capability in context.unmatched_capabilities
    )
    return "\n\n".join(sections)


def _render_agent(agent: AgentContext) -> str:
    lines = [
        f"## {agent.name} (score={agent.score:.2f}, source={agent.source})",
        f"**Description:** {agent.description}",
    ]
    if agent.skills:
        lines.append(f"**Skills:** {', '.join(agent.skills)}")
    if agent.playbook:
        lines.append(f"**Playbook:**\n{agent.playbook}")
    return "\n".join(lines)


def _render_unmatched_capability(capability: str) -> str:
    return (
        f"No agents found for capability '{capability}' "
        f"Consider rephrasing or planning without it."
    )


def make_gather_context(session: "OrchestratorSession"):
    """Build the gather_context tool closed over *session*."""
    ctx = session.ctx

    @tool("gather_context")
    def gather_context(
        goal: str = "",
        capabilities: list[str] | None = None,
        task_analysis: str = "",
    ) -> str:
        """Discover available agents and delegation blueprints.

        MANDATORY before first delegation. Call again with different
        capabilities if you need agents for a new type of work.

        If execution is blocked because a concrete piece of information
        is missing, do not pass the missing artifact itself as the
        capability. Instead pass the kind of work that could obtain that
        information.

        Previously discovered agents are filtered out automatically.

        Args:
            goal: The task objective — used to search for delegation
                blueprints from similar past tasks.
            capabilities: What kinds of agents are needed. Drives agent
                search. Describe the work to be done, especially discovery
                or retrieval capabilities when you are blocked on missing data.
            task_analysis: REQUIRED on the first call. Structured analysis:
                (1) list every concrete requirement from the task,
                (2) define the target state — what must be true when done,
                (3) identify what capabilities are needed and why.
                On subsequent calls this can be empty.
        """
        if not session.context_gathered and not task_analysis.strip():
            return (
                "ERROR: task_analysis is required on the first gather_context call. "
                "Provide a structured analysis: (1) list every requirement from the task, "
                "(2) define the target state, (3) identify needed capabilities."
            )

        if task_analysis.strip() and not session.task_analysis:
            session.task_analysis = task_analysis.strip()

        context = construct_context(
            goal,
            capabilities or [],
            ctx.registry,
            ctx.episode_store,
            ctx.blueprint_store,
            ctx.playbook_store,
            embedder=ctx.embedder,
            metrics=session.metrics,
            enable_agent_filtering=ctx.enable_agent_filtering,
        )

        if context.blueprint_id:
            session.retrieved_blueprint_id = context.blueprint_id
            session.retrieved_blueprint_text = context.blueprint_text

        new_agents = [a for a in context.agents if a.name not in session.seen_agents]
        session.seen_agents.update(a.name for a in new_agents)
        context.agents = new_agents

        metrics = EpisodeMetricsRecorder(session.metrics)
        for agent in new_agents:
            metrics.record_agent_discovery(agent.name, agent.source)
            if agent.matched_capabilities:
                existing = session.agent_matched_capabilities.get(agent.name, [])
                session.agent_matched_capabilities[agent.name] = list(
                    dict.fromkeys(existing + agent.matched_capabilities)
                )

        session.context_gathered = True
        if not new_agents and not context.blueprint_text:
            return (
                "No new agents or blueprints found. "
                "All matching agents were already discovered. "
                "Proceed with the agents you have, or call gather_context() "
                "with DIFFERENT capabilities to search for other types of agents."
            )

        session.timeline.append(ToolCallRecord(
            tool_name="gather_context",
            tool_input=f"goal={goal[:60]} caps={len(capabilities or [])}",
            tool_output=(
                f"{len(new_agents)} new agents, "
                f"blueprint={'yes' if context.blueprint_text else 'no'}"
            ),
        ))

        return _render_context(context)

    return gather_context
