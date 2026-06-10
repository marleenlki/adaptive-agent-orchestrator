"""Blueprint search and formatting."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.memory.stores import PostgresBlueprintStore
    from orchestrator.registration.registry import AgentRegistry


def format_blueprint(record) -> str:
    """Format a BlueprintRecord's delegation chain as a planner-facing string."""
    if record is None or record.blueprint is None:
        return ""

    blueprint = record.blueprint
    lines = [f'## Delegation Blueprint (workflow example from past episode: "{record.task}")']

    for i, step in enumerate(blueprint.steps):
        prefix = "  →" if i > 0 else "  "
        lines.append(
            f"{prefix} {step.agent}: {step.does}\n"
            f"      receives: {step.receives}\n"
            f"      produces: {step.produces}"
        )

    return "\n".join(lines)


def search_blueprint(
    goal: str,
    registry: "AgentRegistry",
    blueprint_store: "PostgresBlueprintStore | None",
    goal_embedding: list[float] | None = None,
) -> tuple[str, set[str], float, str]:
    """Search for a proven delegation blueprint.

    Returns (formatted_text, agent_names, similarity, blueprint_id) from the
    best match, or ("", set(), 0.0, "").
    """
    if blueprint_store is None:
        return "", set(), 0.0, ""

    available_agents = set(registry.agent_names) if registry else set()
    retrieval = blueprint_store.retrieve_blueprint(
        goal, available_agents, goal_embedding=goal_embedding,
    )
    if retrieval is None:
        return "", set(), 0.0, ""
    record, similarity = retrieval

    text = format_blueprint(record)
    agent_names: set[str] = set()
    if record.blueprint:
        for step in record.blueprint.steps:
            if step.agent and registry.get_connection(step.agent) is not None:
                agent_names.add(step.agent)

    return text, agent_names, similarity, record.blueprint_id
