"""Agent metadata model"""

from __future__ import annotations
from dataclasses import dataclass, field
from a2a.types import AgentSkill


@dataclass
class AgentCard:
    """Metadata of an agent (name, description, skills). This is used for agent registry and discovery.

    Attributes:
        name: The name of the agent.
        description: A brief description of the agent's capabilities.
        skills: A list of skills that the agent possesses.
        card_embedding: Pre-computed embedding of the card text (avoids
            runtime embed calls when supplied).
        profiled_bullets: Pre-computed capability bullets.
            When supplied at registration, these are seeded into the playbook
            store and the agent is marked as profiled so the LLM profiler
            is skipped.
    """

    name: str
    description: str = ""
    skills: list[AgentSkill] = field(default_factory=list)
    card_embedding: list[float] = field(default_factory=list)
    profiled_bullets: list[str] = field(default_factory=list)
