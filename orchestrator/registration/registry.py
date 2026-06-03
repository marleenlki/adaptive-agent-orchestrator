"""Agent registry as central management of all known agents."""

from __future__ import annotations

import logging

from a2a.types import AgentSkill

from orchestrator.registration.client import AgentClient

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Registry for agents as central management of all known agents."""

    def __init__(self):
        self._agents: dict[str, AgentClient] = {}

    def register(
        self,
        name: str,
        description: str,
        chain,
        skills: list[AgentSkill] | None = None,
        card_embedding: list[float] | None = None,
        profiled_bullets: list[str] | None = None,
    ) -> dict[str, str]:
        """Register an agent with name, description, and chain.

        Args:
            name: The name of the agent.
            description: The description of the agent.
            chain: The LangChain chain (runnable).
            skills: Optional list of agent skills.
            card_embedding: Pre-computed embedding of the card text.
            profiled_bullets: Pre-computed capability bullets.

        Returns:
            A dict with the name and description of the registered agent.
        """
        connection = AgentClient.create(
            name=name,
            description=description,
            chain=chain,
            skills=skills,
            card_embedding=card_embedding,
            profiled_bullets=profiled_bullets,
        )
        self._agents[name] = connection
        logger.info("Agent '%s' registered", name)

        return {
            "name": name,
            "description": description,
        }

    def list_agents(self) -> list[dict]:
        """Lists all registered agents with name, description, skills, and card_embedding."""
        result = []
        for connection in self._agents.values():
            card = connection.card
            skills = []
            if card.skills:
                skills = [
                    {"name": skill.name, "description": skill.description or ""}
                    for skill in card.skills
                ]
            entry: dict = {
                "name": card.name,
                "description": card.description or "",
                "skills": skills,
            }
            if card.card_embedding:
                entry["card_embedding"] = card.card_embedding
            result.append(entry)
        return result

    def get_connection(self, name: str) -> AgentClient | None:
        """Returns the connection to an agent"""
        return self._agents.get(name)

    def agents_with_profiled_bullets(self) -> dict[str, list[str]]:
        """Return {agent_name: bullets} for agents that have pre-computed bullets."""
        return {
            name: conn.card.profiled_bullets
            for name, conn in self._agents.items()
            if conn.card.profiled_bullets
        }

    @property
    def agent_names(self) -> list[str]:
        """All registered agent names."""
        return list(self._agents.keys())
