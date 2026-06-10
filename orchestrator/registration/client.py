"""Client wrapper around a registered agent's LangChain chain."""

from __future__ import annotations

import logging
from typing import Any

from a2a.types import AgentSkill

from orchestrator.registration.card import AgentCard

logger = logging.getLogger(__name__)


def extract_response(result: Any) -> str:
    """Extract the response text from whatever a chain invocation returned."""
    if isinstance(result, dict) and "structured_response" in result:
        structured = result["structured_response"]
        if hasattr(structured, "final_answer"):
            return str(structured.final_answer)

    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        if messages and hasattr(messages[-1], "content"):
            return str(messages[-1].content)

    if isinstance(result, str):
        return result

    if hasattr(result, "content"):
        return str(result.content)

    return str(result)


class AgentClient:
    """Connects to an agent by invoking its LangChain chain directly.

    Attributes:
        card: Agent metadata (name, description, skills).
    """

    def __init__(self, card: AgentCard, chain):
        self.card = card
        self._chain = chain

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        chain,
        skills: list[AgentSkill] | None = None,
        card_embedding: list[float] | None = None,
        profiled_bullets: list[str] | None = None,
    ) -> AgentClient:
        """Build a client with an agent card from registration data."""
        card = AgentCard(
            name=name,
            description=description,
            skills=skills or [],
            card_embedding=card_embedding or [],
            profiled_bullets=profiled_bullets or [],
        )
        logger.info("Agent '%s' created", name)
        return cls(card=card, chain=chain)

    def send_message(self, text: str, thread_id: str | None = None) -> str:
        """Send a message to the agent and return its response text.

        ``thread_id`` enables the agent's own conversation memory.
        """
        invoke_input = {"messages": [{"role": "user", "content": text}]}
        config = {"configurable": {"thread_id": thread_id}} if thread_id else {}

        result = self._chain.invoke(invoke_input, config=config)
        return extract_response(result)

