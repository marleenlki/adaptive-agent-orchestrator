"""Agent client to connect to Agents"""

from __future__ import annotations

import logging
from typing import Any

from a2a.types import AgentSkill

from orchestrator.registration.card import AgentCard

logger = logging.getLogger(__name__)


def extract_response(result: Any) -> str:
    """Extract text from a result."""
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
    """Client to connect to an agent via its LangChain chain.

    Invokes the chain directly.

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
        """Creates an agent connection.

        Args:
            name: The name of the agent.
            description: The description of the agent.
            chain: The LangChain chain/agent (runnable).
            skills: Optional list of agent skills.
            card_embedding: Pre-computed card embedding vector.
            profiled_bullets: Pre-computed capability bullets.

        Returns:
            An initialized AgentClient.
        """
        card = AgentCard(
            name=name,
            description=description,
            skills=skills or [],
            card_embedding=card_embedding or [],
            profiled_bullets=profiled_bullets or [],
        )
        logger.info("Agent '%s' created", name)
        return cls(card=card, chain=chain)


    def send_message(
        self,
        text: str,
        thread_id: str | None = None,
    ) -> str:
        """Send a message to the agent.

        Args:
            text: The message text for the agent.
            thread_id: Optional thread ID for conversation memory.

        Returns:
            The agent's response as a string.
        """
        invoke_input = {"messages": [{"role": "user", "content": text}]}
        config = {"configurable": {"thread_id": thread_id}} if thread_id else {}

        result = self._chain.invoke(invoke_input, config=config)
        return extract_response(result)

