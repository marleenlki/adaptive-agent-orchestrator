"""Trajectory tracking: event records and timeline collector.

Records three kinds of events on a single timeline:

- **MessageRecord** — top-level user and orchestrator messages.
- **DelegationExchange** — orchestrator → agent exchanges.
- **ToolRecord** — orchestrator's own tool calls (plan, explore, judge, …).

The :class:`AgentCallTracker` collects these events in-memory and
archives them per thread for later persistence via the trajectory store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from pydantic import BaseModel, Field

from orchestrator.shared.constants import ORCHESTRATOR_DELEGATION_SOURCE


# Record types


@dataclass(frozen=True, slots=True)
class MessageRecord:
    """A top-level user or orchestrator message in the conversation."""

    role: str
    content: str


class DelegationExchange(BaseModel):
    """One orchestrator -> agent delegation, with the evaluator's verdict."""

    step_number: int = Field(description="1-based iteration number.")
    agent: str = Field(description="Agent that was delegated to.")
    instruction: str = Field(description="The instruction sent to the agent.")
    actual_output: str = Field(description="What the agent actually returned.")
    reasoning: str = Field(
        default="",
        description="Why this delegation with this instruction was the best action at the time.",
    )
    success: bool = Field(default=True, description="Evaluator's verdict.")
    feedback: str = Field(default="", description="Evaluator's feedback.")
    cited_bullets: list[str] = Field(
        default_factory=list,
        description="Playbook bullet IDs cited by the executor for this delegation.",
    )
    source: str = Field(
        default=ORCHESTRATOR_DELEGATION_SOURCE,
        description="Origin of the exchange (main orchestrator path vs. subflows).",
    )


@dataclass(frozen=True, slots=True)
class ToolRecord:
    """An internal orchestrator tool call (plan, explore, world-model, …).

    Attributes:
        tool_name: Name of the tool that was called.
        input: The arguments / input passed to the tool.
        output: The tool's return value.
    """

    tool_name: str
    input: str
    output: str


# Union type for everything that can appear on the timeline.
TimelineRecord = Union[MessageRecord, DelegationExchange, ToolRecord]


# Tracker


class AgentCallTracker:
    """Records agent delegations and internal tool calls on a single timeline.

    When ``reset(thread_id)`` is called the current state is archived
    under that *thread_id* so it can be retrieved later.
    """

    def __init__(self) -> None:
        self._records: list[DelegationExchange] = []
        self._timeline: list[TimelineRecord] = []
        self._history: dict[str, list[DelegationExchange]] = {}
        self._timeline_history: dict[str, list[TimelineRecord]] = {}

    # Recording

    def record(
        self,
        agent_name: str,
        message: str,
        response: str,
        source: str = ORCHESTRATOR_DELEGATION_SOURCE,
    ) -> None:
        """Record a delegation exchange.

        Args:
            agent_name: Name of the delegated agent.
            message: The message/task sent to the agent.
            response: The agent's response text.
        """
        rec = DelegationExchange(
            step_number=0,
            agent=agent_name,
            instruction=message,
            actual_output=response,
            source=source,
        )
        # Keep the main orchestrator path in _records for the
        # agent-level views, while the unified timeline captures
        # explorer reconnaissance and other subflows.
        if source == ORCHESTRATOR_DELEGATION_SOURCE:
            self._records.append(rec)
        self._timeline.append(rec)

    def record_message(
        self,
        role: str,
        content: str,
    ) -> None:
        """Record a top-level user or orchestrator message."""
        self._timeline.append(
            MessageRecord(role=role, content=content)
        )

    # Current-task accessors
    def get_timeline(self) -> list[TimelineRecord]:
        """Return a copy of the live timeline for the current task."""
        return list(self._timeline)

    # Archiving

    def reset(self, thread_id: str | None = None) -> None:
        """Archive current records under *thread_id* and clear them.

        For single-turn runs each thread_id is only reset once, so the
        previous implementation's accumulation across multiple resets is
        no longer needed.
        """
        if thread_id is not None:
            self._history[thread_id] = list(self._records)
            self._timeline_history[thread_id] = list(self._timeline)
        self._records.clear()
        self._timeline.clear()

    # History accessors

    def get_agent_calls(self, thread_id: str) -> list[str]:
        """Return agent names called during *thread_id* (used by mabench)."""
        return [r.agent for r in self._history.get(thread_id, [])]

    def get_tool_calls(self, thread_id: str) -> list[ToolRecord]:
        """Return tool-call records for *thread_id* (used by mabench)."""
        return [
            r
            for r in self._timeline_history.get(thread_id, [])
            if isinstance(r, ToolRecord)
        ]
