"""Pydantic output schemas for the reflection layer (step + episode curators)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from orchestrator.memory.records import PlaybookDeltaOutput

__all__ = [
    "CuratorOutput",
    "ReflectionOutput",
    "BlueprintStepOutput",
    "SuccessCuratorOutput",
]


class BlueprintStepOutput(BaseModel):
    """LLM output: one step in the ideal delegation chain."""

    agent: str = Field(description="Agent name — MUST be one of the Available Agents listed above.")
    does: str = Field(description="What this agent should do (1 sentence).")
    receives: str = Field(
        description="What input this agent needs and from whom (e.g. 'file paths from shell').",
    )
    produces: str = Field(
        description="What output this agent produces and for whom (e.g. 'raw text → llm').",
    )


class SuccessCuratorOutput(BaseModel):
    """LLM output for successful episodes — blueprint extraction."""

    task: str = Field(
        description=(
            "One-sentence summary of the task. Write it as a reusable, "
            "retrieval-friendly description that would match similar "
            "future tasks."
        ),
    )
    ideal_steps: list[BlueprintStepOutput] = Field(
        description=(
            "The MINIMAL, ORDERED agent chain that achieves the task. "
            "Only use agents from the Available Agents list. "
            "Only include agents whose output was actually consumed. "
            "Describe data flow between steps via receives/produces."
        ),
    )
    rationale: str = Field(
        description="Why this sequence is optimal — 1-2 sentences.",
    )



class ReflectionOutput(BaseModel):
    """Analysis that precedes playbook delta decisions."""

    analysis: str = Field(
        description=(
            "Before making playbook decisions, analyze the execution:\n"
            "- What was attempted and what happened? Reference delegation numbers.\n"
            "- Compare expected vs. actual output.\n"
            "- What was unnecessary? What did the agent figure out alone?\n"
            "- What does this reveal about the agent's capabilities and needs?\n"
            "- Was the instruction too vague or too detailed?\n"
            "Be specific: name commands, tools, flags, formats, and behaviors."
        ),
    )


class CuratorOutput(BaseModel):
    """Combined reflection and playbook update output."""

    reflection: ReflectionOutput = Field(
        description=(
            "Analyze the agent's observed behavior before making any playbook "
            "delta decisions."
        ),
    )
    playbook_delta: PlaybookDeltaOutput = Field(
        default_factory=PlaybookDeltaOutput,
        description=(
            "Incremental playbook changes grounded in the reflection above. "
            "Return empty lists when nothing transferable was learned."
        ),
    )
