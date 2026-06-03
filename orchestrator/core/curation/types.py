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
    """Chain-of-thought analysis of step execution.  
  
    Single freetext field, which exists so the LLM reasons analytically  
    before making playbook delta decisions. Not persisted as  
    structured data on the PlanStep.  
    """  
  
    analysis: str = Field(  
        description=(  
            "Before making playbook decisions, analyze the execution:\n"  
            "- What was attempted and what happened? Reference delegation numbers.\n"  
            "- Compare expected vs actual output: this uncovers wrong assumptions and gaps in the orchestrator in its current knowledge about the agents. "  
            "- What was unnecessary, what did the agent figure out alone?\n"  
            "- What does this reveal about the agent's capabilities and needs?\n"  
            "- Was the instruction too vague (agent guessed wrong) or "  
            "too detailed (agent ignored parts)?\n"  
            "Be specific: name commands, tools, flags, formats, behaviors.\n"  
            "This is your reasoning space, so be thorough."  
        ),  
    )

class CuratorOutput(BaseModel):  
    """Combined reflector + curator output in one. 
  
    Field order matters: reflection before playbook_delta forces the LLM  
    to reason analytically before making structural curation decisions.  
    """  
  
    reflection: ReflectionOutput = Field(  
        description=(  
            "Structured analysis of the execution. Reflect on what you observe of the agents' behavior and compare with what we know complete this "  
            "BEFORE the playbook delta."  
        ),  
    )  
    playbook_delta: PlaybookDeltaOutput = Field(  
        default_factory=PlaybookDeltaOutput,  
        description=(  
            "Incremental changes to the agent's playbook, informed "  
            "by your reflection above."  
        ),  
    )  
