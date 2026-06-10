"""Executor ReAct agent that handles the full orchestration
lifecycle: context retrieval, optional planning, delegation,
adaptation, and completion.
"""

from __future__ import annotations

import logging

from langgraph.errors import GraphRecursionError

from orchestrator.core.execution.graph import build_executor_agent
from orchestrator.prompts.executor import (
    EXECUTOR_PROMPT,
)
from orchestrator.core.session_types import OrchestratorSession
from orchestrator.core.execution.tools.build import build_tools
from orchestrator.shared.constants import RECURSION_LIMIT

logger = logging.getLogger(__name__)


def run_executor(session: OrchestratorSession, task: str) -> str:
    """Build a ReAct agent, invoke it with *task*, return the final answer."""
    ctx = session.ctx
    tools = build_tools(session=session)

    prompt = EXECUTOR_PROMPT.format(task=task)

    agent, config = build_executor_agent(
        llm=ctx.llm,
        tools=tools,
        system_prompt=prompt,
        thread_id=session.thread_id,
        recursion_limit=RECURSION_LIMIT,
    )

    try:
        agent.invoke({"messages": [{"role": "user", "content": task}]}, config=config)
    except GraphRecursionError:
        logger.warning("[executor] Recursion limit hit")
        return session.final_answer or "error: recursion limit exceeded"

    if session.final_answer:
        return session.final_answer

    logger.warning("[executor] Agent exited without task_complete — no final answer")
    return "error: no final answer produced"
