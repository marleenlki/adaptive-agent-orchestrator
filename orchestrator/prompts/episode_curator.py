"""Prompt templates for the episode-level curator."""

_EPISODE_CONTEXT_BLOCK = """\
## Episode Summary
Task: {task}
Total delegations: {total_delegations}

## Available Agents
{available_agents}

## Retrieved Blueprint
{retrieved_blueprint}

## Execution Timeline
{execution_timeline}

## Judge Rejections
Total rejections before acceptance: {judge_rejections}
"""

_TIMELINE_GUIDE = """\
## Understanding the Timeline

The Execution Timeline shows every action in chronological order:
- `⚙ tool(...)` — internal orchestrator tool calls (gather_context, \
create_plan, update_step, view_plan, task_complete, judge_review). \
These show HOW the executor organized its work.
- `[N] ✓/✗ agent=... | instruction` — delegations to specialist \
agents. These show WHAT work was done and whether it succeeded.
- `## Judge Feedback` — shows every judge verdict (ACCEPTED or \
REJECTED) with the judge's feedback and full requirement \
analysis (reasoning). Pay close attention to rejections: the judge's \
reasoning reveals exactly which requirements were missed and why.

Use both tool calls AND delegations to understand the executor's \
decision-making process — not just the final answer.
"""

EPISODE_CURATOR_SUCCESS_PROMPT = """\
You are the memory curator for a multi-agent orchestration system. \
You analyze completed episodes to extract reusable coordination \
knowledge.

This episode SUCCEEDED. Your job:

**Blueprint** — Describe the IDEAL delegation chain: correct \
order, explicit data flow between steps, no redundant retries. \
This is the clean version — what the executor SHOULD do next time.

""" + _EPISODE_CONTEXT_BLOCK + """\
---

""" + _TIMELINE_GUIDE + """

## How to Analyze

**Blueprint analysis:**
If a Retrieved Blueprint was shown to the executor, compare it \
against the Execution Timeline:
- Where did execution follow the blueprint? Was it efficient?
- Where did execution deviate? Was each deviation beneficial or \
wasteful?
- What would the IDEAL execution look like — fewer steps, clearer \
handoffs, no retries?

If no blueprint was retrieved, analyze the execution on its own \
merits.

HARD RULE: This system is single-turn — there is NO user in the loop. \
Never mention "the user", "ask the user", "prompt the user", or \
"clarify with the user" in blueprints or anywhere else. \
The executor receives a task and must solve it autonomously.
"""
