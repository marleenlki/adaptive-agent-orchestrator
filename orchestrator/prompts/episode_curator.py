"""Prompt templates for the episode-level curator."""

_EPISODE_CONTEXT_BLOCK = """\
## Episode Summary
Task: {task}
Outcome: succeeded
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
decision-making process.
"""

EPISODE_CURATOR_SUCCESS_PROMPT = """\
You are the memory curator for a multi-agent orchestration system. \
You analyze completed episodes to extract reusable coordination \
knowledge.

This episode succeeded. Your job is to produce a reusable blueprint: the ideal \ 
delegation chain for solving this kind of task in the future.

""" + _EPISODE_CONTEXT_BLOCK + """\
---

""" + _TIMELINE_GUIDE + """

## How to Analyze and Curate a Successful Episode

### 1. Determine Whether a Retrieved Blueprint Was Used
First check whether the `Retrieved Blueprint` section contains an actual prior 
blueprint. If a retrieved blueprint was provided, compare it against the execution timeline. If no retrieved blueprint was provided, analyze the execution on its own merits \ 
and derive the ideal blueprint from scratch.


### 2. Analyze the Execution
When a retrieved blueprint was provided, evaluate:
- Where did execution follow the blueprint? Was it efficient?
- Where did execution deviate? Was each deviation beneficial or \
wasteful? Where all agents actually needed, or were some redundant?
- Identify unnecessary retries, unclear handoffs, redundant \
delegations, missing checks, or inefficient sequencing.

When no retrieved blueprint was provided, focus on: 
- which delegations were necessary, 
- which delegations were avoidable, 
- which order of agents/tools would have been more efficient, 
- which checks should have happened earlier.

### 3. Derive the Ideal Blueprint 
Describe the ideal execution chain for this task type. It should use the right \
agents in the right order, include clear handoffs, avoid redundant work, and add \
validation steps where they prevent known failures. 

### 4. Decide Whether to Refine 
Set `refines_retrieved = true` only if a retrieved blueprint was provided and \
the ideal chain is an improved version of that same task-type blueprint. 
Set `refines_retrieved = false` if no blueprint was retrieved, the ideal chain \
solves a different task type, or replacing the retrieved blueprint would be too \
broad or misleading.

HARD RULE: This system is single-turn and there is NO user in the loop. \
Never mention "the user", "ask the user", "prompt the user", or \
"clarify with the user" in blueprints or anywhere else. \
The executor receives a task and must solve it autonomously.
"""
