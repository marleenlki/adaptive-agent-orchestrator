"""Prompt templates for agent profiling — enhancing agent descriptions."""

QUERY_GENERATION_PROMPT = """\
You are analysing a registered agent to test whether an embedding-based \
search would find it for every task it can handle.

## Agent Card
- **Name:** {agent_name}
- **Description:** {agent_description}
- **Skills:** {agent_skills}

## Your Task

Generate exactly {n_queries} realistic user queries — requests that a \
user might submit to an orchestrator when they NEED this agent but do \
NOT know its name.

**Approach — systematic coverage:**
1. Read the description and skills carefully. Identify every distinct \
capability, action, domain, and object type mentioned.
2. For EACH identified aspect, write at least one query. Do not over-\
represent any single aspect at the expense of others.
3. After covering explicit aspects, think about implicit capabilities: \
What tasks would logically require this agent that are NOT directly \
stated in the card? Generate queries for those too.

**Rules:**
1. Each query is a natural user request (1–2 sentences).
2. Use diverse wording: vary verbs, nouns, domains, and phrasing styles. \
Rephrase capabilities using synonyms the card does NOT use.
3. Include edge-case / niche tasks the agent could handle.
4. Mix specificity levels: some queries broad ("manage my schedule"), \
some narrow ("move the 3 PM meeting to Thursday").
5. Do NOT use the agent's name in any query.
6. Do NOT generate multiple queries that test the same capability with \
only minor rephrasing — each query should test a DIFFERENT aspect or \
phrasing angle.
"""

CAPABILITY_BULLET_PROMPT = """\
You are improving an agent's discoverability by writing capability bullets.

## Agent Card
- **Name:** {agent_name}
- **Description:** {agent_description}
- **Skills:** {agent_skills}

## User queries that SHOULD find this agent but currently DON'T
{gap_queries}

## Your Task

Write exactly {n_bullets} capability bullets.  Each bullet describes a \
DIFFERENT capability of this agent, phrased so it matches a distinct \
subset of the failing queries.

**Rules:**
1. Each bullet starts with "Can " and is ≤15 words.
2. Each bullet covers a DIFFERENT aspect — do NOT overlap or repeat.
3. Only rephrase capabilities the agent ALREADY has according to its \
card and skills. Do NOT invent capabilities not supported by the card.
4. Use synonyms and phrasing from the failing queries to maximize \
semantic overlap.
5. Be specific: mention concrete actions, formats, or domains.
"""
