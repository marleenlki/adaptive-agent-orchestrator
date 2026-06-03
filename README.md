# Adaptive Agent Orchestrator

Adaptive Agent Orchestrator is a Python package for coordinating a pool of LangChain-compatible specialist agents. It provides an `AdaptiveOrchestrator` that can discover agents from a registry, delegate sub-tasks, optionally verify final answers with a judge model, and persist cross-episode memory in PostgreSQL/pgvector.

Use this package when you already have multiple agent chains or tools and want one orchestration layer that decides which agents to call, keeps track of what happened, and can learn reusable coordination patterns across runs.

## Architecture Overview

The orchestrator is centered around four runtime objects:

- `AgentRegistry`: holds the available specialist agents and their Agent Cards.
- `AdaptiveOrchestrator`: owns the execution loop and decides when to retrieve context, plan, delegate, and finish.
- Memory stores: optional PostgreSQL-backed stores for episodes, playbooks, blueprints, and trajectories.
- Model handles: `llm` for execution, `curator_llm` for memory updates, and `judge_llm` for answer verification.

```text
Agent chains
   |
   v
AgentRegistry
   |
   v
AdaptiveOrchestrator
   |
   +-- gather_context
   |      +-- Agent Card retrieval
   |      +-- playbook retrieval
   |      +-- blueprint retrieval
   |
   +-- create/update plan
   |
   +-- delegate_to_agent
   |      +-- invokes registered agent chains
   |
   +-- task_complete
          +-- optional judge review
          +-- memory curation in train mode
```

Without memory stores, the orchestrator can still run as a stateless coordinator over all registered agents. With memory stores and an embedder, it can retrieve relevant agents and blueprints, record trajectories, and update cross-episode memory.

## What You Need

To run the orchestrator, you need:

- a main chat model for the orchestration loop;
- an `AgentRegistry` with at least one registered agent;
- each registered agent must expose a LangChain-style `.invoke(...)` method;
- optionally, PostgreSQL with pgvector if you want adaptive memory;
- optionally, an embedding model if you want semantic agent and memory retrieval.

The simplest setup does not require a database. In that mode, disable agent filtering so the orchestrator can see all registered agents without embeddings.

## Install

```bash
pip install -e .
```

Or:

```bash
uv sync
```

Python `>=3.10` is required.

## Quick Start Without Memory

This is the easiest way to use the orchestrator. It registers agents, gives the orchestrator access to all of them, and disables memory-dependent features.

```python
from langchain_openai import ChatOpenAI

from orchestrator import AdaptiveOrchestrator
from orchestrator.registration.registry import AgentRegistry


registry = AgentRegistry()

registry.register(
    name="spreadsheet_agent",
    description="Reads and edits spreadsheet files.",
    chain=spreadsheet_chain,
)

registry.register(
    name="email_agent",
    description="Drafts and sends email messages.",
    chain=email_chain,
)

llm = ChatOpenAI(model="gpt-4.1-mini")

orch = AdaptiveOrchestrator(
    llm=llm,
    registry=registry,
    enable_agent_filtering=False,
    enable_playbooks=False,
    enable_blueprints=False,
    enable_trajectory=False,
    enable_judge=False,
    memory_mode="test",
)

answer = orch.solve("Summarize the spreadsheet and draft an email with the key points.")
```

Important: if you do not pass memory stores with an embedder, keep `enable_agent_filtering=False`. Otherwise semantic retrieval has no embedding backend and no agents may be surfaced.

## Register Agents

Agents are registered through `AgentRegistry.register(...)`.

```python
registry.register(
    name="pdf_agent",
    description="Extracts and summarizes content from PDF files.",
    chain=pdf_chain,
    skills=[],
)
```

The `chain` is invoked through:

```python
chain.invoke(
    {"messages": [{"role": "user", "content": instruction}]},
    config={"configurable": {"thread_id": thread_id}},
)
```

The response can be a plain string, a LangChain message, a dict with `messages`, or a dict with `structured_response.final_answer`. The client normalizes these outputs into a string.

Optional registration fields:

- `skills`: A2A-style `AgentSkill` metadata.
- `card_embedding`: precomputed embedding of the Agent Card text.
- `profiled_bullets`: precomputed capability bullets that can be seeded into the playbook store.

## Use Adaptive Memory

Adaptive memory requires PostgreSQL with pgvector and an embedder. The included `docker-compose.yml` starts a local database and mounts the packaged schema.

```bash
docker compose up -d postgres
```

The default local connection string is:

```text
postgresql://orchestrator:orchestrator@localhost:5433/orchestrator
```

Create the stores:

```python
from orchestrator.memory.factory import create_pg_stores
from orchestrator.shared.embedder import Embedder


embedder = Embedder()
conninfo = "postgresql://orchestrator:orchestrator@localhost:5433/orchestrator"

episode_store, blueprint_store, playbook_store, trajectory_store = create_pg_stores(
    conninfo,
    embedder=embedder,
)
```

Then pass the stores into the orchestrator:

```python
orch = AdaptiveOrchestrator(
    llm=llm,
    registry=registry,
    episode_store=episode_store,
    blueprint_store=blueprint_store,
    playbook_store=playbook_store,
    trajectory_store=trajectory_store,
    curator_llm=curator_llm,
    judge_llm=judge_llm,
    memory_mode="train",
)
```

In memory mode, the orchestrator can:

- retrieve relevant agents by Agent Card embeddings;
- retrieve agent playbooks;
- retrieve delegation blueprints from previous episodes;
- save execution trajectories;
- update playbooks and blueprints after an episode when `memory_mode="train"`.

## Environment Variables for Embeddings

`Embedder` uses Azure OpenAI by default.

```bash
export AZURE_OPENAI_ENDPOINT="https://..."
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_API_VERSION="2024-06-01"
```

You can also pass these values directly:

```python
embedder = Embedder(
    azure_endpoint="https://...",
    api_key="...",
    api_version="2024-06-01",
)
```

## Model Roles

`AdaptiveOrchestrator` accepts up to three model handles:

```python
AdaptiveOrchestrator(
    llm=main_llm,
    curator_llm=curator_llm,
    judge_llm=judge_llm,
    registry=registry,
)
```

- `llm`: runs the main ReAct orchestration loop.
- `curator_llm`: extracts playbook and blueprint updates after an episode.
- `judge_llm`: verifies candidate final answers.


## Configuration Flags

Useful constructor options:

| Option | Meaning |
| --- | --- |
| `memory_mode="train"` | Enables memory writes during finalization. |
| `memory_mode="test"` | Keeps memory read-only. |
| `enable_playbooks` | Enables playbook retrieval and curation. |
| `enable_blueprints` | Enables blueprint retrieval and curation. |
| `enable_trajectory` | Saves execution trajectories when a trajectory store is provided. |
| `enable_judge` | Enables final-answer verification. |
| `enable_agent_filtering` | Uses embedding/playbook retrieval to select agents. Disable this if you have no embedder. |
| `enable_planning` | Enables explicit planning tools in the execution loop. |
| `max_judge_rejections` | Number of rejected submissions before force-accepting. |

Common setups:

```python
# Stateless baseline: no memory, no judge, all agents visible.
AdaptiveOrchestrator(
    llm=llm,
    registry=registry,
    enable_agent_filtering=False,
    enable_playbooks=False,
    enable_blueprints=False,
    enable_trajectory=False,
    enable_judge=False,
    memory_mode="test",
)
```

```python
# Train adaptive memory.
AdaptiveOrchestrator(
    llm=llm,
    curator_llm=curator_llm,
    judge_llm=judge_llm,
    registry=registry,
    episode_store=episode_store,
    blueprint_store=blueprint_store,
    playbook_store=playbook_store,
    trajectory_store=trajectory_store,
    enable_judge=True,
    memory_mode="train",
)
```

```python
# Evaluate with frozen memory.
AdaptiveOrchestrator(
    llm=llm,
    registry=registry,
    episode_store=episode_store,
    blueprint_store=blueprint_store,
    playbook_store=playbook_store,
    trajectory_store=trajectory_store,
    enable_judge=False,
    memory_mode="test",
)
```

## Agent Profiling

If a playbook store is available, you can profile registered agents before running tasks:

```python
profiled = orch.profile_agents()
```

Profiling asks the curator model to generate capability bullets for agents whose Agent Cards do not retrieve well for plausible capability queries. These bullets are stored as unconfirmed playbook entries and can later be confirmed or contradicted by real execution episodes.

If you already have offline capability bullets, pass them at registration time:

```python
registry.register(
    name="pdf_agent",
    description="Works with PDF documents.",
    chain=pdf_chain,
    profiled_bullets=[
        "Can extract text from PDF files.",
        "Can summarize document contents.",
    ],
)

orch.seed_profiled_bullets()
```

## Runtime Outputs

After `solve(...)`, useful runtime information is available on the orchestrator:

```python
answer = orch.solve(task, thread_id="episode-001")
metrics = orch.last_metrics
metadata = orch.get_run_metadata()
```

If `trajectory_store` is enabled, the execution timeline is persisted to PostgreSQL. The timeline contains user/orchestrator messages, delegation records, tool calls, and judge records.

## Memory Schema

The schema lives at:

```text
orchestrator/memory/setup/schema.sql
```

To initialize a custom schema:

```bash
python -m orchestrator.memory.setup \
  --conninfo "postgresql://orchestrator:orchestrator@localhost:5433/orchestrator" \
  --schema my_schema
```

To create stores against that schema:

```python
stores = create_pg_stores(conninfo, embedder=embedder, schema="my_schema")
```

## Package Structure

- `orchestrator/adaptive_orchestrator.py`: public orchestrator entry point.
- `orchestrator/registration`: Agent Cards, registry, and local agent client wrapper.
- `orchestrator/core/execution`: LangGraph execution loop and tools.
- `orchestrator/core/retrieval`: agent, playbook, and blueprint retrieval.
- `orchestrator/core/curation`: post-episode memory updates.
- `orchestrator/memory`: PostgreSQL-backed memory stores.
- `orchestrator/instrumentation`: metrics and trajectory recording.

## Notes

This package contains the orchestrator implementation, not the OfficeBench benchmark harness. Agents are invoked in-process through local LangChain-compatible chains. The registry and Agent Card abstraction are A2A-inspired, but this package does not expose an HTTP A2A server. The adaptive memory backend is PostgreSQL with pgvector.
