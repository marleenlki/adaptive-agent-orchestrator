# Orchestrator - Multi-Agent Coordination Module

## Overview

The **Orchestrator** is a LangChain-based coordinator that delegates a user task to multiple specialized agents and composes a final response.

This repository is designed for **benchmarking**. Therefore, it intentionally **skips HTTP/A2A server overhead** and instead implements the relevant A2A concepts **in-process**:

- **Agent Card**: agent metadata (name, description, skills)
- **Agent Registry**: agent discovery
- **Agent Client (Adapter)**: a transport wrapper that *looks like* a remote agent client, but calls a local LangChain chain directly

> In a production A2A setup, `AgentConnection` would typically call an agent via HTTP (A2A endpoints).  
> In this benchmark setup, it calls `chain.ainvoke()` directly - same semantics, no network cost.

---

## How It Works

```
User Task
    ↓
Orchestrator Agent
    ↓
┌─────────────────────────────────┐
│  list_agents                    │  ← Discover available agents
|      ↓
│  delegate_to_agent(name, task)  │  ← Send work to agents
└─────────────────────────────────┘
   ↓ ↓ ↓
┌──────────┬──────────┬──────────┐
│  Agent 1 │  Agent 2 │  Agent 3 │  ← Specialized workers
└──────────┴──────────┴──────────┘
     ↓
Consolidated Response formed by Orchestrator
     ↓
Final Answer to User
```

The Orchestrator uses two tools:
1. **`list_agents()`** - Shows what agents are available and their skills
2. **`delegate_to_agent(name, message)`** - Sends a task to a specific agent and gets the result

---

## Core Concepts (A2A-inspired)

### Agent Card
Each agent is described via an **Agent Card**:
- `name`
- `description`
- `skills` (e.g., `AgentSkill`)

This mirrors the idea of “agent metadata” used for discovery and routing.

### Agent Registry
All agents are registered in a central **Agent Registry**.  
The orchestrator can query the registry to see which agents exist and what they can do.

### Agent Client (Transport Adapter)
`AgentClient` is a small wrapper around a single agent implementation.

- **Production/A2A world**: would send messages to a remote agent endpoint (HTTP).
- **Benchmark world (this repo)**: directly calls a local LangChain chain via `chain.ainvoke()`.

This keeps the orchestrator logic identical while eliminating network/protocol overhead.

---

## Files

### `orchestrator.py` - Main Orchestrator Class
- **What it does:** Creates and runs the orchestrator agent
- **Key method:** `Orchestrator.create(llm)` - Creates a new orchestrator with an LLM
- **How to use:**
  ```python
  orch = Orchestrator.create(llm=my_llm)
  orch.registry.register(name="Calendar", description="...", chain=calendar_chain)
  response = await orch.run("Book a meeting for tomorrow")
  ```


### `agent_registry.py` - Agent Registry
- **What it does:** Manages all available agents
- **Key method:** `registry.register(name, description, chain, skills)` - Add an agent
- **Key method:** `registry.list_agents()` - See all registered agents


### `agent_client.py` - Connection to a Single Agent
- **What it does:** Wraps a LangChain chain so it can be called as an agent
- **Key class:** `AgentClient` - A safe wrapper around a chain
- **How it works:** When the orchestrator sends a message, this class calls `chain.ainvoke()` directly

### `orchestrator_tools.py` - Tools for the Orchestrator
- **What it does:** Defines the two tools the orchestrator can use
- **Tool 1:** `list_agents()` - Orchestrator learns what agents exist
- **Tool 2:** `delegate_to_agent(agent_name, message)` - Orchestrator sends work to agents
- **Purpose:** These tools are added to the orchestrator's LangChain agent, so it can make decisions

### `__init__.py`
- Exports `Orchestrator` class for easy importing

---

## Typical Usage Flow

```python
from orchestrator import Orchestrator
from langchain_openai import ChatOpenAI

# 1. Create orchestrator
llm = ChatOpenAI(model="gpt-4")
orch = Orchestrator.create(llm=llm)

# 2. Register agents (done in PolicyEndpoint for benchmarking)
orch.registry.register(
    name="Calendar Agent",
    description="Manages calendar and meetings",
    chain=calendar_chain,  # Your LangChain chain
)

orch.registry.register(
    name="Email Agent",
    description="Sends and reads emails",
    chain=email_chain,
)

# 3. Execute a task
response = await orch.run("Book a 1-hour meeting tomorrow at 2 PM and email attendees")

# 4. Orchestrator does:
#    - Calls list_agents() to see available agents
#    - Decides to use Calendar Agent + Email Agent
#    - Delegates subtasks to each
#    - Combines results and returns final answer
```

---

## Integration with Benchmarking

In `benchmark/policy_endpoint.py`:
1. Creates all agent chains from `AppAgent`
2. Creates orchestrator with LLM
3. **Registers each chain** as an agent in the orchestrator
4. Runs orchestrator on the task
5. Orchestrator automatically coordinates the agents

---

## Package Installation

This repository is now configured as an installable Python package via `pyproject.toml`.

Install in editable mode:

```bash
pip install -e .
```

Build distribution artifacts:

```bash
python3 -m pip install build
python3 -m build
```
