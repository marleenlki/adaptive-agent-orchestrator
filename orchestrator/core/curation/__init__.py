"""Post-episode curation: turn a finished episode into memory writes.

Modules:
    pipeline            — finalize_episode: orchestrates the steps below.
    agent_curation      — per-agent playbook curation, playbook consolidation.
    blueprint_curation  — LLM creation of a delegation blueprint.
    episode_storage     — persistence of episode memory and blueprints.
    types               — curator output schemas.
"""
