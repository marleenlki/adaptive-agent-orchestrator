"""Prompt template for ACE-style playbook bullet consolidation."""

PLAYBOOK_MERGE_PROMPT = """\
You are consolidating similar playbook bullets for the "{agent}" agent.

These bullets are in the "{section}" section and were identified as \
semantically overlapping.

Merge them into ONE comprehensive bullet that:
1. captures ALL actionable information from every source bullet
2. removes redundancy and repetition
3. preserves specific details (commands, flags, formats, tools)
4. follows the section format convention and word limit

## Source bullets to merge
{bullets_text}

Produce ONE merged rule.
Do NOT add information that is not present in the sources.
"""
