"""Prompt template for ACE-style playbook bullet consolidation."""

PLAYBOOK_MERGE_PROMPT = """\
You are consolidating similar playbook bullets for the '{agent}' agent.

These bullets are in the '{section}' section and were identified as \
semantically overlapping. Merge them into ONE comprehensive bullet that:
1. Captures ALL actionable information from every source bullet
2. Removes redundancy and repetition
3. Preserves specific details (commands, flags, formats, tools)
4. Follows the section format convention and word limit

## Source bullets to merge
{bullets_text}

Produce ONE merged rule. Do NOT add information not present in the sources.
"""
