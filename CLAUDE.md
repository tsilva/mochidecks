# CLAUDE.md

Instructions for Claude Code when working with this flashcard deck repository for [mochi-mochi](https://github.com/tsilva/mochi-mochi).

## File Format

Deck files: `{topic}-{id}.md`

Card structure:
```markdown
---
card_id: unique_id
---
Question (markdown, LaTeX with $$...$$, code blocks)
---
Answer (markdown, LaTeX with $$...$$, code blocks)
---
```

## Formatting Rules

- **Card IDs** - unique alphanumeric format (e.g., "00TVK0st", "2CskO98D")
  - **IMPORTANT**: When creating NEW cards, ALWAYS use `card_id: null` - never generate or make up card IDs
  - The mochi-mochi application will automatically generate proper IDs when importing
  - Only existing cards should have alphanumeric IDs
  - **When fixing malformed cards**: If you need to WRITE or RECONSTRUCT a question or answer (not just edit existing text), treat it as a NEW card and set `card_id: null`
    - Example: If a card has an answer but no question, and you write a question for it → `card_id: null`
    - Example: If you're just fixing a typo in an existing question → keep the existing card_id
  - **Rule of thumb**: If the card content exists and is readable, keep the ID. If you're creating content that wasn't there before, use `null`.
- **Delimiters** - triple dashes `---` separate card_id, question, and answer
- **Markdown** - GitHub-flavored markdown supported
- **LaTeX** - wrap math in `$$...$$`
- **No tables in answers** - use lists or other formatting instead

## Curation Principles

**Quality standards:**
- **Atomic** - one question, one answer per card; no multi-part questions/answers
- **Concise** - clear, focused questions and answers; use bullet lists instead of paragraphs when answers contain multiple facts
- **Unique** - no duplicate cards within or across decks
- **Unambiguous** - questions must have a single, clear interpretation with no ambiguity about what is being asked

**Removing redundancies:**
- DO remove: identical questions or near-identical content with no learning variation
- DON'T merge: different questions about the same concept ("What is X?" vs. "When to use X?" vs. "Advantages of X?")
- Each distinct question gets its own card, even if related
- When in doubt, keep cards separate
- Delete rather than merge if cards test different things
