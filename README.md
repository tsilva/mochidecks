# mochi-decks

Markdown-formatted flashcard decks for use with [mochi-mochi](../mochi-mochi/). Each deck is stored as `{topic}-{id}.md`.

## Deck Format

Cards are separated by `---` delimiters with frontmatter for metadata:

```markdown
---
card_id: unique_id
---
Question content
---
Answer content
---
```

Supports standard markdown, LaTeX equations (`$$...$$`), and code blocks.

## Current Decks

- `aiml-dMZe0dy2.md` - AI/ML concepts and fundamentals

## Usage

Managed via [mochi-mochi](../mochi-mochi/) CLI. See the tool's documentation for sync commands.
