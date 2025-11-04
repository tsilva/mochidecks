# mochi-decks

Markdown-formatted flashcard decks for use with [mochi-mochi](https://github.com/tsilva/mochi-mochi). Each deck is stored as `{topic}-{id}.md`.

## Deck Format

Cards are separated by `---` delimiters with YAML frontmatter for metadata:

```markdown
---
card_id: unique_id
tags: ["tag1", "tag2"]  # optional
archived: false          # optional
---
Question content
---
Answer content
---
```

**Content features:**
- Standard markdown formatting
- LaTeX equations with `$$...$$`
- Code blocks with syntax highlighting

**Card ID handling:**
- Existing cards: Keep the original Mochi ID
- New cards: Use `card_id: null` to create new cards on push

## Current Decks

- `aiml-dMZe0dy2.md` - AI/ML concepts and fundamentals

## Workflow

1. **Pull**: Download decks from Mochi using `mochi-mochi pull <deck_id>`
2. **Edit**: Modify markdown files locally
3. **Commit**: Track changes with Git
4. **Push**: Sync back to Mochi using `mochi-mochi push <filename>`

See [mochi-mochi documentation](https://github.com/tsilva/mochi-mochi) for installation and detailed usage.
