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

## Deck Glossary

### AI/ML
- **deck-aiml-dMZe0dy2.md** - Core AI/ML concepts including bias-variance tradeoff, correlation analysis, statistical measures, and sampling techniques
- **deck-aiml-fundamentals-kflWEgVw.md** - Foundational AI/ML concepts covering statistical analysis, model complexity, and data sampling methods
- **deck-aiml-neural-networks-gkIM7hjD.md** - Neural network concepts including regularization (L1/L2), CNNs, batch normalization, activation functions, and translation invariance

### Data Science
- **deck-numpy-mbmeRMnD.md** - NumPy operations covering broadcasting rules, array indexing, array creation functions, and common operations

### Mathematics
- **deck-linear-algebra-4x2I7duf.md** - Linear algebra fundamentals including vectors, matrices, dot products, transformations, and ML applications

## Workflow

1. **Pull**: Download decks from Mochi using `mochi-mochi pull <deck_id>`
2. **Edit**: Modify markdown files locally
3. **Commit**: Track changes with Git
4. **Push**: Sync back to Mochi using `mochi-mochi push <filename>`

See [mochi-mochi documentation](https://github.com/tsilva/mochi-mochi) for installation and detailed usage.
