# mochi-decks

This repository stores flashcard decks for use with [mochi-mochi](../mochi-mochi/).

## About

This repository contains markdown-formatted flashcard decks covering various topics. Each deck is stored as a separate `.md` file with a naming pattern of `{topic}-{id}.md`.

## Deck Format

Each deck file uses a simple markdown format with cards separated by `---` delimiters:

```markdown
---
card_id: unique_id
---
Question content goes here
---
Answer content goes here
---
```

The format supports:
- **Markdown formatting** for rich text content
- **LaTeX equations** using `$$...$$` syntax
- **Code blocks** and other standard markdown features

## Current Decks

- `aiml-dMZe0dy2.md` - AI/ML concepts and fundamentals

## Usage

These deck files are designed to be imported and used by the mochi-mochi application. Refer to the [mochi-mochi documentation](../mochi-mochi/) for instructions on importing and studying with these decks.

## Contributing

To add a new deck:
1. Create a new `.md` file following the naming convention
2. Format cards using the structure shown above
3. Ensure each card has a unique `card_id`
4. Commit and push to the repository

## License

See [LICENSE](LICENSE) for details.
