# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a content repository that stores flashcard decks for use with the [mochi-mochi](https://github.com/tsilva/mochi-mochi) application. It contains markdown-formatted flashcard decks, not application code.

## Deck File Format

Each deck is stored as a `.md` file with naming pattern: `{topic}-{id}.md`

Card structure:
```markdown
---
card_id: unique_id
---
Question content (supports markdown, LaTeX with $$...$$, code blocks)
---
Answer content (supports markdown, LaTeX with $$...$$, code blocks)
---
```

## Deck Curation Principles

When curating deck files, maintain these quality standards:

- **No duplicates** - each card should be unique within and across decks
- **Minimize redundancies** - avoid repetitive cards unless they serve a learning purpose (e.g., asking the same concept in different ways)
- **Concise questions** - questions should be clear and to the point
- **Concise answers** - answers should be focused and avoid unnecessary elaboration
- **Atomic cards** - each card should focus on a single question and single answer; avoid multi-part questions or answers when possible

## Key Guidelines

- **Card IDs must be unique** within and across decks - use the existing alphanumeric format (e.g., "00TVK0st", "2CskO98D")
- **Preserve markdown formatting** - cards use GitHub-flavored markdown
- **LaTeX math** is wrapped in `$$...$$` for equations
- **Triple dashes `---`** are structural delimiters between card_id, question, and answer sections
- When adding cards, follow the existing formatting patterns precisely
- The repository is designed for content management, not software development - there are no build, test, or lint commands

## Relationship to mochi-mochi

This repository is a companion to the [mochi-mochi](https://github.com/tsilva/mochi-mochi) application. The deck files here are imported and consumed by that application for flashcard study sessions.
