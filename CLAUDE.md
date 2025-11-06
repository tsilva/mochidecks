# CLAUDE.md

Instructions for Claude Code when working with flashcard decks for [mochi-mochi](https://github.com/tsilva/mochi-mochi).

## File Format

**Deck naming:**
- Existing decks: `deck-{topic}-{id}.md` (e.g., `deck-aiml-dMZe0dy2.md`)
- New decks: `deck-{topic}.md` (NO ID suffix, e.g., `deck-aiml-fundamentals.md`)

**Card structure:**
```markdown
---
card_id: unique_id
---
Question (markdown, LaTeX with $$...$$, code blocks)
---
Answer (markdown, LaTeX with $$...$$, code blocks)
---
card_id: next_id
---
Question
---
Answer
---
```

**CRITICAL Rules:**
- First line: `---`, Last line: `---`
- NO headers at top - start with first card
- NO blank lines between cards
- NO extra `---` between cards (after answer `---`, next line is `card_id:`)
- Math: `$$...$$` for LaTeX
- No tables in answers (use lists instead)

**Card IDs:**
- NEW decks or SPLIT decks: `card_id: null` (never generate IDs yourself)
- EXISTING cards being edited: Keep original alphanumeric ID (e.g., "00TVK0st")
- Content rewrites (new Q/A): `card_id: null`
- Typo fixes only: Keep existing ID

## Card Design

**Five question types** (use multiple per concept):
1. **DEFINITION** - "What is X?" → One sentence answer
2. **RECOGNITION** - "What does X do/measure/solve?" → Purpose/function
3. **APPLICATION** - "When/why use X?" → Conditions/scenarios
4. **MECHANISM** - "How does X work?" → Internal process
5. **COMPARISON** - "How does X differ from Y?" → Contrasts

**Cards per concept:**
- Foundation concepts: 4-6 cards (bias-variance, regularization, precision/recall)
- Supporting concepts: 2-3 cards (activation functions, pooling types)
- Reference: 1 card (terminology, abbreviations)

**Scenario questions** (high value - create these):
- "Model: 5% train error, 40% validation error. Problem?" → Overfitting
- "99% negative class. Model predicts all negative, 99% accuracy. Problem?" → Imbalanced data
- "Loss → NaN during training. Solutions?" → Learning rate/gradient issues

## Question Rules

- Front-load key term: "What is **dropout**?" not "What is the technique called dropout?"
- Bold tested concept: "How does **batch normalization** reduce training time?"
- Be specific: "What are advantages of ReLU?" not "What about ReLU?"
- Prefer "What" over "Explain"

## Answer Patterns

**Definitions:** One sentence, 10-20 words
- *Example: "Measures the fraction of predicted positives that are actually positive."*

**Formulas:** Formula + symbol definitions only
- *Example: `$$\text{Precision} = \frac{TP}{TP + FP}$$` with `$TP$: true positives`, `$FP$: false positives`*

**Characteristics:** 3-5 bullets, parallel structure
- *Example: "Produces sparse models", "Performs automatic feature selection"*

**Procedures:** 3-5 numbered steps, start with verb
- *Example: "1. Split data into k folds 2. Train on k-1 folds..."*

**Examples:** 1-2 instances, no explanation
- *Example: "Income, house prices, response times"*

**Comparisons:** Parallel structure
- *Example: "**Bagging**: Models in parallel, reduces variance / **Boosting**: Models sequential, reduces bias+variance"*

**Conditions:** Bulleted
- *Example: "Choose L1 when: Need feature selection, Many irrelevant features"*

## Atomicity

**One question, one answer.** If answer has multiple parts, split into separate cards:
- "What is X?" + examples → 2 cards: definition + examples
- "What is X?" + when to use → 2 cards: definition + application
- Use the five question types to enforce atomicity

## Duplicates

- **Remove:** Identical questions/answers with no learning variation
- **Keep:** Different questions about same concept ("What is X?" vs "When use X?")
