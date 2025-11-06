# CLAUDE.md

Instructions for Claude Code when working with flashcard decks for [mochi-mochi](https://github.com/tsilva/mochi-mochi).

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

**Card IDs:**
- NEW cards: `card_id: null` (never generate IDs)
- EXISTING cards: Keep alphanumeric ID (e.g., "00TVK0st")
- RECONSTRUCTED content: `card_id: null` (new question/answer)
- TYPO fixes: Keep existing ID

**Formatting:**
- Delimiters: `---` separates card_id, question, answer
- Math: `$$...$$` for LaTeX
- No tables in answers (use lists)

## Card Design Framework

### Five Question Types

Create cards across these dimensions for important concepts:

1. **DEFINITION** - "What is X?" (one sentence)
2. **RECOGNITION** - "What does X do/measure/solve?" (purpose/function)
3. **APPLICATION** - "When/why use X?" (conditions/scenarios)
4. **MECHANISM** - "How does X work?" (internal process)
5. **COMPARISON** - "How does X differ from Y?" (contrasts)

### Cards Per Concept

- **Foundation** (4-6 cards): bias-variance, regularization, precision/recall, gradient descent
- **Supporting** (2-3 cards): activation functions, pooling types, specific metrics
- **Reference** (1 card): terminology, abbreviations

### Scenario Questions (High Value)

Create diagnostic questions requiring synthesis:
- "Model has 5% training error, 40% validation error. What's the problem?"
- "Dataset is 99% negative class. Model predicts all negative, 99% accuracy. What's wrong?"
- "Training diverges (loss → NaN). What solutions?"

## Question Rules

- Front-load key term: "What is **dropout**?" not "What is the technique called dropout?"
- Bold the tested concept: "How does **batch normalization** reduce training time?"
- Be specific: "What are advantages of ReLU?" not "What about ReLU?"
- Use "What" over "Explain"

## Answer Structure

**Definitions:** One sentence, 10-20 words
```
Measures the fraction of predicted positives that are actually positive.
```

**Formulas:** Formula + symbols only (no explanation)
```
$$\text{Precision} = \frac{TP}{TP + FP}$$

- $TP$: true positives
- $FP$: false positives
```

**Characteristics:** 3-5 bullets, parallel structure
```
- Produces sparse models
- Performs automatic feature selection
- Less smooth optimization
```

**Procedures:** 3-5 numbered steps, start with verb
```
1. Split data into k folds
2. Train on k-1 folds, validate on remaining
3. Repeat k times, average scores
```

**Examples:** 1-2 instances, no explanation
```
Income, house prices, response times
```

**Comparisons:** Parallel structure, 2-3 differences
```
**Bagging**: Models in parallel, reduces variance
**Boosting**: Models sequential, reduces bias and variance
```

**When to use:** Bulleted conditions
```
Choose L1 when:
- Need feature selection
- Many irrelevant features
- Want sparse model
```

## Atomicity Rules

**One question, one answer.** Answer only what's asked.

Stop as soon as the question is answered. If the answer has multiple sections, split into separate cards:
- "What is X?" + examples → two cards: definition card + example card
- "What is X?" + when to use → two cards: definition card + application card
- Definition + comparison → two cards: definition card + comparison card

Use the five question types to naturally enforce atomicity.

## Removing Duplicates

- Remove: Identical questions or answers with no learning variation
- Keep: Different questions about same concept ("What is X?" vs "When use X?")
- When unsure: Keep separate

