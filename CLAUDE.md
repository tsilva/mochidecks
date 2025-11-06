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

## Card Design Framework

**The Five Question Types** - For important concepts, create cards across these dimensions:

1. **DEFINITION** - "What is X?"
   - Pure definition, one sentence maximum
   - Example: "What is dropout?" → "Randomly deactivates neurons during training with probability p"

2. **RECOGNITION** - "What does X do/measure/solve?"
   - Purpose or function of the concept
   - Example: "What problem does dropout solve?" → "Prevents overfitting by reducing co-adaptation"

3. **APPLICATION** - "When/why would you use X?"
   - Conditions or scenarios for use
   - Example: "When should you use dropout?" → "When model has high variance/overfitting on training data"

4. **MECHANISM** - "How does X work?"
   - Internal workings or process
   - Example: "How does dropout prevent co-adaptation?" → "Forces neurons to learn robust features independently by randomly removing connections"

5. **COMPARISON** - "How does X differ from Y?"
   - Contrasts with related concepts
   - Example: "How does dropout differ from L2 regularization?" → "Dropout randomly removes neurons; L2 shrinks all weights proportionally"

**Permutation Strategy** - How many cards per concept:

- **Foundation concepts** (4-6 cards): Core ML knowledge - bias-variance tradeoff, regularization, precision/recall, gradient descent
  - Use all 5 question types as applicable
  - Add formula cards if relevant
  - Add scenario/diagnostic cards

- **Supporting concepts** (2-3 cards): Important but narrower - specific activation functions, pooling types, metrics
  - Definition + when to use, OR
  - Formula + what it measures, OR
  - What it is + example

- **Reference concepts** (1 card): Terminology, abbreviations, minor definitions
  - Just enough to recognize when reading papers
  - Example: "What is Recall also known as?" → "Sensitivity or True Positive Rate"

## Curation Principles

**Quality standards:**
- **Atomic** - one question, one answer per card; no multi-part questions/answers
  - **Answers must ONLY contain information that directly answers the question asked**
  - Watch for answers with multiple distinct sections (definition + examples, definition + implications, definition + context, etc.)
  - **Atomicity test**: Remove each section/paragraph from the answer. If the question is still fully answered after removing it, that section should be a separate card
  - Common violations:
    - Adding examples when the question asks "What is X?" (create separate card: "What are examples of X?")
    - Adding use cases when the question asks for a definition (create separate card: "When to use X?")
    - Adding implications/effects when the question asks for a definition (create separate card: "What does X affect?")
    - Adding comparisons when the question asks about one concept (create separate card: "How does X compare to Y?")
  - **The answer should stop as soon as the question is answered** - resist the urge to add helpful context
- **Concise** - clear, focused questions and answers; use bullet lists instead of paragraphs when answers contain multiple facts
- **Unique** - no duplicate cards within or across decks
- **Unambiguous** - questions must have a single, clear interpretation with no ambiguity about what is being asked

**Removing redundancies:**
- DO remove: identical questions or near-identical content with no learning variation
- DON'T merge: different questions about the same concept ("What is X?" vs. "When to use X?" vs. "Advantages of X?")
- Each distinct question gets its own card, even if related
- When in doubt, keep cards separate
- Delete rather than merge if cards test different things

## Question Formulation

**Best practices:**
- **Use specific, unambiguous language**
  - ✗ Weak: "What about ReLU?"
  - ✓ Strong: "What are the advantages of ReLU?"

- **Front-load the key term**
  - ✓ Better: "What is **dropout** in neural networks?"
  - ✗ Weaker: "In neural networks, what is the technique called dropout?"

- **Use bold for the concept being tested**
  - "What are the characteristics of **L1 (Lasso) regularization**?"
  - "How does **batch normalization** enable faster training?"

- **Prefer "What" over "Explain"**
  - ✓ "What does high precision, low recall indicate?"
  - ✗ "Explain what it means when precision is high and recall is low"

**Scenario-based questions** (high retention value):
- Present a concrete situation requiring diagnosis or decision
- Examples:
  - "You have a dataset with 1% positive class. A model predicts all negative and achieves 99% accuracy. What's the problem?"
  - "A model has 5% training error and 40% validation error. What's the problem?"
  - "Training error is low but validation error is high. What solutions would you try?"
- Create these for key diagnostic skills - they're harder to write but incredibly valuable

## Answer Structure

**For definitions:**
- One sentence, 10-20 words maximum
- ✓ "Measures the fraction of predicted positives that are actually positive"
- ✗ "Measures the fraction of predicted positives that are actually positive. It's calculated by TP/(TP+FP). This is important because..."

**For formulas:**
- Formula first, then symbol definitions below
- Never add explanatory text or context
- Example:
  ```
  $$\text{Precision} = \frac{TP}{TP + FP}$$

  - $TP$: true positives
  - $FP$: false positives
  ```

**For characteristics/properties:**
- Bullet lists, 3-5 items maximum
- Parallel structure (all bullets same grammatical form)
- Example:
  ```
  - Produces sparse models (drives weights to zero)
  - Performs automatic feature selection
  - Less smooth optimization
  ```

**For procedures/processes:**
- Numbered steps, 3-5 maximum
- Start each with a verb
- Example:
  ```
  1. Split data into k folds
  2. Train on k-1 folds, validate on remaining
  3. Repeat k times, average scores
  ```

**For examples:**
- 1-2 concrete instances only
- No explanation unless asked "Why?"
- "Income, house prices, response times" (not "Income because outliers affect...")

**For comparisons:**
- Use parallel structure for contrasting points
- 2-3 key differences maximum
- Example:
  ```
  **Bagging**: Models in parallel, reduces variance
  **Boosting**: Models sequential, reduces bias and variance
  ```

**For "when to use" questions:**
- Bullet list of conditions/scenarios
- Start with "When:" or similar framing
- Example:
  ```
  Choose L1 (Lasso) when:
  - You want feature selection
  - Many features are irrelevant
  - You need a sparse, interpretable model
  ```

**Exception to brevity:**
Longer answers are acceptable when addressing natural follow-up questions that prevent confusion. Example: "Why does F1 use harmonic mean?" can include comparison to arithmetic mean because it answers the implicit "why not arithmetic mean?" question.
