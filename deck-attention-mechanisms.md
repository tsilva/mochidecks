---
card_id: null
---
What is an **attention mechanism** in neural networks?
---
A method that allows models to dynamically focus on relevant parts of the input sequence.
---
card_id: null
---
What problem does **attention** solve in sequence processing?
---
Not all elements in a sequence are equally important for predictions; attention focuses on relevant context.
---
card_id: null
---
When would you use **attention mechanisms**?
---
When processing sequences where different elements have varying importance (text, time series, images).
---
card_id: null
---
How does **attention** determine which parts of input to focus on?
---
By computing similarity scores between queries and keys, then using these to weight values.
---
card_id: null
---
What are **Query, Key, and Value** in attention?
---
Three learned representations: Q searches for information, K describes what elements offer, V contains information to retrieve.
---
card_id: null
---
What does the **Query (Q)** represent in attention?
---
The current element asking for information ("what am I looking for?").
---
card_id: null
---
What do **Keys (K)** represent in attention?
---
Descriptions of what each element offers for matching ("what do I contain?").
---
card_id: null
---
What do **Values (V)** represent in attention?
---
The actual information to retrieve based on attention weights ("what information do I have?").
---
card_id: null
---
What is the library search analogy for **Query, Key, and Value**?
---
**Query**: Your search terms
**Keys**: Book titles/descriptions
**Values**: Book contents
---
card_id: null
---
What is the formula for **scaled dot-product attention**?
---
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q$: queries, $K$: keys, $V$: values, $d_k$: key dimension
---
card_id: null
---
How does **scaled dot-product attention** compute its output?
---
1. Compute $QK^T$ similarity scores
2. Scale by $\sqrt{d_k}$
3. Apply softmax
4. Multiply by $V$ for weighted sum
---
card_id: null
---
What does **$QK^T$** compute in attention?
---
Similarity scores between queries and keys using dot products.
---
card_id: null
---
What role does **softmax** play in attention?
---
Converts similarity scores to a probability distribution where each row sums to 1.
---
card_id: null
---
Why is attention scaled by **$\sqrt{d_k}$**?
---
To prevent dot products from growing too large with dimension, which pushes softmax into regions with vanishing gradients.
---
card_id: null
---
What problem occurs without **scaling in attention**?
---
Dot products grow with dimension, causing softmax to concentrate on maximum values with tiny gradients elsewhere.
---
card_id: null
---
When does **scaling by $\sqrt{d_k}$** become more critical?
---
As the key dimension $d_k$ increases, since dot products of random vectors grow with dimension.
---
card_id: null
---
What are **attention weights**?
---
Probability distributions over sequence positions indicating how much each position attends to others.
---
card_id: null
---
How are **attention weights** used to produce output?
---
By computing a weighted sum of value vectors based on the attention probabilities.
---
card_id: null
---
What property must **attention weights** satisfy in each row?
---
Each row must sum to 1.0 (they form a probability distribution).
---
card_id: null
---
What is **causal attention**?
---
Attention where each position can only attend to itself and previous positions, not future ones.
---
card_id: null
---
When would you use **causal attention**?
---
In autoregressive language modeling where predicting position $i$ should not use information from positions after $i$.
---
card_id: null
---
How is **causal masking** implemented?
---
By setting future position attention scores to negative infinity before softmax, using a lower triangular mask.
---
card_id: null
---
What does a **causal mask** look like?
---
A lower triangular matrix with 1s (attend) below diagonal and 0s (mask) above diagonal.
---
card_id: null
---
What is **multi-head attention**?
---
Running multiple attention mechanisms in parallel where each head learns different attention patterns.
---
card_id: null
---
What problem does **multi-head attention** solve?
---
Single attention can only learn one type of relationship; multiple heads learn diverse patterns like syntactic, semantic, and positional dependencies.
---
card_id: null
---
How does **multi-head attention** combine multiple heads?
---
1. Project input to $h$ different Q, K, V heads
2. Apply scaled dot-product attention to each head independently
3. Concatenate all head outputs
4. Apply final linear projection $W^O$
---
card_id: null
---
What is the formula for **multi-head attention**?
---
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
---
card_id: null
---
Why use **multiple attention heads** instead of one large head?
---
Different heads can specialize in different types of relationships and patterns in the data.
---
card_id: null
---
What constraint must be satisfied for **multi-head attention**?
---
The model dimension $d_{model}$ must be divisible by the number of heads.
---
card_id: null
---
What is the dimension per head in **multi-head attention**?
---
$$d_k = \frac{d_{model}}{h}$$

where $d_{model}$: model dimension, $h$: number of heads
---
card_id: null
---
Scenario: Attention scores before softmax range from -15 to 20. Problem?
---
Scaling factor may be too small or missing; large scores cause softmax saturation and vanishing gradients.
---
card_id: null
---
Scenario: You want to predict next token in a sequence. Which attention type?
---
Causal (masked) attention to prevent the model from seeing future tokens.
---
card_id: null
---
What is **self-attention**?
---
Attention where Q, K, and V all come from the same input sequence.
---
card_id: null
---
What is **cross-attention**?
---
Attention where Q comes from one sequence and K, V come from a different sequence.
---
card_id: null
---
How does **self-attention** differ from **cross-attention**?
---
**Self-attention**: Q, K, V from same sequence, models internal dependencies
**Cross-attention**: Q from one sequence, K, V from another, models inter-sequence dependencies
---
card_id: null
---
How does **bidirectional attention** differ from **causal attention**?
---
**Bidirectional**: Each position attends to all positions (past and future)
**Causal**: Each position only attends to current and previous positions
---
card_id: null
---
What is the computational complexity of **attention**?
---
$O(n^2 \cdot d)$ where $n$: sequence length, $d$: model dimension.
---
card_id: null
---
Why does **attention** have $O(n^2)$ complexity?
---
Computing $QK^T$ requires comparing every position with every other position.
---
card_id: null
---
How are **Q, K, V** created from input embeddings?
---
Linear projections using learned weight matrices $W_Q$, $W_K$, $W_V$ transform input to query, key, and value representations.
---
card_id: null
---
What happens to attention weights after **softmax**?
---
They become probability distributions where each query's weights across all keys sum to 1.0.
---
card_id: null
---
Why apply **negative infinity** to masked positions?
---
After softmax, $e^{-\infty} = 0$, effectively zeroing out attention to those positions.
---
card_id: null
---
Scenario: Training a language model like GPT. Which attention mechanism?
---
Multi-head causal self-attention to attend only to previous tokens while learning diverse patterns.
---
card_id: null
---
Scenario: Building a translation model encoder. Which attention type?
---
Bidirectional multi-head self-attention to capture context from entire source sentence.
---
card_id: null
---
Scenario: All attention heads learn nearly identical patterns. Problem?
---
Heads are not diversifying; may need different initialization, learning rates, or more heads.
---
card_id: null
---
What types of patterns can different **attention heads** learn?
---
Syntactic relationships, semantic meanings, positional patterns, long-range dependencies, local context.
---
card_id: null
---
What does **concatenating attention heads** achieve?
---
Combines diverse attention patterns from all heads into a single representation.
---
card_id: null
---
Why project concatenated heads with **$W^O$**?
---
To mix information across heads and transform to desired output dimension.
---
card_id: null
---
Scenario: Sequence length increases from 100 to 1000 tokens. Attention cost impact?
---
Cost increases by $10^2 = 100\times$ since attention complexity is $O(n^2)$.
---
card_id: null
---
What does a high **attention weight** between two positions indicate?
---
The query position finds that key position highly relevant and uses more of its value information.
---
card_id: null
---
When would **attention weights** be uniform across all positions?
---
When all positions are equally relevant, or queries and keys have similar scores (no clear preference).
---
card_id: null
---
What must be true for **Q, K, V** dimensions in attention?
---
Q and K must have same dimension for dot product; V dimension determines output dimension.
---
card_id: null
---
How do **positional encodings** interact with attention?
---
They're added to input embeddings so attention can distinguish different positions in the sequence.
---
card_id: null
---
Why is attention called a **"soft"** mechanism?
---
It computes weighted combinations (soft selection) rather than hard selection of single elements.
---
---
