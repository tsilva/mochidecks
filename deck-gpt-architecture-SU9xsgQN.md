---
card_id: R30JVG5P
---
What is **GPT**?
---
A decoder-only transformer model that uses causal masking for autoregressive text generation.
---
card_id: cy22kCLJ
---
What is the core architectural difference between **GPT** and **BERT**?
---
GPT uses causal (unidirectional) attention while BERT uses bidirectional attention.
---
card_id: QeZhG3t7
---
What is **causal masking**?
---
A triangular attention mask where each token can only attend to itself and previous tokens, never future tokens.
---
card_id: 7ySMCD15
---
What does **causal masking** prevent during training?
---
Prevents the model from "cheating" by looking at future tokens when predicting the current token.
---
card_id: d5N4yLi0
---
When should you use **causal masking**?
---
When building autoregressive models that generate text sequentially or need to predict the next token based only on past context.
---
card_id: S7zuIH9j
---
How does **causal masking** work in attention scores?
---
Sets future positions to negative infinity before softmax, making their attention weights become zero.
---
card_id: 9N2lBjw9
---
What visual pattern does a **causal mask** create?
---
An upper triangular matrix where 1 (masked) appears above the diagonal and 0 (can attend) on and below the diagonal.
---
card_id: FBZC0eBV
---
How does **causal attention** differ from **bidirectional attention**?
---
**Causal**: Each token sees only itself and past tokens (triangular mask) / **Bidirectional**: Each token sees all tokens in sequence (no mask).
---
card_id: NCrKdeJG
---
What is the primary training objective of **GPT**?
---
Next-token prediction: predicting the next token given all previous tokens in the sequence.
---
card_id: y8FXnDC8
---
What three advantages make **next-token prediction** effective for pre-training?
---
Self-supervised (no labels), efficient (every position trains), and scalable (unlimited text data).
---
card_id: AAWszv7l
---
What is **autoregressive generation**?
---
Generating text sequentially where each new token depends only on previously generated tokens.
---
card_id: 5AD7AG0e
---
When is **autoregressive generation** preferred over other approaches?
---
For open-ended text generation tasks like story writing, code completion, and conversational AI.
---
card_id: NTMBxl6z
---
What are the steps of **autoregressive generation** in GPT?
---
1. Start with input tokens 2. Compute logits for next token 3. Sample next token 4. Append to sequence 5. Repeat until done.
---
card_id: osCsZFVl
---
What is the **decoder-only architecture**?
---
A transformer architecture using only decoder blocks with causal self-attention, without encoder components.
---
card_id: h9K7d3JF
---
What are the two main sub-layers in a **GPT decoder block**?
---
Causal self-attention and feed-forward network, each with layer norm and residual connection.
---
card_id: 8bbVR76J
---
What is **pre-norm** architecture?
---
Applying layer normalization before sub-layers (attention/FFN) rather than after.
---
card_id: W29d8pMc
---
Which activation function does GPT use in its **feed-forward networks**?
---
GELU (Gaussian Error Linear Unit) instead of ReLU.
---
card_id: 4hGY7HvL
---
What is **greedy sampling**?
---
Always selecting the token with the highest probability at each generation step.
---
card_id: lKEKhcT1
---
What problem does **greedy sampling** have?
---
Often produces repetitive and boring text because it always picks the most likely token.
---
card_id: vbmQL41h
---
When should you use **greedy sampling**?
---
When you want deterministic, "safe" output and repetition is acceptable.
---
card_id: x1NcIw1s
---
What is **temperature sampling**?
---
Scaling logits by a temperature value before softmax to control randomness in token selection.
---
card_id: ElOywGYL
---
How does **temperature** affect sampling?
---
**T < 1**: More deterministic (sharper distribution) / **T = 1**: Unchanged / **T > 1**: More random (flatter distribution).
---
card_id: KaUJN276
---
What is the formula for **temperature sampling**?
---
$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$ where $T$ is temperature and $z_i$ are logits.
---
card_id: GISKzpXv
---
What does **top-k sampling** do?
---
Only considers the k most likely tokens and samples from that subset, preventing sampling from very unlikely tokens.
---
card_id: Vq3rHHlX
---
What limitation does **top-k sampling** have?
---
Fixed k doesn't adapt to distribution shape—uses k tokens whether model is confident or uncertain.
---
card_id: 5aovw7KG
---
When is **top-k sampling** useful?
---
When you want bounded diversity and need to prevent unlikely tokens while maintaining some randomness.
---
card_id: FB8qFnlT
---
What is **nucleus sampling** (top-p)?
---
Dynamically selecting the smallest set of tokens whose cumulative probability exceeds p, then sampling from that set.
---
card_id: xEyIvLGl
---
How does **nucleus sampling** adapt to model confidence?
---
Uses a small nucleus when one token is very likely (deterministic) and a large nucleus when probabilities are spread out (diverse).
---
card_id: ct6Miwil
---
Why is **nucleus sampling** preferred in modern LLMs?
---
It provides the best quality-diversity tradeoff by adapting the number of candidate tokens based on the probability distribution.
---
card_id: glZQY6G2
---
What typical **nucleus sampling** parameters do modern LLMs use?
---
Top-p around 0.9 with temperature around 0.7-0.9.
---
card_id: tpqOF0D4
---
How do **top-k** and **nucleus sampling** differ?
---
**Top-k**: Fixed number of tokens (k) / **Nucleus**: Variable number of tokens based on cumulative probability threshold (p).
---
card_id: haWxp9cd
---
What are the complete components of the **GPT architecture**?
---
Token embedding, positional embedding, stack of decoder blocks, final layer norm, and language modeling head.
---
card_id: Wgx8kO0u
---
What type of positional embeddings does **GPT** use?
---
Learnable positional embeddings rather than fixed sinusoidal embeddings.
---
card_id: lzSWe05b
---
What is **weight tying** in GPT?
---
Sharing the same weight matrix between token embeddings and the language modeling head output projection.
---
card_id: BC5ZjjLF
---
What is the training objective function for **GPT**?
---
$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{<t})$$ (negative log-likelihood of predicting each token).
---
card_id: qHpuoqLO
---
What use cases is **GPT** better suited for than **BERT**?
---
Text generation, code completion, story writing, and conversational AI tasks.
---
card_id: jWvpa8wX
---
What use cases is **BERT** better suited for than **GPT**?
---
Text classification, named entity recognition, question answering, and understanding tasks requiring full context.
---
card_id: BCxu7gnJ
---
Model: Loss is 3.5 at position 0 and 1.2 at position 5 in a sequence. Why?
---
Early positions have less context (higher uncertainty and loss) while later positions have more accumulated context (lower uncertainty and loss).
---
card_id: 5EefMmUH
---
Why does each position in a sequence provide a **training signal** in GPT?
---
Because next-token prediction is applied at every position simultaneously—position t predicts token t+1.
---
card_id: nGTIv1qq
---
What is **self-supervised learning** in GPT?
---
Training on unlabeled data where the task (next-token prediction) is automatically created from the data itself.
---
card_id: HrQRKAfy
---
What are the key architectural choices that define **GPT**?
---
Decoder-only, learnable positional embeddings, pre-norm, GELU activation, and causal masking throughout.
---
card_id: Q2AaqNuY
---
During generation, what does GPT do when the sequence exceeds **max_len**?
---
Truncates to the most recent max_len tokens to stay within the positional embedding range.
---
card_id: yrzK0S5D
---
What is the role of the **language modeling head** in GPT?
---
Projects the final hidden states to vocabulary-sized logits for next-token prediction.
---
card_id: V89XRjf1
---
How does **gradient clipping** help GPT training?
---
Prevents exploding gradients by limiting the norm of gradients to a maximum value (typically 1.0).
---
card_id: JaA18szA
---
What does the **scale factor** in attention computation do?
---
Divides attention scores by $$\sqrt{d_k}$$ (square root of head dimension) to prevent very large values before softmax.
---
card_id: Gs0uhgb0
---
In a trained GPT model, why can changing a future token not affect past token representations?
---
Causal masking ensures past positions have zero attention weight to future positions, preventing information flow backward.
---
card_id: U4YW056s
---
What is the standard ratio for **feed-forward hidden dimension** in GPT?
---
4x the model dimension (if d_model=128, then d_ff=512).
---
card_id: EI3pO7NO
---
Model generates "the the the the the..." repeatedly. Which sampling strategy failed and what should you use?
---
Greedy sampling failed. Use temperature sampling (0.7-0.9) or nucleus sampling (p=0.9) to add diversity.
---
card_id: qFOvSf9B
---
Model generates nonsensical words with high temperature (T=2.0). Problem and solution?
---
Too much randomness samples very unlikely tokens. Lower temperature to 0.7-1.0 or use nucleus sampling to filter unlikely tokens.
---
card_id: zJjYEyBd
---
Why use **pre-norm** (layer norm before sub-layers) instead of post-norm in GPT?
---
Pre-norm provides more stable gradients and easier training, especially for deep models.
---
card_id: EzN94LZh
---
What advantage does **self-supervised learning** provide for scaling GPT?
---
Can train on unlimited unlabeled text data from the internet without manual annotation costs.
---
card_id: LBzyS0Y7
---
Why does GPT use **GELU** activation instead of **ReLU**?
---
GELU is smoother and provides better gradients, leading to improved training performance in transformers.
---
card_id: AaVhXKdj
---
Training a character-level GPT. Why is predicting the first character harder than the last?
---
First character has no context (just start token) while last character has full name context, reducing uncertainty.
---
card_id: lAf6GtQl
---
What happens to masked positions after **softmax** in causal attention?
---
They become exactly zero probability because exp(-inf) = 0.
---
card_id: Z2ZrQZaj
---
Why is **weight tying** used between embeddings and output projection?
---
Reduces parameters and helps the model learn consistent representations for input and output tokens.
---
card_id: wBj5w4lA
---
What are examples of **GPT**-style models in production?
---
GPT-3, GPT-4, ChatGPT, GitHub Copilot, and other large language models.
---
card_id: 9qtd3Jw2
---
Model output quality is good but too predictable. Which parameter should you adjust?
---
Increase temperature (to 0.8-1.2) or lower top-p (to 0.8-0.85) to increase diversity.
---
card_id: ShvWjOwO
---
Why does GPT apply attention masking **before softmax** rather than after?
---
Setting masked positions to -inf before softmax makes them become zero probability, completely blocking information flow.
---
card_id: k7jGH4Ev
---
What is the order of operations in **pre-norm** GPT blocks?
---
Layer norm → Sub-layer (attention or FFN) → Add to residual (skip connection).
---
card_id: RS4yM1Ji
---
How does **multi-head attention** work in GPT's causal attention?
---
Splits embeddings into multiple heads, applies causal attention independently per head, then concatenates results.
---
card_id: QunAyaLM
---
Why does every position provide a **training signal** in GPT?
---
Each position simultaneously predicts its next token, making training efficient—one sequence provides multiple training examples.
---
card_id: vWnlHwQW
---
Model must generate exactly 50 tokens with no randomness. Which sampling strategy?
---
Greedy sampling—always selects the highest probability token deterministically.
---
card_id: cTnz3gFd
---
What does the **attention scale factor** prevent in GPT?
---
Prevents very large attention scores that would cause vanishing gradients after softmax by dividing by $$\sqrt{d_k}$$.
---
card_id: brPGxAzV
---
Why does GPT use **learnable positional embeddings** instead of sinusoidal?
---
Learnable embeddings can adapt to the specific patterns in training data, potentially improving performance.
---
card_id: MGfsDEgE
---
What loss function is used for **next-token prediction** in GPT?
---
Cross-entropy loss between predicted logits and actual next tokens.
---
card_id: uGHFvdst
---
In GPT training, how are padding positions handled in the loss?
---
Padding targets are set to -100, which CrossEntropyLoss ignores automatically.
---
card_id: dabOphym
---
What is the difference between **encoder-decoder** and **decoder-only** transformers?
---
**Encoder-decoder**: Separate encoder and decoder with cross-attention / **Decoder-only** (GPT): Only decoder blocks with causal self-attention.
---
card_id: pdNLIneV
---
You want creative, diverse story generation. What **nucleus sampling** parameters?
---
Top-p around 0.92-0.95 with temperature 1.0-1.2 for high diversity while filtering very unlikely tokens.
---
card_id: 0mQ6fKeX
---
What happens in GPT when you sample the **special token** (e.g., end-of-sequence)?
---
Generation stops and the sequence is considered complete.
---
card_id: aGnULtW8
---
Why does GPT process a **batch of training sequences** efficiently?
---
All positions predict their next token simultaneously in parallel, not sequentially.
