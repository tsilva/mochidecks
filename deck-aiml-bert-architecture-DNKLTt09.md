---
card_id: njGHv6dB
---
What is **BERT**?
---
Model that learns bidirectional text representations by seeing both left and right context simultaneously.
---
card_id: bGrGKSeD
---
What does **bidirectional architecture** enable in BERT?
---
Each token incorporates information from entire sequence, not just previous tokens.
---
card_id: dLaFY7UY
---
When should you use **BERT** over GPT-style models?
---
Choose BERT when:
- Full sentence understanding needed
- Classification or extraction tasks
- Question answering
- No text generation required
---
card_id: aGGLugWc
---
How does **BERT** differ from **GPT**?
---
**BERT**: Bidirectional attention, full context, understanding tasks / **GPT**: Unidirectional attention, left context only, generation tasks
---
card_id: qrHgGbPt
---
What are **attention patterns** in bidirectional vs unidirectional models?
---
**Bidirectional**: Full matrix, attend to all positions / **Unidirectional**: Triangular mask, attend only to previous positions
---
card_id: OJfv1DBS
---
Why can't **BERT** generate text autoregressively?
---
Bidirectional attention allows seeing future tokens, breaking the left-to-right generation constraint needed for autoregressive models.
---
card_id: PVsxlLgb
---
What is **Masked Language Modeling (MLM)**?
---
Pre-training task that masks random input tokens and predicts them using bidirectional context.
---
card_id: IR6IqVmV
---
What does **MLM** teach BERT?
---
Deep bidirectional representations by forcing the model to understand full sentence structure.
---
card_id: 4Sw41VHF
---
How does the **MLM masking strategy** work?
---
For each selected token (15% of input):
1. Replace with [MASK] (80%)
2. Replace with random token (10%)
3. Keep unchanged (10%)
---
card_id: ZLZKjUcc
---
Why use **random and unchanged tokens** in MLM (not just [MASK])?
---
Prevents learning representations only for [MASK], ensures robust representations for all vocabulary items.
---
card_id: Hoe3GaUW
---
What percentage of tokens does **MLM** mask?
---
15% of input tokens.
---
card_id: ZLjWrsU3
---
What are BERT's **special tokens**?
---
`[CLS]`: Classification token
`[SEP]`: Separator token
`[MASK]`: Mask token
`[PAD]`: Padding token
---
card_id: idXLe5pq
---
What does the **[CLS] token** represent in BERT?
---
Sentence-level representation used for classification and sentence-pair tasks.
---
card_id: 0x6GioVq
---
When is the **[SEP] token** used?
---
To separate sentence pairs in tasks like question answering or textual entailment.
---
card_id: c3XMa34k
---
What are the **three embedding types** in BERT?
---
Token embeddings (vocabulary)
Position embeddings (location)
Segment embeddings (sentence A vs B)
---
card_id: ISDtF8LE
---
How does BERT compute **input embeddings**?
---
$$\text{Embedding} = \text{Token} + \text{Position} + \text{Segment}$$
Then applies LayerNorm and dropout.
---
card_id: YFslzJ8w
---
How do BERT's **position embeddings** differ from the original Transformer?
---
**BERT**: Learned embeddings / **Original Transformer**: Fixed sinusoidal encodings
---
card_id: cQRxuSbz
---
What is the limitation of **learned position embeddings**?
---
Maximum sequence length fixed at training time (typically 512 tokens).
---
card_id: kgSmekzK
---
What are **segment embeddings** used for?
---
Distinguish sentence A from sentence B in sentence-pair tasks.
---
card_id: vyhECb0A
---
What is **WordPiece tokenization**?
---
Subword tokenization where common words are single tokens, rare words split into pieces.
---
card_id: fHLTzdP1
---
How does **WordPiece** handle rare words?
---
Splits into subword units (e.g., "playing" → "play" + "##ing").
---
card_id: nAz2ThRs
---
What activation function does **BERT** use?
---
GELU (Gaussian Error Linear Unit).
---
card_id: qKyndPLJ
---
What are advantages of **GELU** over **ReLU**?
---
Smooth and differentiable everywhere
Allows small negative values
Better gradient flow
---
card_id: 1vocw6YJ
---
What are the components of a **BERT encoder layer**?
---
1. Multi-head self-attention
2. Add & Norm (residual + LayerNorm)
3. Feed-forward network (GELU activation)
4. Add & Norm (residual + LayerNorm)
---
card_id: tme7GqRe
---
How does **BERT's self-attention** differ from Transformer encoder?
---
No difference in mechanism - both use full bidirectional attention.
---
card_id: 1xw6lxm7
---
What is the typical **intermediate size** in BERT's feed-forward layer?
---
4× hidden size (e.g., 3072 for BERT-base with 768 hidden size).
---
card_id: gxzlFzj2
---
What are the two outputs from **BERT**?
---
Sequence output: All token representations (batch, seq_len, hidden)
Pooled output: [CLS] token only (batch, hidden)
---
card_id: 66BXAiyj
---
When should you use **sequence output** vs **pooled output**?
---
**Sequence**: Token-level tasks (NER, QA span extraction) / **Pooled**: Sentence-level tasks (classification, sentiment)
---
card_id: 58Y6BNpL
---
What are **BERT-base** specifications?
---
110M parameters, 12 layers, 768 hidden size, 12 attention heads.
---
card_id: Q1waCW1z
---
What are **BERT-large** specifications?
---
340M parameters, 24 layers, 1024 hidden size, 16 attention heads.
---
card_id: 4vPLC41O
---
Model: BERT. Task: Generate text left-to-right. Problem?
---
Cannot generate because bidirectional attention sees future tokens. Need unidirectional model like GPT.
---
card_id: FHap4WrO
---
Task: Classify sentence pairs as similar/dissimilar. Which BERT components?
---
[CLS] token, [SEP] separator, segment embeddings, classification head on pooled output.
---
card_id: q6knkv73
---
Task: Extract named entities. Which BERT output?
---
Sequence output (all tokens), not pooled output, for token-level predictions.
---
card_id: K3bqtIo1
---
What is **transfer learning** with BERT?
---
Pre-train on large corpus with MLM, then fine-tune on specific tasks with labeled data.
---
card_id: UN9HS4FH
---
What are the two stages of **BERT training**?
---
Pre-training: MLM on unlabeled corpus
Fine-tuning: Task-specific training on labeled data
---
card_id: EYWrAf08
---
How do you **fine-tune BERT** for classification?
---
1. Add classification head on [CLS] token
2. Train end-to-end on labeled data
3. Update all BERT parameters
---
card_id: Gn7MIoSb
---
What are examples of **BERT variants**?
---
RoBERTa, ALBERT, DistilBERT, ELECTRA
---
card_id: jHNhgPRX
---
How does **RoBERTa** differ from BERT?
---
Removes Next Sentence Prediction task and uses dynamic masking during training.
---
card_id: u4yvDrtX
---
What makes **DistilBERT** different from BERT?
---
Smaller distilled version (40% fewer parameters) that retains 97% of BERT's performance.
---
card_id: V9rCgSdB
---
Why does MLM **masking strategy prevent overfitting**?
---
Random (10%) and unchanged (10%) tokens force learning representations for all vocabulary, not just [MASK].
---
card_id: M21ADFlk
---
Why is **bidirectional context** crucial for question answering?
---
Must see both question and surrounding context simultaneously to locate answer spans accurately.
---
card_id: BnHJqTzj
---
What loss function is used for **MLM**?
---
Cross-entropy computed only on masked positions (unmasked ignored with -100 label).
---
card_id: VRiqBym5
---
How does BERT handle **variable-length sequences**?
---
Pads with [PAD] tokens to batch maximum length, uses attention mask to ignore padding.
---
card_id: qB5nMSF1
---
What is the purpose of **attention masks**?
---
Prevents attending to padding by setting their attention scores to large negative values before softmax.
---
card_id: gFCDI2bu
---
MLM accuracy stuck at 20% during training. Possible causes?
---
Learning rate too high/low
Masking probability too high
Model too small for vocabulary
Insufficient training epochs
---
card_id: AowSev2e
---
What is **Next Sentence Prediction (NSP)**?
---
BERT's second pre-training task that predicts if sentence B follows sentence A in original text.
---
card_id: qB4jbjWN
---
Sentence: "The cat [MASK] on the mat". What does MLM predict?
---
The masked token (e.g., "sat") using both left context ("The cat") and right context ("on the mat").
---
card_id: NOACm6XC
---
When would you choose **BERT** over **RoBERTa**?
---
Choose RoBERTa - it generally outperforms BERT with better training strategy (unless you need NSP explicitly).
---
card_id: qwnlZij2
---
What makes BERT's pre-training **"bidirectional"**?
---
Each masked token prediction uses both preceding and following tokens, unlike left-to-right language models.
---
card_id: mJAc7LJ1
---
Why does BERT apply **LayerNorm** after combining embeddings?
---
Normalizes the summed embeddings to prevent large magnitude variations and stabilize training.
