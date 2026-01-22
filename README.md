<div align="center">
  <img src="logo.png" alt="mochi-decks" width="512"/>

  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Decks](https://img.shields.io/badge/Decks-19-purple)](.)
  [![mochi-mochi](https://img.shields.io/badge/Sync%20with-mochi--mochi-orange)](https://github.com/tsilva/mochi-mochi)

  **🧠 Curated flashcard decks for AI/ML, data science, and mathematics — ready to sync with [Mochi](https://mochi.cards) 🃏**

  [mochi-mochi CLI](https://github.com/tsilva/mochi-mochi) · [Mochi App](https://mochi.cards)
</div>

## Overview

A collection of markdown-formatted flashcard decks designed for spaced repetition learning. Each deck follows a structured format optimized for retention, covering topics from neural networks to linear algebra.

## Features

- **Atomic cards** — One concept per card for optimal recall
- **Rich formatting** — LaTeX math, syntax-highlighted code, markdown
- **Git-tracked** — Version control your learning progress
- **Sync-ready** — Pull from and push to Mochi via [mochi-mochi](https://github.com/tsilva/mochi-mochi)

## Available Decks

### AI/ML
| Deck | Topics |
|------|--------|
| `deck-aiml-fundamentals` | Statistical analysis, model complexity, data sampling |
| `deck-aiml-neural-networks` | Regularization, CNNs, batch normalization, activation functions |
| `deck-aiml-residual-connections` | Skip connections, gradient flow, deep network training |
| `deck-aiml-bert-architecture` | BERT model, transformers, pre-training objectives |
| `deck-aiml-autoencoders` | Encoder-decoder architecture, latent representations |
| `deck-aiml-sparse-autoencoders` | Sparsity constraints, feature learning |
| `deck-aiml-siamese-networks` | Similarity learning, contrastive loss |
| `deck-neural-networks` | Fundamentals of neural network training |

### Deep Learning Components
| Deck | Topics |
|------|--------|
| `deck-attention-mechanisms` | Self-attention, cross-attention, scaled dot-product |
| `deck-gpt-architecture` | GPT model structure, autoregressive generation |
| `deck-layer-normalization` | LayerNorm vs BatchNorm, stabilization techniques |
| `deck-gradient-stability` | Vanishing/exploding gradients, initialization |
| `deck-learning-rate-schedules` | Warmup, decay strategies, cyclical learning rates |
| `deck-optimizers` | SGD, Adam, AdamW, momentum |

### Generative Models
| Deck | Topics |
|------|--------|
| `deck-autoencoders` | VAE, reconstruction loss, latent space |
| `deck-vq-vae` | Vector quantization, discrete latent codes |
| `deck-kl-divergence` | KL divergence, ELBO, distribution matching |

### Foundations
| Deck | Topics |
|------|--------|
| `deck-linear-algebra` | Vectors, matrices, dot products, transformations |
| `deck-numpy` | Broadcasting, indexing, array operations |

## Quick Start

```bash
# Install the sync tool
pip install mochi-mochi

# Clone this deck collection
git clone https://github.com/tsilva/mochi-decks.git
cd mochi-decks

# Push a deck to your Mochi account
mochi-mochi push deck-aiml-neural-networks-gkIM7hjD.md
```

## Card Format

Cards use a simple markdown structure with YAML frontmatter:

```markdown
---
card_id: unique_id
---
What is **dropout**?
---
A regularization technique that randomly sets neuron outputs to zero during training, preventing co-adaptation.
---
```

**Supported content:**
- Standard markdown formatting
- LaTeX equations: `$$E = mc^2$$`
- Code blocks with syntax highlighting

## Workflow

1. **Pull** — Download decks from Mochi: `mochi-mochi pull <deck_id>`
2. **Edit** — Modify markdown files locally
3. **Commit** — Track changes with Git
4. **Push** — Sync back to Mochi: `mochi-mochi push <filename>`

## Contributing

Contributions welcome. When adding or editing cards:

- Follow the one-concept-per-card principle
- Use the five question types: definition, recognition, application, mechanism, comparison
- Bold the key concept being tested
- Keep answers concise (definitions: 10-20 words)

## License

[MIT](LICENSE)
