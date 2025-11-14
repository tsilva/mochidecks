---
card_id: FtYFNqxY
---
What is a **VQ-VAE**?
---
Vector-Quantized Variational Autoencoder: a generative model that uses discrete latent representations instead of continuous latent spaces.
---
card_id: KrCw46gy
---
What does the **codebook** in VQ-VAE store?
---
A learned set of discrete vectors $$\mathcal{E} = \{e_1, e_2, ..., e_K\}$$ where each $$e_k \in \mathbb{R}^d$$ represents a possible latent code.
---
card_id: SnymKNIh
---
How does **vector quantization** work in VQ-VAE?
---
Maps continuous encoder output $$z_e$$ to nearest codebook vector: $$z_q = e_k$$ where $$k = \arg\min_j \|z_e - e_j\|_2$$
---
card_id: r7pPUhCo
---
What are the three components of **VQ-VAE architecture**?
---
**Encoder**: Maps input $$x$$ to continuous $$z_e$$
**Vector Quantizer**: Quantizes $$z_e$$ to discrete $$z_q$$
**Decoder**: Reconstructs $$\hat{x}$$ from $$z_q$$
---
card_id: JOEVdSvo
---
What is the **VQ-VAE loss function** with EMA updates?
---
$$\mathcal{L} = \|x - \hat{x}\|_2^2 + \beta \|z_e - \text{sg}[e]\|_2^2$$

Reconstruction loss + commitment loss (no codebook loss with EMA).
---
card_id: ZRKMy9a4
---
What does $$\text{sg}[\cdot]$$ represent in VQ-VAE loss?
---
Stop-gradient operator: prevents backpropagation through the codebook vectors in commitment loss.
---
card_id: IwAOMrnL
---
What is the **straight-through estimator** in VQ-VAE?
---
A gradient trick that copies gradients from decoder directly to encoder: $$\frac{\partial \mathcal{L}}{\partial z_e} = \frac{\partial \mathcal{L}}{\partial z_q}$$
---
card_id: Bk2tfL48
---
How is the straight-through estimator implemented in code?
---
`z_q = z_e + (z_q - z_e).detach()`

This maintains discrete quantization while allowing gradient flow.
---
card_id: UiQ4ALSn
---
Why is the **straight-through estimator** needed in VQ-VAE?
---
Quantization (nearest neighbor selection) is not differentiable, so gradients cannot flow through it naturally.
---
card_id: RLicvhPf
---
What does **commitment loss** encourage in VQ-VAE?
---
Encourages encoder outputs to stay close to their chosen codebook vectors: $$\|z_e - \text{sg}[e]\|_2^2$$
---
card_id: z7iP1qrg
---
What is **EMA (Exponential Moving Average)** used for in VQ-VAE?
---
Updates codebook vectors without gradients by tracking moving averages of encoder outputs assigned to each code.
---
card_id: nx0nbMPY
---
What are the three EMA update equations for VQ-VAE codebook?
---
**Cluster size**: $$N_i^{(t)} = \gamma N_i^{(t-1)} + (1-\gamma) n_i^{(t)}$$
**Embedding sum**: $$m_i^{(t)} = \gamma m_i^{(t-1)} + (1-\gamma) \sum_{j: \text{code}(z_j)=i} z_j^{(t)}$$
**Normalize**: $$e_i^{(t)} = \frac{m_i^{(t)}}{N_i^{(t)}}$$
---
card_id: DAKfb5ef
---
What are advantages of **EMA updates** over gradient-based codebook updates?
---
More stable training (no learning rate for codebook)
Better code utilization
Simpler loss (no codebook loss term needed)
---
card_id: JZPlvoeV
---
What is **perplexity** in VQ-VAE?
---
Measures diversity of codebook usage: $$\text{Perplexity} = \exp\left(-\sum_{k=1}^K p_k \log p_k\right)$$ where $$p_k$$ is probability code $$k$$ is used.
---
card_id: hfAkhogH
---
What does **high perplexity** indicate in VQ-VAE?
---
Diverse codebook usage with many codes being actively used (healthy model).
---
card_id: c5gng9CJ
---
What does **low perplexity** indicate in VQ-VAE?
---
Codebook collapse: only a few codes dominate while most remain unused.
---
card_id: xfoKpMjf
---
What is **codebook collapse** in VQ-VAE?
---
When only a small subset of codebook vectors are used, wasting most of the codebook capacity.
---
card_id: vLP9KJT1
---
When should you use a **smaller codebook** in VQ-VAE?
---
Simple architectures (e.g., MLPs)
Limited training data
Want better code utilization with EMA
Faster training convergence
---
card_id: 0BDRmc5L
---
When should you use a **larger codebook** in VQ-VAE?
---
Complex data with fine details
Strong architectures (e.g., CNNs)
High-dimensional latent spaces
Need more expressive representations
---
card_id: kmo0Gizv
---
What is **posterior collapse** in standard VAEs?
---
When the model ignores the latent space and relies only on the decoder, making the latent representation useless.
---
card_id: J4KQigXw
---
Why does VQ-VAE avoid **posterior collapse**?
---
Discrete codes force the model to use the latent space since the decoder can only access quantized representations.
---
card_id: ecc6A3dI
---
How does VQ-VAE differ from standard VAE in **latent space**?
---
**VQ-VAE**: Discrete latent codes from learned codebook
**VAE**: Continuous latent vectors $$z \in \mathbb{R}^d$$
---
card_id: ss4I1xBK
---
How does VQ-VAE differ from standard VAE in **reconstruction quality**?
---
**VQ-VAE**: Sharper reconstructions preserving fine details
**VAE**: Often blurry reconstructions due to continuous averaging
---
card_id: AlxomuS5
---
How does VQ-VAE differ from standard VAE in **prior modeling**?
---
**VQ-VAE**: Can use autoregressive models (PixelCNN, Transformers) on discrete codes
**VAE**: Uses parametric prior (typically $$\mathcal{N}(0, I)$$)
---
card_id: 4QZ5ORMQ
---
What are four advantages of **discrete latent spaces** in VQ-VAE?
---
No posterior collapse
Sharper reconstructions
Compositional/hierarchical generation
Compatible with autoregressive models
---
card_id: waJXAUjx
---
What problem do **blurry reconstructions** in VAE stem from?
---
Continuous interpolation in latent space averages out sharp features during reconstruction.
---
card_id: cpdnfjyr
---
How many codes should be used if perplexity is 64 in a 512-code VQ-VAE?
---
Codebook utilization: $$64/512 = 12.5\%$$ - indicates severe codebook collapse.
---
card_id: ffuCSmtx
---
What is maximum possible perplexity for a codebook of size $$K$$?
---
$$K$$ (when all codes are used equally).
---
card_id: j6cD6BUq
---
What is minimum possible perplexity in VQ-VAE?
---
1 (when only a single code is used for all inputs).
---
card_id: BQjw0q7t
---
Model: 512 codebook size, perplexity = 480. Problem?
---
Healthy model with excellent codebook utilization (93.8%).
---
card_id: FBO0bonS
---
Model: 128 codebook size, perplexity = 15. Problem?
---
Severe codebook collapse (11.7% utilization) - consider smaller codebook or more training.
---
card_id: nUckVUdu
---
VQ-VAE reconstruction loss keeps decreasing but perplexity drops to 5. Problem?
---
Codebook collapse - model is overfitting to few codes. Reduce codebook size or adjust commitment cost.
---
card_id: bWzp1Nsq
---
What are three real-world applications of VQ-VAE?
---
**DALL-E**: Text-to-image generation with VQ-VAE + Transformer
**Jukebox**: Music generation
**VQ-VAE-2**: High-resolution image synthesis
---
card_id: X0LFCvpo
---
Why is VQ-VAE suitable for **hierarchical generation**?
---
Discrete codes can be organized in multiple scales/levels and combined compositionally to generate complex outputs.
---
card_id: eSm2MFSB
---
What is the role of **beta ($$\beta$$)** in VQ-VAE commitment loss?
---
Controls strength of commitment: higher $$\beta$$ forces encoder to stay closer to codebook vectors.
---
card_id: 8EIHfoC8
---
What happens if **commitment cost is too high** in VQ-VAE?
---
Encoder becomes overly constrained to codebook, limiting expressiveness and hurting reconstruction quality.
---
card_id: eT3MQX85
---
What happens if **commitment cost is too low** in VQ-VAE?
---
Encoder outputs may drift far from codebook vectors, making quantization less effective.
---
card_id: qdYXMwV1
---
Why are VQ-VAE reconstructions called **"sharp"**?
---
Discrete representations preserve fine details without continuous averaging that causes blurriness.
---
card_id: L8X3NUwM
---
What is the quantization operation in VQ-VAE mathematically?
---
Find nearest neighbor: $$k^* = \arg\min_k \|z_e - e_k\|_2$$ then set $$z_q = e_{k^*}$$
---
card_id: BS9fVkA6
---
How is distance to codebook vectors computed efficiently?
---
$$\|z_e - e\|^2 = \|z_e\|^2 + \|e\|^2 - 2 z_e \cdot e$$

Avoids explicit pairwise distance computation.
---
card_id: zoqvBWWU
---
What does **codebook utilization** measure in VQ-VAE?
---
Percentage of codebook vectors actively used: $$\frac{\text{perplexity}}{K} \times 100\%$$
---
card_id: MkaZ8xXN
---
Why can't standard backpropagation update the codebook in VQ-VAE?
---
The argmin operation in nearest neighbor search is discrete and non-differentiable.
---
card_id: VLFockIu
---
How does VQ-VAE enable use of **Transformer models** for generation?
---
Discrete codes form sequences that Transformers can model autoregressively: $$p(z_1, ..., z_n) = \prod_i p(z_i | z_{<i})$$
---
card_id: abPFvdFL
---
What is **VQ-VAE-2**?
---
Hierarchical extension of VQ-VAE using multiple scales of discrete latent codes for high-resolution image generation.
---
card_id: OBNHLGpY
---
Compare **VQ-VAE** and **standard autoencoder** latent representations.
---
**VQ-VAE**: Discrete codes from learned codebook (quantized)
**Autoencoder**: Continuous unconstrained vectors
---
card_id: 9RvdGKql
---
What training instability does EMA help prevent?
---
Prevents dead codes (unused codebook vectors) and reduces codebook update variance compared to gradient-based updates.
---
card_id: w1cLEBxg
---
What is the typical range for **decay parameter** $$\gamma$$ in EMA updates?
---
0.95 to 0.99 (commonly 0.99) - controls how much historical information is retained.
---
card_id: bwhr2rzW
---
Why is **Laplace smoothing** used in EMA cluster size updates?
---
Prevents cluster sizes from going to zero, which would cause division errors when normalizing codebook vectors.
---
card_id: ixqanNZ8
---
What does it mean for codes to be **compositional** in VQ-VAE?
---
Different codes can represent reusable parts/patterns that combine to form complete outputs (e.g., digit strokes).
---
card_id: lX3EJxPf
---
How do different digit classes use the codebook?
---
Each digit class uses a subset of codes; some codes are shared between similar digits, others are digit-specific.
---
card_id: 4LoTFp6S
---
What do the **most frequently used codebook vectors** typically represent?
---
Common patterns or prototypes in the dataset (e.g., typical digit shapes in MNIST).
---
card_id: WB6BR2mD
---
Why is interpolation **discrete** in VQ-VAE latent space?
---
Because latent codes are discrete, interpolation shows discrete jumps between codes rather than smooth transitions.
---
card_id: TI5VtC3q
---
What is a **learned prior** in VQ-VAE context?
---
A separate model (e.g., PixelCNN, Transformer) trained to model the distribution $$p(z)$$ over discrete codes for generation.
---
card_id: 7WKFBU3m
---
Why is random sampling from codebook not ideal for generation?
---
Without a learned prior, random codes don't follow the data distribution and produce poor quality samples.
---
card_id: SnNvFWyy
---
How does DALL-E use VQ-VAE?
---
Uses VQ-VAE to encode images as discrete codes, then trains a Transformer to generate codes conditioned on text.
---
card_id: Lua15y5p
---
What architecture improvement helps VQ-VAE on images?
---
Using **CNNs** instead of MLPs preserves spatial structure and improves reconstruction quality.
---
card_id: 4Gl4vopH
---
Training VQ-VAE: perplexity starts high then drops significantly. What's happening?
---
Codebook collapse during training - some codes dominating while others become unused.
---
card_id: CLCvrrY8
---
What is the **commitment cost** typically set to in VQ-VAE?
---
0.25 (beta = 0.25) is a standard value that balances encoder flexibility and codebook alignment.
---
