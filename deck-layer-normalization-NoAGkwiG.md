---
card_id: difoGmQI
---
What is **layer normalization**?
---
A normalization technique that normalizes activations across features for each sample independently, transforming each sample to have mean ≈ 0 and std ≈ 1 across its features.
---
card_id: 8XcOfvw2
---
What batch-size problem does **layer normalization** solve?
---
It eliminates batch-size dependency, enabling consistent behavior across any batch size including 1, unlike BatchNorm which fails with small batches.
---
card_id: ZjCQBX1W
---
When should you use **layer normalization** instead of batch normalization?
---
- Transformers and sequence models
- Small or variable batch sizes (especially batch_size=1)
- Online learning scenarios
- Variable-length sequences
---
card_id: kj7e5mrh
---
How does **layer normalization** compute normalization statistics?
---
It computes mean and variance across features (horizontally) for each sample independently, not across samples in a batch.
---
card_id: xMLr1kyu
---
What is the formula for **layer normalization**?
---
$$\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$$
$$\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$$
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$y_i = \gamma \cdot \hat{x}_i + \beta$$

where $d$ is the number of features, $\gamma$ is scale, $\beta$ is shift (both learnable).
---
card_id: UVnnXyVv
---
How does **layer normalization** differ from **batch normalization** in normalization axis?
---
**LayerNorm**: Normalizes across features (horizontally) for each sample independently

**BatchNorm**: Normalizes across samples (vertically) for each feature independently
---
card_id: RvduNABp
---
How does **layer normalization** differ from **batch normalization** in batch dependency?
---
**LayerNorm**: Batch independent, works with any batch size including 1

**BatchNorm**: Batch dependent, fails with batch_size=1 and unstable with small batches (<8)
---
card_id: VgT0pBcS
---
How does **layer normalization** differ from **batch normalization** during inference?
---
**LayerNorm**: Same computation as training, no running statistics needed

**BatchNorm**: Uses running averages computed during training, different from training mode
---
card_id: r5sjNKiX
---
Model with **layer normalization** gets batch_size=1 during inference. Problem?
---
No problem. LayerNorm computes statistics per sample independently, so it works perfectly with batch_size=1 (unlike BatchNorm which fails).
---
card_id: GSVMLAe7
---
What are the advantages of **layer normalization**?
---
- Batch size independent (works with any size including 1)
- No train/test discrepancy
- Natural handling of variable-length sequences
- Enables online learning
- Essential for Transformers
---
card_id: m3IQ4Ahv
---
What are the disadvantages of **layer normalization**?
---
- Requires sufficient features (~10+) for stable statistics
- Less effective for CNNs than BatchNorm
- Normalizes all features together (no per-channel independence)
- Different semantics than BatchNorm may be unintuitive
---
card_id: eeFDngTr
---
In which architectures is **layer normalization** essential?
---
Transformers (GPT, BERT, LLaMA, etc.) and other sequence models like RNNs/LSTMs where variable sequence lengths and batch_size=1 are common.
---
card_id: a22n0El1
---
What is **Pre-LN** in Transformers?
---
A variant where layer normalization is applied before the self-attention or feed-forward layer (instead of after), providing better gradient flow and more stable training.
---
card_id: AokNYdEf
---
How does **Pre-LN** differ from **Post-LN** in Transformers?
---
**Pre-LN**: Normalizes before layer, better gradient flow, modern standard

**Post-LN**: Normalizes after layer, original Transformer design, less stable
---
card_id: 6yKFziD2
---
When would **Pre-LN** be preferred over **Post-LN**?
---
When training very deep Transformers or experiencing training instability, as Pre-LN provides better gradient flow and smoother convergence.
---
card_id: hrnakWoU
---
What is **RMSNorm**?
---
A simplified variant of layer normalization that only scales (no mean centering), computed as $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2}$, faster and often equivalent in performance.
---
card_id: F982BBbq
---
How does **RMSNorm** differ from **layer normalization**?
---
**RMSNorm**: Only scales using root mean square, no mean centering, faster

**LayerNorm**: Subtracts mean then scales, full normalization
---
card_id: O8UeXX6L
---
Building a Transformer for text generation. Which normalization technique?
---
Layer normalization with Pre-LN placement (modern standard), as it handles variable sequence lengths, works with batch_size=1 during generation, and provides stable training.
---
card_id: Yi2LLGbu
---
Training an image classifier with batch_size=2 due to memory constraints. BatchNorm or LayerNorm?
---
LayerNorm (or GroupNorm), as BatchNorm is unstable with very small batches (<8) while LayerNorm works consistently at any batch size.
---
card_id: STqN7YpH
---
Training a large CNN with batch_size=128. BatchNorm or LayerNorm?
---
BatchNorm, as it's specifically designed for CNNs with large batches and exploits spatial structure better than LayerNorm.
---
card_id: XMDjlglj
---
What is the minimum number of features recommended for **layer normalization** stability?
---
At least 10+ features, as LayerNorm computes statistics across features and needs sufficient dimensions for stable mean and variance estimates.
---
card_id: PysjoBr2
---
What are the learnable parameters in **layer normalization**?
---
$\gamma$ (scale/weight) and $\beta$ (shift/bias), both with the same shape as the normalized dimension, allowing the model to undo normalization if needed.
---
card_id: jm9Q2uLF
---
What is the purpose of epsilon ($\epsilon$) in **layer normalization**?
---
Added to variance ($\sigma^2 + \epsilon$) to prevent division by zero when variance is very small, typically set to 1e-5.
---
card_id: iSjdgg0z
---
How do you implement **layer normalization** in PyTorch?
---
```python
ln = nn.LayerNorm(features)
output = ln(input)
```

where `features` is the dimension to normalize (typically the last dimension).
---
card_id: hHQkXG1c
---
**Layer normalization** applied to 3D tensor with shape (batch, seq_len, features). What is normalized?
---
The `features` dimension independently for each (batch, seq_len) position, normalizing each sequence position's features to mean≈0, std≈1.
---
card_id: MPLlGSN5
---
Common pitfall: applying **layer normalization** to wrong dimension in shape (batch, features). Solution?
---
Use `nn.LayerNorm(features)` which normalizes the last dimension, or manually specify `dim=1` to normalize across features, not across batch (dim=0).
---
card_id: 0pJJiLuL
---
Building an online learning system that processes one sample at a time. Which normalization?
---
Layer normalization, as it computes statistics per sample independently and works perfectly with batch_size=1 (unlike BatchNorm which requires batch statistics).
---
card_id: duSyLcIm
---
What happens when **batch normalization** receives batch_size=1?
---
It fails because variance calculation across a single sample is meaningless (always zero), making normalization unstable or undefined.
---
card_id: mp7f9BFE
---
Why is **layer normalization** preferred for RNNs over batch normalization?
---
RNNs process variable-length sequences and often need batch_size=1, where LayerNorm works naturally while BatchNorm fails or requires awkward handling of sequence lengths.
---
card_id: yBrCPC7V
---
When should you choose **layer normalization** vs **batch normalization**?
---
**Choose LayerNorm**: Sequences, Transformers, RNNs, small batches, online learning, batch_size=1

**Choose BatchNorm**: Images with large batches (>32), CNNs with stable batch sizes
---
card_id: p1Te04J4
---
What does it mean that **layer normalization** has no train/test discrepancy?
---
LayerNorm uses the same computation during training and inference (per-sample statistics), unlike BatchNorm which uses batch statistics during training but running averages during inference.
---
card_id: CO1RkGbT
---
Why does **layer normalization** not need running statistics?
---
It computes statistics per sample independently rather than across batches, so no accumulated/running statistics are needed for inference.
---
card_id: 88dHqMOF
---
Model training becomes unstable when using **layer normalization** with only 3 features. Problem?
---
Insufficient features for stable statistics. LayerNorm computes mean and variance across features, requiring ~10+ features for reliable normalization. Use more features or consider batch normalization.
---
card_id: v6dV4VCd
---
What are the steps to compute **layer normalization** for a sample?
---
1. Compute mean across features: $\mu = \frac{1}{d}\sum x_i$
2. Compute variance across features: $\sigma^2 = \frac{1}{d}\sum(x_i - \mu)^2$
3. Normalize: $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$
4. Scale and shift: $y_i = \gamma \cdot \hat{x}_i + \beta$
---
card_id: dy8SMt1R
---
What is **Group Normalization**?
---
A normalization technique that divides channels into groups and normalizes within each group independently, serving as middle ground between LayerNorm and InstanceNorm for CNNs with small batches.
---
card_id: zNG8hO0m
---
When should you use **Group Normalization** instead of layer normalization?
---
For CNNs with small batches (<8) where BatchNorm is unstable but spatial structure matters more than LayerNorm handles, particularly in computer vision tasks.
---
card_id: EguFdDCZ
---
Why does **layer normalization** work with batch_size=1?
---
It computes statistics from multiple features (e.g., 64 hidden units) within that single sample, not from multiple samples, so sufficient features provide stable mean and variance.
---
card_id: CwCPaVWg
---
Why does **batch normalization** fail with batch_size=1?
---
Variance across a single sample is always zero (no variation to measure), making the normalization $\frac{x-\mu}{0}$ undefined or requiring running statistics that don't reflect current batch.
---
card_id: p5ZlUAUE
---
What does "normalizing across features" mean semantically in **layer normalization**?
---
It means comparing how each feature of a sample relates to other features in the same sample, bringing all features to the same scale (mean=0, std=1) within that sample.
---
card_id: oqX72Ykw
---
Why is `keepdim=True` important when implementing **layer normalization**?
---
It preserves dimensions for proper broadcasting when subtracting mean and dividing by std, ensuring (batch, features) shapes align correctly with (batch, 1) statistics.
---
card_id: zna4ztmb
---
Input to **layer normalization** has all identical values (e.g., `torch.ones(4, 10) * 5.0`). What happens?
---
Variance = 0, causing division by zero. The epsilon parameter ($\epsilon$, typically 1e-5) prevents this: $\frac{x-\mu}{\sqrt{0 + \epsilon}}$ remains defined.
---
card_id: FuXk02X8
---
What is the role of learnable parameters ($\gamma$, $\beta$) in **layer normalization**?
---
They learn optimal scale and shift for each feature after normalization, providing flexibility to adjust or even undo normalization if that improves task performance.
---
card_id: Ej8nstPL
---
How is **Pre-LN** placed in a Transformer layer?
---
```
x = x + SelfAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```
Normalizes input before each sublayer, not the output.
---
card_id: HUbx46dV
---
How is **Post-LN** placed in a Transformer layer?
---
```
x = LayerNorm(x + SelfAttention(x))
x = LayerNorm(x + FeedForward(x))
```
Normalizes output after residual connection (original Transformer design).
---
card_id: 6ILBEoeS
---
Why does **Pre-LN** provide better gradient flow than Post-LN?
---
Gradients flow through normalized activations before sublayers, preventing explosion/vanishing, making very deep networks easier to train with less need for learning rate warmup.
---
card_id: d29bJ8XV
---
Which modern LLMs use **RMSNorm** instead of layer normalization?
---
LLaMA, Gopher, and other modern language models use RMSNorm (simplified LayerNorm without mean centering) for faster computation with equivalent performance.
---
card_id: zKlslcGQ
---
Why does **layer normalization** handle variable-length sequences naturally?
---
It normalizes per-sample per-position independently across features, so sequence length doesn't affect normalization statistics (unlike BatchNorm which struggles with varying lengths).
---
card_id: RLtbMxX5
---
Why does **layer normalization** stabilize training in deep networks?
---
It eliminates scale differences between features within each sample, preventing gradient explosion/vanishing caused by features with vastly different magnitudes.
---
card_id: iqUoQy8j
---
What is **Instance Normalization**?
---
A normalization technique that normalizes each channel of each sample independently, primarily used in style transfer where per-instance, per-channel statistics matter.
---
card_id: WTKpgkm9
---
How do **layer normalization**, **batch normalization**, and **instance normalization** differ in normalization scope?
---
**LayerNorm**: Across all features per sample

**BatchNorm**: Across all samples per feature

**InstanceNorm**: Per sample, per channel independently
---
card_id: XApHo8jl
---
Training CNN with **layer normalization** performs worse than BatchNorm despite large batch. Why?
---
LayerNorm doesn't exploit spatial structure well in CNNs—it normalizes across all spatial locations and channels together, losing per-channel independence that BatchNorm preserves.
---
card_id: tOze0tvy
---
Common **layer normalization** implementation error: `nn.LayerNorm(batch_size)` for input shape (batch_size, seq_len, features). Problem?
---
Wrong `normalized_shape` parameter. Should be `nn.LayerNorm(features)` to normalize the feature dimension, not the batch dimension.
---
card_id: Nh6LjwnH
---
Building a reinforcement learning agent with variable batch sizes during training. Which normalization?
---
Layer normalization, as RL often requires small/variable batch sizes and online learning, where LayerNorm's batch independence is critical.
---
card_id: JQILzAKu
---
Transformer training loss spikes and gradients explode despite using **layer normalization**. Should you switch to Post-LN?
---
No, keep Pre-LN (more stable). Check learning rate (reduce it), gradient clipping, and initialization instead. Pre-LN provides better gradient flow than Post-LN.
