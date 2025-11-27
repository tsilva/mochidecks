---
card_id: RMyBSvWo
---
What is the **vanishing gradient problem**?
---
Gradients become exponentially smaller as they propagate backward through layers, eventually becoming so small that early layers barely learn.
---
card_id: ZYXTA3Oi
---
What is the **exploding gradient problem**?
---
Gradients become exponentially larger as they propagate backward, causing unstable parameter updates and potentially divergent training with NaN values.
---
card_id: qWZ21ZEE
---
Why do gradients multiply through layers during backpropagation?
---
By the chain rule, the gradient at each layer is the product of all subsequent layer gradients: $\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a_3} \cdot \frac{\partial a_3}{\partial a_2} \cdot \frac{\partial a_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial w_1}$, where each layer contributes a multiplicative factor.
---
card_id: 67krHhBE
---
What happens when gradient factors are consistently less than 1 across layers?
---
The product shrinks exponentially (vanishing gradients). For example, with factor $f=0.5$ for 10 layers: $0.5^{10} \approx 0.001$.
---
card_id: GyYmTjV6
---
What happens when gradient factors are consistently greater than 1 across layers?
---
The product grows exponentially (exploding gradients). For example, with factor $f=2.0$ for 10 layers: $2^{10} = 1024$.
---
card_id: PStNce0W
---
What is the maximum value of the **sigmoid derivative**?
---
0.25, occurring at $x=0$ where $\sigma(x) = 0.5$.
---
card_id: Jzew9t5A
---
What is the formula for the **sigmoid derivative**?
---
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$.
---
card_id: VoUT0JCp
---
What happens to sigmoid derivative values in saturated regions (very large or small inputs)?
---
The derivative approaches 0, making neurons insensitive to inputs and preventing gradients from flowing through.
---
card_id: TIFMEjLf
---
What causes **vanishing gradients** in deep sigmoid networks?
---
Each sigmoid layer contributes a derivative factor ≤ 0.25, causing exponential gradient decay. For 10 layers, gradients scale by $0.25^{10} \approx 10^{-6}$ in best case.
---
card_id: uy5kWlqy
---
What are the symptoms of **vanishing gradients**?
---
Early layers (furthest from loss) receive tiny gradients and don't learn, loss plateaus, and training longer doesn't help.
---
card_id: hkNDYO4B
---
What causes **exploding gradients**?
---
Large weight initialization combined with non-saturating activations (like ReLU) that allow gradients to multiply to huge values across layers.
---
card_id: gdUz8Lq4
---
What are the symptoms of **exploding gradients**?
---
Training loss spikes or diverges, NaN values appear in loss or parameters, and training becomes unstable.
---
card_id: oDj0iIwo
---
Network has 20 sigmoid layers. Early layers have gradients 1000× smaller than later layers. What is the problem and solution?
---
**Problem:** Vanishing gradients due to sigmoid saturation.
**Solutions:** Replace sigmoid with ReLU, add batch normalization, or add residual connections.
---
card_id: RA8A3eh1
---
Training diverges with NaN loss after a few steps. What is the problem and immediate solution?
---
**Problem:** Exploding gradients.
**Solutions:** Apply gradient clipping (e.g., max_norm=1.0), use proper weight initialization (He for ReLU, Xavier for tanh), or reduce learning rate.
---
card_id: Lx7fkmeg
---
What is the derivative of **ReLU**?
---
1 for positive inputs ($x > 0$), 0 for negative inputs ($x \leq 0$).
---
card_id: 5ERfgvck
---
What is the derivative of **Leaky ReLU**?
---
1 for positive inputs ($x > 0$), 0.2 for negative inputs ($x < 0$).
---
card_id: 1wGaKwiV
---
Why does **ReLU** help prevent vanishing gradients?
---
ReLU derivative is exactly 1 for positive inputs (no saturation), preserving gradient magnitude during backpropagation without exponential decay.
---
card_id: 6d3TbNlC
---
How does **ReLU** differ from **sigmoid** for gradient flow?
---
**ReLU:** Derivative = 1 for positive inputs, no saturation, preserves gradients.
**Sigmoid:** Max derivative = 0.25, saturates for large |x|, causes vanishing gradients.
---
card_id: uWJDUQ4c
---
What is the **dying ReLU** problem?
---
When many ReLU neurons output 0 for all inputs, their gradients become 0 and they stop learning permanently.
---
card_id: maJHIKwH
---
When should you use **Leaky ReLU** instead of **ReLU**?
---
When concerned about dying ReLU problem, as Leaky ReLU allows small gradient (0.2) for negative inputs, keeping neurons alive.
---
card_id: vcP1L0MX
---
What advantage does **tanh** have over **sigmoid** for gradient flow?
---
Tanh has maximum derivative of 1.0 (vs sigmoid's 0.25), reducing but not eliminating vanishing gradients.
---
card_id: hgYrplcu
---
What is **Xavier (Glorot) initialization**?
---
Weight initialization method that draws weights from $\mathcal{U}\left[-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right]$ to preserve variance for symmetric activations like tanh and sigmoid.
---
card_id: oFlKJnXH
---
What is **He initialization**?
---
Weight initialization method that draws weights from $\mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$ to preserve variance for ReLU activations, accounting for half the neurons being zeroed.
---
card_id: ursFQXJ8
---
When should you use **He initialization** vs **Xavier initialization**?
---
**He:** For ReLU and Leaky ReLU (accounts for zeroed neurons).
**Xavier:** For tanh, sigmoid, and linear layers (symmetric activations).
---
card_id: MO5Ay3CM
---
Why does proper weight initialization matter for gradient flow?
---
Proper initialization preserves activation and gradient variance across layers, preventing initial vanishing or exploding gradients and starting the network in a trainable state.
---
card_id: IRPfvnNE
---
What happens with too small weight initialization (e.g., uniform [-0.01, 0.01])?
---
Activations and gradients become very small, causing more severe vanishing gradients even with proper activation functions.
---
card_id: jx7cAsgU
---
What happens with too large weight initialization (e.g., uniform [-2, 2])?
---
With ReLU, causes exploding gradients and unstable training. With sigmoid/tanh, causes immediate saturation and vanishing gradients.
---
card_id: TX2n0trB
---
What is **gradient clipping**?
---
A technique that caps gradient magnitude at a threshold while preserving direction, preventing exploding gradients.
---
card_id: ezZZZSdu
---
What is the formula for **gradient clipping**?
---
$$\text{if } ||g|| > \text{threshold}: \quad g \leftarrow g \cdot \frac{\text{threshold}}{||g||}$$

where $g$ is the gradient vector and $||g||$ is its norm.
---
card_id: PyIW5uvx
---
When should you use **gradient clipping**?
---
For RNNs/LSTMs (essential), when training is unstable with loss spikes, or when using high learning rates with poorly conditioned loss surfaces.
---
card_id: 7KwZLINN
---
Does **gradient clipping** solve the root cause of exploding gradients?
---
No, it treats the symptom. Root causes are poor initialization, inappropriate activation functions, or architectural issues. Better to fix these when possible.
---
card_id: 7FnRFuEa
---
What is **batch normalization**?
---
A normalization technique that normalizes layer inputs to have mean=0 and variance=1 across the batch, then learns affine transformation parameters $\gamma$ and $\beta$.
---
card_id: HlQ9dj64
---
What is the formula for **batch normalization**?
---
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

where $\mu_B$ is batch mean, $\sigma_B^2$ is batch variance, and $\gamma$, $\beta$ are learnable parameters.
---
card_id: H64TU58N
---
What are the benefits of **batch normalization** for gradient flow?
---
Prevents internal covariate shift, reduces sensitivity to initialization, enables higher learning rates, and prevents activation saturation by keeping inputs in reasonable range.
---
card_id: fsDDwEWK
---
When should you use **batch normalization**?
---
Training very deep networks (>20 layers), using saturating activations (sigmoid/tanh), or when you want to use higher learning rates.
---
card_id: a0czZk8X
---
When should you avoid **batch normalization**?
---
When batch size is very small (statistics are noisy), with RNNs (use Layer Normalization instead), or when deployment complexity is a concern (train/eval mode handling).
---
card_id: gMtWSDK8
---
How does **batch normalization** help deep sigmoid networks?
---
Prevents internal covariate shift by keeping layer inputs in non-saturated regions, allowing gradients to flow even through sigmoid activations.
---
card_id: uNcHdbT3
---
What are **residual connections** (skip connections)?
---
Architectural connections that add the input directly to the layer output: $y = \text{activation}(F(x) + x)$, where $F(x)$ is the layer transformation.
---
card_id: DVByzpu9
---
What is the **gradient highway** mechanism in residual connections?
---
During backpropagation: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \left(\frac{\partial F(x)}{\partial x} + 1\right)$. The "+1" term provides a direct path for gradients to flow backward without attenuation.
---
card_id: y3z3dUBo
---
What is **pre-activation** residual architecture?
---
Adding the residual connection before the activation function: `x = activation(BN(linear(x)) + identity)`, maintaining normalized scale from batch normalization.
---
card_id: RwvhBr8G
---
Why should residual connections be added **before** the activation function?
---
Batch normalization outputs have mean=0, std=1. Adding the identity (also ~mean=0) maintains normalized scale before activation. Adding after causes scale mismatch and instability.
---
card_id: sWTjGrn9
---
What are the benefits of **residual connections**?
---
Enable training of very deep networks (100+ layers), provide gradient highways preventing vanishing gradients, and allow networks to easily learn identity mappings when needed.
---
card_id: C8eCl0aM
---
What are the convergence characteristics of **residual networks**?
---
Slower initial convergence (need warmup period of 7-8 epochs for deep networks) but achieve better final performance, as the network must learn when to use skip connections vs transformations.
---
card_id: I01WAY8i
---
What architectural requirements do very deep residual networks (>30 layers) have?
---
Must use batch normalization (or Layer Normalization) to prevent activation explosion, and residual must be added before activation (pre-activation style).
---
card_id: gC4PK3qL
---
When should you use **residual connections**?
---
Very deep networks (>20 layers), when training extremely deep models (50-100+ layers), or in Transformer architectures (always used).
---
card_id: Ic9pc601
---
When is it acceptable to NOT use **residual connections**?
---
Shallow networks (3-10 layers) where vanishing gradients aren't severe, when dimensions don't match (input/output shapes differ), or when training time budget is limited (residuals need longer convergence).
---
card_id: fjVNZ4pq
---
Network is 50 layers with ReLU and He initialization, but early layers still have tiny gradients. What solution?
---
Add residual connections (skip connections) to create gradient highways, ideally combined with batch normalization for stability.
---
card_id: FzVJDYTT
---
Training a 20-layer sigmoid network. What three techniques would most help gradient flow?
---
1. Replace sigmoid with ReLU
2. Add batch normalization
3. Add residual connections

Or keep sigmoid but use all three: batch normalization + residual connections + proper initialization.
---
card_id: azilyPWx
---
Deep network (100 layers) with residual connections trains unstably with exploding activations. What is missing?
---
Batch normalization (or Layer Normalization). Very deep residual networks require normalization to prevent activation explosion.
---
card_id: kmoBlxtG
---
What is the recommended activation function for deep networks (>10 layers)?
---
ReLU or its variants (Leaky ReLU, PReLU, ELU, GELU) because they don't saturate for positive inputs and preserve gradients.
---
card_id: CNtELBr7
---
What initialization should be used with **ReLU** activations?
---
He initialization, which accounts for half the neurons being zeroed by ReLU.
---
card_id: gX5TiaO3
---
What initialization should be used with **tanh** or **sigmoid** activations?
---
Xavier (Glorot) initialization, which preserves variance for symmetric activations.
---
card_id: jVSPs6Iv
---
What normalization technique should be used for CNNs and MLPs?
---
Batch Normalization (works well with spatial/feature dimensions and reasonable batch sizes).
---
card_id: 5tChKNIz
---
What normalization technique should be used for RNNs and Transformers?
---
Layer Normalization (doesn't depend on batch statistics, works with sequence data).
---
card_id: 5kbyfmX5
---
What normalization technique should be used with small batch sizes?
---
Group Normalization or Layer Normalization (don't rely on batch statistics which are noisy for small batches).
---
card_id: pngv0xBA
---
Early layers barely learning, loss plateaus, training longer doesn't help. What is the problem?
---
Vanishing gradients. Early layers receive tiny gradient signals and can't update effectively.
---
card_id: dMyti5Kp
---
Training shows gradients >1000 in heatmap, loss spikes to NaN. What are two immediate solutions?
---
1. Apply gradient clipping (e.g., max_norm=1.0)
2. Fix weight initialization (use He for ReLU, Xavier for tanh)
---
card_id: sJLv8KCw
---
Many ReLU neurons output 0 for all inputs. What is this problem and the solution?
---
**Problem:** Dying ReLU (neurons permanently dead).
**Solution:** Use Leaky ReLU (allows small negative gradient), lower learning rate, or better initialization.
---
card_id: eZcTRBv6
---
All sigmoid activations near 0 or 1 (saturated). What causes this and what's the solution?
---
**Cause:** Poor initialization causing large inputs to sigmoid.
**Solution:** Use proper initialization (Xavier), or better, replace sigmoid with ReLU.
---
card_id: hZMZoDI9
---
How can you diagnose vanishing gradients during training?
---
Monitor gradient magnitudes: if gradients <1e-5 in early layers, or gradient ratio (first/last layer) >1000×, you have vanishing gradients.
---
card_id: R5rol460
---
How can you diagnose exploding gradients during training?
---
Monitor gradient magnitudes: if gradients >1000, loss diverges to NaN, or training becomes unstable with spikes, you have exploding gradients.
---
card_id: aG2Yteoo
---
What is the modern deep learning recipe for networks >20 layers?
---
ReLU activation + He initialization + Batch Normalization + Residual connections + (optional) Gradient clipping if unstable.
---
card_id: cpqUb6cB
---
How does **ReLU** with large weights cause exploding gradients while **sigmoid** with large weights causes vanishing gradients?
---
**ReLU:** Derivative=1 for positive inputs (no saturation), large weights → large activations → large gradients multiply.
**Sigmoid:** Saturates for large inputs (derivative→0), large weights → saturation → tiny gradients → vanishing.
---
card_id: DH6kiEVW
---
Why can't training longer fix vanishing gradients?
---
Early layers receive gradient signals too small to cause meaningful parameter updates. No amount of training time helps when gradient magnitude is ~1e-6.
---
card_id: R7JfUsNW
---
What is **internal covariate shift**?
---
The phenomenon where layer input distributions shift during training, causing saturation and instability. Batch normalization addresses this.
---
card_id: fdvJHVTK
---
What does it mean when gradient ratio (first layer / last layer) is 1e-6?
---
Severe vanishing gradients: the first layer receives gradients a million times smaller than the last layer, preventing effective learning.
---
card_id: 65ell64x
---
Why does increasing network depth amplify gradient problems?
---
Each layer contributes a multiplicative factor to gradients. With $n$ layers and factor $f$, gradients scale by $f^n$, causing exponential growth/decay.
---
card_id: ksWQY5x9
---
What is the advantage of **Leaky ReLU** parameter (0.2 for negative inputs)?
---
Allows small gradient flow even for negative inputs, preventing neurons from dying permanently (unlike ReLU which has 0 gradient for negatives).
---
card_id: crbOxnwR
---
What is the maximum derivative of **tanh**?
---
1.0, occurring at $x=0$ where $\tanh(x) = 0$.
---
card_id: N5fi2zcu
---
What does **variance preservation** mean for weight initialization?
---
Initializing weights such that variance of activations (forward pass) and gradients (backward pass) remains constant across layers.
---
card_id: bftdpvaT
---
Residual network (50 layers) with BN shows loss=0.67 after 3 epochs. Is this a problem?
---
No, residual networks need longer warmup (7-8 epochs) to learn when to use skip connections. Check performance after 15+ epochs.
---
card_id: UUfdsLPv
---
Why do residual networks require more training epochs than plain networks?
---
Network must learn when to use skip connections (identity) vs when to transform data through layers, requiring a warmup period.
---
card_id: KTHhB9hJ
---
What is the role of learnable parameters **gamma** and **beta** in batch normalization?
---
After normalizing to mean=0, std=1, gamma (scale) and beta (shift) are learned to allow the network to undo normalization if needed, giving the model flexibility.
---
card_id: kGKjc54e
---
What epsilon value is used in batch normalization formula and why?
---
A small constant (e.g., 1e-5) added to variance in denominator: $\sqrt{\sigma_B^2 + \epsilon}$, preventing division by zero for constant inputs.
---
card_id: BXc5VIUB
---
Which layers should gradient statistics be monitored in: all layers, only early layers, or only final layers?
---
All layers, but especially compare early (first few) vs late (last few) layers to detect vanishing (large ratio) or exploding (all large) gradients.
---
card_id: PHvmKseP
---
What is the gradient flow problem specific to **RNNs**?
---
Gradients must backpropagate through time over many timesteps, causing severe vanishing/exploding gradients. Always require gradient clipping and proper architecture (LSTM/GRU).
