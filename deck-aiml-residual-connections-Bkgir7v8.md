---
card_id: jekol9xI
---
What is a **residual connection** (skip connection)?
---
An architectural element that adds the input directly to the output: `output = F(x) + x`.
---
card_id: ZwMFDIvJ
---
What problem do **residual connections** solve in deep neural networks?
---
The degradation problem where deeper networks perform worse than shallower ones, even on training data.
---
card_id: Hwf7IvOD
---
When should you use **residual connections** in a neural network?
---
When:
- Building very deep networks (20+ layers)
- Vanishing gradients prevent training
- Need to preserve input information throughout depth
---
card_id: BCF0fbgc
---
How does a **residual block** compute its output?
---
1. Compute transformation: `residual = F(x)`
2. Add input (skip connection): `output = residual + x`
3. Apply activation: `output = activation(output)`
---
card_id: mpU6LmeK
---
How does a **residual connection** differ from a **plain connection**?
---
**Residual**: Adds input directly to output (`F(x) + x`), preserves gradient flow / **Plain**: Only passes through transformations (`F(x)`), gradients can vanish.
---
card_id: e80sWNLk
---
What is the **degradation problem** in deep neural networks?
---
The phenomenon where deeper networks have higher training error than shallower networks, indicating optimization difficulty rather than overfitting.
---
card_id: lnf0mkTo
---
What does the **degradation problem** reveal about deep networks?
---
That learning the identity function through multiple layers is surprisingly difficult for plain networks.
---
card_id: Qe1eSCoF
---
Why is learning the **identity function** `H(x) = x` hard for plain neural networks?
---
Must learn complex inverse relationships through nonlinear layers, and ReLU destroys information at negative values.
---
card_id: 5z0ez6Nf
---
How do **residual connections** make learning identity easier?
---
Network only learns `F(x) = 0` (push weights to zero) while skip connection provides identity automatically.
---
card_id: t2tpdBoW
---
What is the **residual formulation** in ResNets?
---
Learn the difference from identity `F(x) = H(x) - x`, then compute output as `H(x) = F(x) + x`.
---
card_id: s2MDdNtI
---
What is a **residual** in the context of ResNets?
---
The difference between the desired output and input: `F(x) = H(x) - x`.
---
card_id: qEptGqdc
---
What is the **gradient highway** in residual networks?
---
The direct gradient path through skip connections that prevents vanishing gradients during backpropagation.
---
card_id: iloCzBWK
---
What does the **gradient highway** enable in deep networks?
---
Training very deep networks by ensuring gradients flow directly to early layers without exponential decay.
---
card_id: SVkT9ogu
---
How does the gradient formula for **residual connections** prevent vanishing gradients?
---
The gradient `∂y/∂x = ∂F(x)/∂x + 1` always includes `+1`, ensuring gradients never vanish even if `∂F(x)/∂x` becomes small.
---
card_id: TzLbtK0M
---
Model: 50-layer plain network has higher training error than 20-layer plain network. Problem?
---
Degradation problem - vanishing gradients prevent deep networks from learning effectively, even though the deeper network has more capacity.
---
card_id: lThm8w2R
---
What is the **ensemble interpretation** of ResNets?
---
A ResNet with `n` blocks behaves as an ensemble of `2^n` networks of varying depths.
---
card_id: lHpY7yjs
---
Why do ResNets behave like **ensembles**?
---
Each block offers two paths (through `F(x)` or skip), creating `2^n` possible paths through `n` blocks.
---
card_id: zpgms99w
---
How does the **ensemble interpretation** explain ResNet robustness?
---
Most paths are shallow (easy to train) and dominate learning, while long paths provide bonus capacity. Dropping individual blocks has limited impact.
---
card_id: I4RaVJHv
---
What is **feature reuse** in residual networks?
---
Skip connections propagate early layer features unchanged to later layers for combined use.
---
card_id: 779PhNHn
---
What does **feature reuse** enable in ResNets?
---
- Access to both original and refined features at all depths
- No need to re-learn basic features
- Prevents information loss through compression
---
card_id: ifxUz7pa
---
How do **residual connections** prevent the information bottleneck?
---
Skip connections propagate raw input directly to all layers, bypassing layer-by-layer compression.
---
card_id: 35MC1Pke
---
What do **residual blocks** learn in practice?
---
Small refinements to input where `||F(x)|| << ||x||`, not complete transformations.
---
card_id: ceIyVJma
---
Model: ResNet trained successfully. Dropping one block causes small performance drop. Why?
---
Ensemble behavior - other paths among `2^n` total paths compensate for the removed block.
---
card_id: SOthpcYL
---
What is **modular learning** in ResNets?
---
Each block learns small incremental refinements rather than complete transformations.
---
card_id: aMilnBAX
---
How does **gradient flow** differ between plain and residual networks at depth 50?
---
**Plain**: Gradients decay exponentially (e.g., 10^-6 at early layers) / **Residual**: Gradients remain strong (e.g., 0.8+ at early layers) due to skip connections.
---
card_id: Pd8Gth5h
---
Why can't plain deep networks learn **identity mappings** for unnecessary layers?
---
Requires learning complex inverse relationships through nonlinearities, made harder by ReLU information loss.
---
card_id: 5C6BcuKZ
---
What makes the **residual formulation** easier to optimize than direct learning?
---
Learning `F(x) = 0` (push weights to zero) is simpler than learning `H(x) = x`.
---
card_id: bBmBCzRB
---
What is **vanishing gradients** in deep plain networks?
---
Exponential decay of gradient magnitudes during backpropagation, causing early layers to stop learning.
---
card_id: ulheTZ6X
---
How do **skip connections** maintain gradient magnitude?
---
The derivative `∂(F(x) + x)/∂x = ∂F(x)/∂x + 1` always includes `+1`, preventing multiplicative decay.
---
card_id: yUHal8br
---
What is the mathematical form of a **residual block**?
---
$$y = F(x, \{W_i\}) + x$$
where $F$ is the residual function with weights $W_i$, and $x$ is the skip connection.
---
card_id: TVnTskA5
---
How many paths exist through a ResNet with `n` residual blocks?
---
$$2^n$$ paths, as each block offers two choices: through $F(x)$ or via skip connection.
---
card_id: BrAua6X9
---
What is the gradient derivative for a residual connection `y = F(x) + x`?
---
$$\frac{\partial y}{\partial x} = \frac{\partial F(x)}{\partial x} + 1$$
The $+1$ term ensures gradients always flow.
---
card_id: zuByrtEH
---
What distribution do **path lengths** follow in a ResNet with `n` blocks?
---
Binomial distribution peaking at middle depths - most paths are shallow rather than using all blocks.
---
card_id: sS1h1m9c
---
Model: Deep plain network trains slowly, first layers barely update. Solution?
---
Add residual connections to create gradient highways, ensuring the `+1` gradient term prevents vanishing gradients in early layers.
---
card_id: dX8zFMCp
---
Why do later layers in ResNets have access to **early features**?
---
Skip connections bypass intermediate transformations, propagating raw features unchanged to any depth.
---
card_id: fHFsPO3w
---
How does **cosine similarity** between first and last layer features differ in plain vs residual networks?
---
**Plain**: Low similarity (~0.1-0.3), features transformed completely / **Residual**: High similarity (~0.6-0.9), input information preserved via skip connections.
---
card_id: Q80fzFz1
---
What are the five key reasons **residual connections** work?
---
1. Easy identity learning: $F(x) = 0$
2. Gradient highway: $+1$ term
3. Implicit ensemble: $2^n$ paths
4. Feature reuse: inputs propagate unchanged
5. Modular learning: small refinements
---
card_id: pxNvnE82
---
Which came first: **Highway Networks** or **ResNets**?
---
Highway Networks (2015) used learnable gates; ResNets simplified this with fixed skip connections.
---
card_id: R8661SIj
---
How do **Transformers** use residual connections?
---
Around attention and feed-forward layers using `output = sublayer(x) + x`.
---
card_id: 9w2hmCaj
---
How does **DenseNet** extend the idea of skip connections?
---
Connects each layer to ALL previous layers for maximum feature reuse.
---
card_id: oRb1cvUO
---
What architectural innovation did ResNet introduce in 2015?
---
Skip connections `F(x) + x` that solved the degradation problem and enabled training 100+ layer networks.
---
card_id: twYJPH4R
---
Model: Training loss stays high in 100-layer plain network but 20-layer converges. Why?
---
Vanishing gradients prevent deep plain networks from optimizing effectively, despite having more capacity.
---
card_id: Rqie0d2P
---
What makes **ResNet-50** different from a plain 50-layer network?
---
Uses residual blocks with skip connections, enabling effective training where plain networks fail.
---
card_id: x62dWf9e
---
How does **U-Net** use skip connections differently than ResNet?
---
Connects encoder layers to corresponding decoder layers for segmentation, preserving spatial information.
---
card_id: jI4MFtPw
---
What is **pre-activation** in residual networks?
---
Applying batch normalization and activation before the convolution layers rather than after.
---
card_id: mE8YtBgV
---
Why is **pre-activation** beneficial in ResNets?
---
Creates cleaner gradient paths by ensuring the skip connection carries only the identity without transformations.
---
card_id: eEAW1YAR
---
Model: Gradient norm at layer 1 is 10^-8, at layer 50 is 1.0. Architecture?
---
Plain network - exponential gradient decay indicates lack of skip connections.
---
card_id: 2oktOj0c
---
What happens to the loss landscape when **residual connections** are added?
---
The optimization surface becomes smoother with fewer local minima, making training easier.
---
card_id: tEu95WWS
---
How do **residual connections** affect loss landscape geometry?
---
Create smoother, more convex optimization surfaces compared to the chaotic landscapes of plain deep networks.
---
card_id: s5ip4bgn
---
What is the relationship between **depth** and **gradient magnitude** in ResNets?
---
Gradients remain strong (~0.5-1.0) even at depth 100+, unlike plain networks where they decay exponentially.
---
card_id: g4ysUMIC
---
Model: Network trains well, then performance degrades when adding 20 more plain layers. Solution?
---
Use residual blocks for the additional layers to prevent degradation from vanishing gradients.
