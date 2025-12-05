---
card_id: a1vPZE05
---
What is a **sparse representation**?
---
A representation where only a small fraction of neurons are active at any given time, with most values near zero.
---
card_id: 9TWKasMT
---
What does **sparsity** achieve in neural networks?
---
Forces each neuron to specialize in detecting specific patterns, leading to more interpretable and disentangled features.
---
card_id: wKbHRdOB
---
When should you use **sparse representations** over dense representations?
---
When you need interpretable features, want to enable overcomplete representations, or require better generalization through forced specialization.
---
card_id: XOhuDr39
---
How does a **sparse autoencoder** enforce sparsity during training?
---
By adding penalty terms to the loss function that discourage neurons from being active, such as KL divergence or L1 regularization.
---
card_id: wpjaQLLL
---
How does **sparse representation** differ from **dense representation**?
---
**Sparse**: Only a few neurons active (mostly zeros), features specialized / **Dense**: All neurons contribute, features may be entangled.
---
card_id: gEWKoUGV
---
Model has `latent_dim > input_dim` but no sparsity constraint. Problem?
---
Learns trivial identity mapping, as the overcomplete representation has enough capacity to memorize inputs without learning useful features.
---
card_id: QZnKqhHo
---
What is **KL divergence penalty** for sparsity?
---
A penalty that encourages each neuron's average activation across samples to match a target sparsity level.
---
card_id: qd8en7TC
---
How does **KL divergence** enforce sparsity in autoencoders?
---
Penalizes neurons whose average activation $\hat{\rho}_j$ deviates from target sparsity $\rho$, discouraging neurons from being too active or inactive.
---
card_id: dhdrfK9V
---
When to use **KL divergence** vs **L1** for sparsity enforcement?
---
**KL**: For consistent sparsity across dataset, more principled / **L1**: Simpler, encourages exact zeros, works per sample.
---
card_id: qsTaxvLU
---
What is the formula for **KL divergence sparsity penalty**?
---
$$\text{KL}(\rho \| \hat{\rho}_j) = \rho \log\frac{\rho}{\hat{\rho}_j} + (1-\rho)\log\frac{1-\rho}{1-\hat{\rho}_j}$$

$\rho$: target sparsity, $\hat{\rho}_j$: average activation of neuron $j$
---
card_id: kNzzXsQp
---
What is **L1 regularization** for sparsity?
---
A penalty on the absolute values of activations that encourages them to be exactly zero: $\mathcal{L}_{L1} = \lambda \sum_j |z_j|$
---
card_id: swswCqeF
---
What does **L1 penalty** encourage in neural network activations?
---
Encourages activations to be exactly zero by penalizing any non-zero absolute values, creating perfectly sparse representations.
---
card_id: 2hSnnMLB
---
How does **L1 regularization** differ from **KL divergence** for sparsity?
---
**L1**: Simpler, exact zeros, per-sample sparsity / **KL**: Dataset-wide consistency, principled statistical measure, requires sigmoid activation.
---
card_id: hMaj6qtO
---
What are the main benefits of **sparse representations**?
---
More interpretable features, better generalization, disentangled features, efficient computation, and enables overcomplete representations.
---
card_id: 4WsiA8TK
---
Why does **sparsity** improve interpretability in neural networks?
---
Forces each neuron to specialize in detecting specific patterns, making each feature have clear meaning rather than distributed entangled representations.
---
card_id: EDzMwAvo
---
What biological principle motivates **sparse coding**?
---
Visual cortex neurons are sparse with only ~1% active at once, following efficient coding theory that minimizes energy while preserving information.
---
card_id: 5ImfM8cq
---
What is **lifetime sparsity**?
---
The fraction of samples for which a neuron is active across an entire dataset.
---
card_id: sCJa3W3p
---
What does high **lifetime sparsity** indicate about a neuron?
---
The neuron is highly selective, only activating for specific inputs or patterns rather than responding broadly.
---
card_id: f6yjQ6t2
---
What is an **overcomplete autoencoder**?
---
An autoencoder where the latent dimension is larger than the input dimension (`latent_dim > input_dim`).
---
card_id: OJzjf4bV
---
Why do **overcomplete autoencoders** require sparsity constraints?
---
Without sparsity, they learn trivial identity mapping since they have more capacity than needed, failing to learn useful compressed features.
---
card_id: 3QffJpCf
---
What advantage do **overcomplete sparse autoencoders** provide?
---
Can learn more fine-grained features than input dimensions by having many specialized detectors, each active only for specific patterns.
---
card_id: oqYjvDSS
---
How does a **sparse autoencoder** differ from a **regular autoencoder**?
---
**Sparse**: Adds sparsity constraint, can be overcomplete, specialized features / **Regular**: Uses bottleneck for compression, all neurons typically active.
---
card_id: RV58nbMb
---
How does a **regular autoencoder** prevent identity mapping?
---
Uses bottleneck architecture where latent dimension is smaller than input dimension, forcing information compression.
---
card_id: YIP9FDzM
---
How does a **sparse autoencoder** prevent identity mapping?
---
Adds sparsity constraints that limit the number of active neurons, forcing the model to learn efficient specialized features.
---
card_id: jO8OBHax
---
How do learned features differ between **sparse** and **dense autoencoders**?
---
**Sparse**: Localized, interpretable, specialized features / **Dense**: Distributed, entangled, all features contribute to all inputs.
---
card_id: xuB4cDon
---
What are **selective neurons** in sparse autoencoders?
---
Neurons that activate strongly for specific input patterns or classes while remaining inactive for others.
---
card_id: keqnK4Nq
---
What does high **selectivity** indicate in sparse autoencoder neurons?
---
The neuron specializes in detecting one or few specific patterns, measured by high variance in activation across different input classes.
---
card_id: iXJMaFpf
---
**Sparse autoencoder** uses sigmoid activation on latent layer. Why?
---
Keeps activation values in [0, 1] range required for KL divergence penalty computation and enables probabilistic interpretation of sparsity.
---
card_id: hwIUKrNa
---
**Sparse autoencoder** with `sparsity_target=0.01` vs `sparsity_target=0.2`. What happens?
---
Lower target (0.01) produces very sparse activations with more specialized features but may hurt reconstruction quality; higher (0.2) balances sparsity and reconstruction.
---
card_id: O54sQ3XJ
---
Training **sparse autoencoder** but all neurons equally active. Solutions?
---
Increase sparsity penalty weight, lower sparsity target, or add L1 regularization to force differentiation among neurons.
---
card_id: AIneLEWF
---
What is the total loss formula for **sparse autoencoders**?
---
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{KL}} + \lambda \mathcal{L}_{\text{L1}}$$

Reconstruction error plus weighted sparsity penalties.
---
card_id: xQBrvkvK
---
Why use **sigmoid** on latent activations in sparse autoencoders?
---
Bounds activations to [0, 1] for KL divergence computation and prevents negative values that would interfere with sparsity measurement.
---
card_id: gzIIhFkD
---
What does **sparsity target** ($\rho$) represent in sparse autoencoders?
---
The desired average activation level per neuron, typically 0.01-0.05 (1-5% average activity).
---
card_id: KifcFhcx
---
How does **sparsity** enable overcomplete representations?
---
Prevents identity mapping by limiting active neurons, allowing more features than dimensions without trivial solutions.
---
card_id: Qi2T5Nte
---
What is **activation threshold** used for in sparse autoencoders?
---
A cutoff value (e.g., 0.1 or 0.3) to determine whether a neuron is considered "active" for measuring sparsity statistics.
---
card_id: zYaORy10
---
What does **efficient coding theory** predict about neural representations?
---
Optimal neural codes should be sparse to minimize energy consumption while maximizing information preservation about the environment.
