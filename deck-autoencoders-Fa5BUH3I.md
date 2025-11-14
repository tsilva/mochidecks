---
card_id: bWVyLuVI
---
What is an **autoencoder**?
---
A neural network that learns to compress data into a compact representation and then reconstruct the original input.
---
card_id: LDi20vSi
---
What does an **autoencoder** learn to do?
---
Compress input data into a smaller representation while preserving the information needed for reconstruction.
---
card_id: 2Gh7KTVe
---
When should you use **autoencoders**?
---
- Dimensionality reduction (non-linear alternative to PCA)
- Feature learning without labels
- Denoising images
- Anomaly detection
- Data compression
---
card_id: zeuKK4dH
---
How does an **autoencoder** compress and reconstruct data?
---
1. Encoder compresses input to latent representation
2. Bottleneck limits dimensions (forces compression)
3. Decoder reconstructs original from latent code
---
card_id: ceEXt0ua
---
How do **autoencoders** differ from **PCA** for dimensionality reduction?
---
**Autoencoders**: Non-linear transformations, neural network-based / **PCA**: Linear transformations, matrix decomposition
---
card_id: 2qnJjPQ5
---
What is the **encoder-decoder architecture**?
---
A neural network structure where the encoder compresses input to low-dimensional latent space, and the decoder reconstructs the input from this representation.
---
card_id: hW6sMwAz
---
What does the **encoder** component in an autoencoder do?
---
Compresses high-dimensional input into a low-dimensional latent representation (the bottleneck).
---
card_id: VqwiKgBR
---
What does the **decoder** component in an autoencoder do?
---
Reconstructs the original high-dimensional input from the compressed latent representation.
---
card_id: lj2FVR3z
---
How does the **bottleneck** force useful learning in autoencoders?
---
By limiting dimensions, the bottleneck forces the network to learn only the most essential features needed for reconstruction, discarding redundant information.
---
card_id: 8D3XwsWB
---
What is the **latent space** in an autoencoder?
---
The compressed low-dimensional representation learned by the encoder that captures essential features of the input data.
---
card_id: BFbkhQUq
---
What property do similar inputs have in **autoencoder latent space**?
---
Similar inputs cluster together and map to nearby points in the latent space.
---
card_id: sLk9YSKH
---
Why is **latent space** useful for interpolation?
---
Points in latent space can be smoothly interpolated to generate meaningful intermediate representations that decode to realistic outputs.
---
card_id: kKlTXUdv
---
How does the **bottleneck** create structured latent space?
---
To reconstruct similar inputs well, the encoder must map them to similar latent codes, forcing the network to organize the space by semantic similarity.
---
card_id: Af8yXyUW
---
What is **reconstruction loss**?
---
A measure of the difference between the original input and the autoencoder's reconstruction of that input.
---
card_id: muGMAlwq
---
What does **reconstruction loss** measure in autoencoders?
---
How well the autoencoder can reconstruct the original input from its compressed latent representation.
---
card_id: 4oeoUywD
---
What is the formula for **MSE reconstruction loss**?
---
$$\mathcal{L}_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{x}_i)^2$$

where $x_i$ is the input and $\hat{x}_i$ is the reconstruction.
---
card_id: AaOEagQE
---
Which loss function works better for images with values in [0,1]?
---
Binary Cross-Entropy (BCE) often works better because it treats each pixel as a probability.
---
card_id: 2WppLAhp
---
How does **MSE** differ from **BCE** for autoencoder reconstruction?
---
**MSE**: Treats pixels as continuous values, penalizes squared differences / **BCE**: Treats pixels as probabilities, penalizes probabilistic divergence
---
card_id: XV5CAVgF
---
What is **latent space interpolation**?
---
Creating intermediate representations by linearly combining two latent vectors to generate smooth transitions between their decoded outputs.
---
card_id: 6RtktKxe
---
How do you interpolate between two images in **latent space**?
---
1. Encode both images to get $z_1$ and $z_2$
2. Create intermediate vectors: $z_t = (1-t) \cdot z_1 + t \cdot z_2$
3. Decode each $z_t$ to generate interpolated images
---
card_id: 5sCk3svJ
---
What does interpolation reveal about **latent space structure**?
---
Whether the latent space is continuous and well-structured, with smooth paths producing realistic intermediate outputs.
---
card_id: SPL2o0Uf
---
What is a **denoising autoencoder**?
---
An autoencoder trained on noisy inputs to reconstruct clean outputs, learning to remove noise while preserving important features.
---
card_id: AV3wtMT5
---
How does **denoising autoencoder** training differ from regular autoencoders?
---
**Denoising**: Trained with noisy inputs, clean targets / **Regular**: Trained with clean inputs, same clean targets (reconstruction)
---
card_id: kVCD3ewe
---
How does a **denoising autoencoder** remove noise?
---
The bottleneck forces the encoder to learn robust features that capture signal (digit structure) while discarding noise (random variations).
---
card_id: Dryq5jmV
---
What does high **reconstruction error** indicate in anomaly detection?
---
The input is anomalous or unusual, as the autoencoder cannot reconstruct it well using patterns learned from normal data.
---
card_id: nfmhXDI9
---
How do **autoencoders** detect anomalies?
---
Normal data reconstructs with low error, anomalous data reconstructs with high error, allowing threshold-based detection.
---
card_id: kysShcOV
---
When using **autoencoders for anomaly detection**, what data should you train on?
---
Train only on normal data so the model learns normal patterns and produces high reconstruction error for anomalies.
---
card_id: zIAmC1Pq
---
What is the **compression ratio** in an autoencoder?
---
The ratio of input dimensions to latent dimensions, indicating how much the data is compressed.
---
card_id: l3L1Bqim
---
An autoencoder compresses 784 dimensions to 32. What is the **compression ratio**?
---
24.5× compression (784 ÷ 32 = 24.5)
---
card_id: rLaF1xlO
---
What are the main applications of **autoencoders**?
---
Dimensionality reduction, feature learning, denoising, anomaly detection, compression, and foundation for VAEs.
---
card_id: Bt4BDjXm
---
Model has low reconstruction error on training data but high error on new similar data. Problem?
---
Overfitting - the autoencoder memorized training examples rather than learning general features for reconstruction.
---
card_id: iNPlhrTz
---
Autoencoder with 2-dimensional latent space vs 128-dimensional. Trade-off?
---
**2D**: High compression, more information loss, easier to visualize / **128D**: Less compression, better reconstruction, harder to visualize
---
card_id: pY3C3Uhl
---
What is the training objective for **autoencoders**?
---
Minimize reconstruction error between input and output, typically using MSE or binary cross-entropy loss.
---
card_id: dnKUDB8x
---
Why do **autoencoder** reconstructions sometimes look blurry?
---
The bottleneck forces lossy compression, discarding fine details that aren't essential for minimizing reconstruction loss.
---
card_id: uTty6bOu
---
What does **unsupervised learning** mean for autoencoders?
---
Autoencoders learn useful representations from data structure alone without requiring labeled training examples.
---
card_id: cTys0OTP
---
Why use **sigmoid activation** in the autoencoder decoder output?
---
Ensures output values are in [0, 1] range, matching the normalized input image pixel values.
---
card_id: usFasUFz
---
What advantage do **convolutional autoencoders** have over fully-connected ones for images?
---
Convolutional layers preserve spatial structure and use fewer parameters while learning better local features for images.
---
card_id: dZWTwihk
---
Need to compress medical images for storage while preserving diagnostic features. Solution?
---
Train an autoencoder on medical images, use the encoder for compression and decoder for reconstruction when needed.
---
card_id: re0tFUIZ
---
What limitation do basic **autoencoders** have for generating new samples?
---
Latent space can have "holes" (regions that decode poorly), making it unreliable for sampling new realistic outputs.
---
card_id: KY8Mw214
---
How do **Variational Autoencoders (VAEs)** improve on basic autoencoders?
---
VAEs add probabilistic structure to the latent space, eliminating holes and enabling reliable generation of new samples.
---
card_id: JdxvzAUd
---
Why use **t-SNE** or **PCA** for latent space visualization?
---
High-dimensional latent spaces (e.g., 32D) cannot be visualized directly, so dimensionality reduction to 2D enables visual inspection of clustering and structure.
---
card_id: DcP6IVAb
---
How does **t-SNE** differ from **PCA** for latent space visualization?
---
**t-SNE**: Non-linear dimensionality reduction, preserves local structure / **PCA**: Linear dimensionality reduction, preserves global variance
---
card_id: Kydjyf9S
---
When should you use **t-SNE** vs **PCA** for visualizing latent space?
---
**t-SNE**: When local clustering patterns matter / **PCA**: When global linear structure and explained variance matter
---
card_id: rug8RyJW
---
What does **t-SNE visualization** reveal about autoencoder latent space?
---
Similar data points cluster together, showing the autoencoder learned semantic structure without labels (unsupervised learning).
---
card_id: ALjFLWlZ
---
Why can **latent space interpolations** produce unrealistic images?
---
The latent space is not perfectly structured, and some intermediate regions may not correspond to realistic data points.
---
card_id: 9pxjoA2m
---
What limitation motivates **Variational Autoencoders** over basic autoencoders?
---
Basic autoencoders can have "holes" in latent space where interpolations produce unrealistic outputs, while VAEs explicitly structure the space.
---
card_id: Em7AVqYG
---
How do you add **Gaussian noise** for denoising autoencoder training?
---
Add random noise from a normal distribution scaled by a noise factor: `noisy = clean + noise_factor * random_noise`, then clamp to [0, 1].
---
card_id: 6WMHAymg
---
What does the **noise factor** control in denoising autoencoders?
---
The strength or amount of Gaussian noise added to training inputs (higher values = more noise to remove).
---
card_id: Uzd4L09p
---
Why **clamp** noisy values to [0, 1] in denoising autoencoders?
---
To ensure pixel values remain in the valid range after adding Gaussian noise, which can produce values outside [0, 1].
---
card_id: KDZ44uCM
---
What is **symmetric architecture** in autoencoders?
---
When the decoder mirrors the encoder structure (e.g., encoder: 784→512→256→32, decoder: 32→256→512→784).
---
card_id: avBwOsqc
---
Is **symmetric architecture** required for autoencoders?
---
No, it's common but not required. The decoder can have a different structure than the encoder.
---
card_id: YjDdqh6c
---
What type of learning do **autoencoders** perform?
---
Unsupervised learning and self-supervised learning (learning useful representations from data structure without labels).
---
card_id: ZekqWDn3
---
How are **autoencoders** used in self-supervised learning?
---
Pretraining encoders on unlabeled data to learn useful feature representations that can be transferred to downstream supervised tasks.
---
card_id: L47Kg0ob
---
What happens with very small **latent dimensions** (e.g., 2D)?
---
Extreme compression causes information loss and poor reconstruction, but enables direct 2D visualization without t-SNE.
---
card_id: LJdN0K7S
---
What happens with very large **latent dimensions** (e.g., 128D)?
---
Better reconstruction quality but less compression, and the model may memorize rather than learn general features.
---
card_id: S0gDDexf
---
What is the formula for **BCE reconstruction loss**?
---
$$\mathcal{L}_{BCE} = -\sum_{i=1}^{n}[x_i \log(\hat{x}_i) + (1-x_i)\log(1-\hat{x}_i)]$$

where $x_i$ is the input and $\hat{x}_i$ is the reconstruction.
---
card_id: GPLgLjRk
---
What does the **ReLU activation function** do in autoencoders?
---
Introduces non-linearity by outputting max(0, x), allowing the network to learn complex non-linear patterns and transformations.
---
card_id: GGOxnwe1
---
Why use **ReLU activations** in autoencoder hidden layers?
---
Non-linearity enables learning complex patterns beyond what linear transformations can represent, while being computationally efficient.
---
card_id: 2KuMrNwU
---
How do you compute **compression ratio** in an autoencoder?
---
$$\text{Compression Ratio} = \frac{\text{Input Dimensions}}{\text{Latent Dimensions}}$$

For example: 784 ÷ 32 = 24.5× compression
---
card_id: gNadiFLv
---
How should you choose the **latent dimension size**?
---
Balance compression and reconstruction quality: too small loses information, too large may memorize rather than compress. Experiment to find optimal trade-off for your task.
---
card_id: vdjBMQ3G
---
What does **sigmoid activation** do in the decoder output layer?
---
Maps unbounded values to [0, 1] range using $$\sigma(x) = \frac{1}{1 + e^{-x}}$$, matching normalized image pixel values.
---
card_id: Rrfy7efz
---
When should you use **sigmoid** vs **tanh** for autoencoder output?
---
**Sigmoid**: When data is normalized to [0, 1] (e.g., images) / **Tanh**: When data is normalized to [-1, 1]
---
card_id: fE3HiINd
---
How do you evaluate **autoencoder quality** quantitatively?
---
Measure reconstruction error (MSE or BCE) on test set. Compare errors across architectures and track whether error plateaus during training.
---
card_id: 7nrA9TF7
---
Why does **parameter count** matter in autoencoders?
---
Affects training time, memory usage, and risk of overfitting. More parameters increase capacity but may memorize training data.
---
card_id: 8J99kTTa
---
How does **network depth** affect autoencoder performance?
---
Deeper networks can learn more complex features but require more training time and parameters. May overfit on small datasets.
---
card_id: j1h3aqdm
---
What advantage do **pretrained autoencoder encoders** have for supervised tasks?
---
Transfer learning: The encoder provides useful feature representations learned from unlabeled data, improving performance when labeled data is limited.
---
card_id: uTgGIcsk
---
How do **autoencoders** differ from **supervised learning** for feature learning?
---
**Autoencoders**: Learn features from unlabeled data using reconstruction objective / **Supervised**: Learn features from labeled data using task-specific loss
---
card_id: x5PPhuPX
---
Need to reduce 10,000-dimensional data to 50 dimensions for visualization and clustering. Approach?
---
Train an autoencoder with 50-dimensional latent space, then use the encoder to compress all data points for downstream analysis.
---
card_id: tooNqMAU
---
Autoencoder reconstructs training images perfectly but test images poorly. Problem?
---
Overfitting - the model memorized training examples. Solutions: increase latent bottleneck compression, add regularization, or use dropout.
---
card_id: MbDfqjTb
---
How does **Gaussian likelihood** encourage blur in VAEs?
---
VAEs assume $$p(x|z) = \mathcal{N}(x; \hat{x}(z), \sigma^2 I)$$, equivalent to MSE loss. MSE rewards averaging over plausible outputs, producing smooth blurry reconstructions instead of sharp alternatives.
---
card_id: HSEeL1iC
---
How does **KL divergence** contribute to VAE blurriness?
---
The term $$D_{KL}(q(z|x) \| p(z))$$ forces latent representations to be smooth and continuous, reducing capacity to encode precise pixel-level information like edges and textures.
---
card_id: gxnQ11Ho
---
How does the **reparameterization trick** promote blur in VAEs?
---
The trick injects noise: $$z = \mu + \sigma \odot \epsilon$$ where $$\epsilon$$ is random. Because of this randomness, the decoder learns to reconstruct well on average, promoting blurred outputs.
---
card_id: uJC0HAK5
---
Why can't standard **VAE decoders** produce sharp images?
---
Most VAEs use simple pixel-wise Gaussian decoders that cannot model complex, multimodal pixel distributions, so they average outputs and produce blur.
---
card_id: 5JWFbZBZ
---
How can you reduce **blurriness in VAE reconstructions**?
---
- VQ-VAE / VQ-VAE-2 (discrete latents)
- β-VAE with tuned β parameter
- Hierarchical VAEs
- Hybrid VAE-GAN models
- Autoregressive decoders (PixelCNN)
- Diffusion-VAE hybrids
---
