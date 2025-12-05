---
card_id: LMQkMbiQ
---
What is an **autoencoder**?
---
Neural network that learns compressed data representations through unsupervised reconstruction training.
---
card_id: eXG8syhc
---
What does an **autoencoder** learn to do?
---
Compress high-dimensional input into compact latent representation, then reconstruct the original input from it.
---
card_id: eP3KRXw0
---
How does an **autoencoder** work?
---
1. Encoder compresses input to latent representation
2. Decoder reconstructs input from latent code
3. Network trained to minimize reconstruction error
---
card_id: Pkh3jgFV
---
What are the three components of **autoencoder** architecture?
---
- Encoder: Compresses input to latent code
- Bottleneck: Low-dimensional latent representation
- Decoder: Reconstructs input from latent code
---
card_id: JZ4dPJCR
---
What is the **training objective** for autoencoders?
---
Minimize reconstruction error between input and output.

$$\mathcal{L} = \|x - \hat{x}\|^2$$

$x$: input, $\hat{x}$: reconstruction
---
card_id: 63IaXkZt
---
What is the **latent space** in autoencoders?
---
The compressed low-dimensional representation learned by the encoder that captures essential input features.
---
card_id: RlUYDsMR
---
What is the **bottleneck** in autoencoders?
---
The layer with lowest dimensionality that forces the network to learn compressed representations.
---
card_id: zP8IvVDG
---
How does the **bottleneck** force learning of useful features?
---
Limited capacity prevents memorization, forcing the network to extract only the most essential features for reconstruction.
---
card_id: HAxamHGM
---
Why do similar inputs **cluster in latent space** without labels?
---
The encoder maps similar inputs to nearby latent codes because this minimizes reconstruction error efficiently.
---
card_id: XmU7tRSf
---
What is **compression ratio** in autoencoders?
---
Ratio of input dimensions to latent dimensions (e.g., 784-to-32 gives 24.5× compression).
---
card_id: wSrhfpxI
---
How does **MSE** compare to **BCE** for autoencoder loss?
---
**MSE**: Squared differences, continuous values, simpler

**BCE**: Probabilistic interpretation, better for [0,1] images
---
card_id: n5mQZVz0
---
When should you use **BCE vs MSE** for autoencoder loss?
---
- **BCE**: Images normalized to [0,1], probabilistic modeling
- **MSE**: Continuous data, unbounded values, simpler optimization
---
card_id: OQnWXANN
---
What is **latent space interpolation**?
---
Creating smooth transitions between inputs by linearly blending their latent representations and decoding.
---
card_id: w0TGzA5z
---
How do you perform **latent space interpolation**?
---
1. Encode inputs to get $z_1$ and $z_2$
2. Create intermediate vectors: $z_t = (1-t) \cdot z_1 + t \cdot z_2$
3. Decode each $z_t$ to generate transition images
---
card_id: c3xsf0bV
---
Why can **interpolated images** look unrealistic or blurry?
---
Basic autoencoders don't structure the latent space, so some regions may not correspond to realistic data.
---
card_id: fCYiGCj1
---
What do **latent space interpolations** reveal about the encoder?
---
- Smoothness of the learned representation
- Which latent regions decode to realistic outputs
- Overall quality of latent space organization
---
card_id: dH9bAYbO
---
What is a **denoising autoencoder**?
---
Autoencoder trained on noisy inputs with clean targets to learn noise-robust feature representations.
---
card_id: nv2CqU27
---
How does **denoising autoencoder** training differ from standard training?
---
**Standard**: Input clean, target same clean image (reconstruction)

**Denoising**: Input noisy, target clean image (noise removal)
---
card_id: VNByyqe6
---
How do **denoising autoencoders** remove noise?
---
1. Encoder extracts structural features, ignoring noise
2. Bottleneck compresses only essential signal
3. Decoder reconstructs clean image from noise-free latent code
---
card_id: 3v5rCDlN
---
Why train **denoising autoencoders** on corrupted inputs?
---
Forces the network to learn which features are signal versus noise, creating more robust representations.
---
card_id: nuDv6kf1
---
How do **autoencoders** perform anomaly detection?
---
Normal samples reconstruct well (low error), while anomalies reconstruct poorly (high error) since the network learned normal patterns.
---
card_id: KJmAmFwh
---
What training data is needed for **autoencoder anomaly detection**?
---
Train only on normal data. The network learns normal patterns, making anomalies produce high reconstruction error.
---
card_id: i37yUtqX
---
What are applications of **autoencoders**?
---
- Dimensionality reduction
- Denoising
- Anomaly detection
- Feature learning
- Data compression
- Foundation for VAEs
---
card_id: HkoteFBw
---
When should you use **autoencoders** for dimensionality reduction?
---
- Need non-linear compression (PCA only linear)
- Want task-specific learned features
- Have sufficient training data
---
card_id: 0sALilVN
---
How do **autoencoders** compare to **PCA** for dimensionality reduction?
---
**Autoencoders**: Non-linear, task-specific, requires training data

**PCA**: Linear only, no training needed, optimal for linear structure
---
card_id: zoDohNFU
---
Why use **sigmoid activation** in autoencoder decoder output?
---
Constrains output values to [0, 1] range to match normalized input images.
---
card_id: KFpa5OyS
---
Why use **ReLU activation** in autoencoder hidden layers?
---
- Enables non-linear transformations
- Prevents vanishing gradients
- Computationally efficient
- Allows learning complex patterns
---
card_id: ulM1xTcT
---
What does **symmetric architecture** mean in autoencoders?
---
Decoder mirrors encoder structure (e.g., 784→512→256→32→256→512→784), though not strictly required.
---
card_id: MoqUWTgJ
---
What are limitations of **basic autoencoders**?
---
- Unstructured latent space with holes
- Cannot generate new samples reliably
- Interpolations may be unrealistic
- No probabilistic framework
---
card_id: jIfXSR3W
---
How do **Variational Autoencoders (VAEs)** improve on standard autoencoders?
---
VAEs add probabilistic structure to latent space, ensuring all regions decode to realistic outputs and enabling generation.
---
card_id: 0qnFsqIB
---
When should you use **convolutional vs fully-connected autoencoders**?
---
- **Convolutional**: Images, spatial data, better quality, fewer parameters
- **Fully-connected**: Non-spatial data, tabular data, simpler
---
card_id: WOLM9apn
---
Autoencoder reconstructs training images well but test images poorly. What's the problem?
---
Overfitting. The autoencoder memorized training data instead of learning generalizable feature representations.
---
card_id: cFAj5nmF
---
Autoencoder with 2D latent space vs 128D latent space. What are the trade-offs?
---
**2D**: Maximum compression, direct visualization, higher error, may lose important details

**128D**: Less compression, better reconstruction, may overfit, harder to visualize
---
card_id: rJ9r1hZc
---
Latent space interpolations produce blurry or unrealistic intermediate images. Solution?
---
Use Variational Autoencoders (VAEs) which structure the latent space to ensure smooth, realistic interpolations.
---
card_id: PGosxelW
---
Autoencoder outputs all-black images during training. What could be wrong?
---
- Learning rate too high causing divergence
- Wrong activation function (ReLU instead of sigmoid at output)
- Vanishing gradients
- Loss function mismatch
---
card_id: VZV2pXKP
---
Latent dimension too small for autoencoder. What symptoms appear?
---
- High reconstruction error
- Loss of important details
- Inability to distinguish different classes
- Blurry reconstructions
---
card_id: juf527J9
---
Choosing reconstruction loss: continuous sensor data with unbounded values. Which loss function?
---
MSE (Mean Squared Error) because data is continuous and unbounded. BCE assumes values in [0,1].
---
card_id: nVhS9LfJ
---
Want to visualize autoencoder latent space directly without t-SNE. What latent dimension enables this?
---
2D or 3D latent space allows direct visualization without dimensionality reduction.
---
card_id: 3QrG7Fws
---
Need to detect manufacturing defects in product images. How to train the **autoencoder**?
---
Train only on normal (non-defective) products. Defects will have high reconstruction error during inference.
---
card_id: DNVuJ4x4
---
Autoencoder for MNIST digits: 784 input → ? → 32 latent. What hidden layer sizes are appropriate?
---
Gradually decreasing sizes like 512 → 256 → 32, allowing progressive compression while learning hierarchical features.
