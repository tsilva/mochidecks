---
card_id: 2kiLoeDd
---
What are the characteristics of **L1 (Lasso) regularization**?
---
- Produces **sparse models** (drives many weights to exactly zero)
- Performs automatic **feature selection** (eliminates irrelevant features)
- Less smooth optimization (not differentiable at zero)
---
---
card_id: 96nV20bO
---
How do **convolutional layers** contribute to translation invariance?
---
Same filters (kernels) are applied across the entire image through **parameter sharing**.
---
---
card_id: LN4bCVK5
---
How do **pooling layers** contribute to translation invariance?
---
By summarizing regions, pooling creates tolerance to small spatial shifts.
---
---
card_id: Y1vuPj96
---
How does **hierarchical architecture** contribute to translation invariance in CNNs?
---
Learns features progressively from local patterns (edges, textures) to global structure (shapes, objects), building position-independent representations at each layer.
---
---
card_id: Ap2G6Y1Z
---
How does **batch normalization** enable faster training?
---
Allows use of **higher learning rates** without diverging by preventing gradients from becoming too large or too small.
---
---
card_id: w2knuDZX
---
How does **batch normalization** act as regularization?
---
Slight noise from using different batch statistics during training (each mini-batch has slightly different mean/variance) adds randomness that reduces overfitting.
---
---
card_id: 1iXPivNz
---
How does **batch normalization** reduce sensitivity to initialization?
---
By normalizing layer inputs, even poor initial weights can be quickly adjusted during training.
---
---
card_id: E5FTs0pm
---
When would you use the **ReLU** activation function?
---
**Default choice for hidden layers** in modern deep networks.
---
---
card_id: otpBB2Jm
---
What are the **advantages** of the ReLU activation function?
---
- **Fast computation**: Simple threshold operation $\max(0, x)$
- **Avoids vanishing gradients**: Gradient is either 0 or 1
- **Sparse activation**: Many neurons output 0, creating efficient representations
---
---
card_id: pqrWuW8t
---
What is the **"dying ReLU" problem**?
---
A drawback of ReLU where neurons can get stuck always outputting 0 (when input is always negative). Once "dead", these neurons stop learning because their gradient is always zero.
---
---
card_id: IXlVHAM3
---
What are solutions for training models on **imbalanced datasets**?
---
Use resampling techniques (oversampling minority/undersampling majority), assign higher class weights to minority class errors, or generate synthetic examples (e.g., SMOTE).
---
---
card_id: KWgsQacj
---
What is **dropout** in neural networks?
---
**Dropout** randomly deactivates neurons (sets outputs to zero) during training with probability $p$ (typically 0.5).
---
---
card_id: KfXgBQCx
---
What are the advantages of using **max pooling** layers?
---
- Reduces parameters
- Helps avoid overfitting
- Provides translation invariance
---
---
card_id: N6uBJTZ0
---
When should you choose **L1** over **L2 regularization**?
---
Choose **L1 (Lasso)** when:
- You want **feature selection** (automatic removal of irrelevant features)
- You believe many features are irrelevant
- You need a sparse, interpretable model
- Storage/computation efficiency matters (fewer non-zero weights)
---
---
card_id: Y5gL62EZ
---
What is an example of a **high precision, low recall** classifier?
---
A spam filter that rarely marks legitimate emails as spam, but also misses catching many spam emails.
---
---
card_id: I94brlLK
---
How can **reducing model complexity** help with overfitting?
---
Fewer parameters (layers, neurons, features) limit the model's capacity to memorize, forcing it to learn only the most important patterns.
---
---
card_id: PjCkzoUQ
---
How is **dropout** applied differently at test time vs training time?
---
**Training time**: Neurons randomly dropped with probability $p$

**Test time**: All neurons active, but outputs scaled by $(1-p)$ to account for more neurons being active than during training.
---
---
card_id: RiAIDyWx
---
What is the **ReLU** activation function formula?
---
$$\text{ReLU}(x) = \max(0, x)$$
---
---
card_id: XBOtRLIg
---
What are the characteristics of **L2 (Ridge) regularization**?
---
- Produces **dense models** (shrinks all weights toward zero, but rarely to exactly zero)
- Performs **feature shrinkage** (reduces impact of all features proportionally)
- Smoother optimization (differentiable everywhere)
---
---
card_id: a2dKNgsc
---
What is **batch normalization**?
---
**Batch normalization** normalizes layer inputs to have mean 0 and variance 1 within each mini-batch.
---
---
card_id: 8AeZNfcx
---
What is the **batch normalization** formula?
---
$$\hat{x} = \frac{x - \mu_{batch}}{\sqrt{\sigma^2_{batch} + \epsilon}}$$

where $\mu_{batch}$ and $\sigma^2_{batch}$ are the mean and variance of the current mini-batch.
---
---
card_id: jcGg9U6M
---
What is the key effect of **L1 regularization** on model weights?
---
Drives some weights to **exactly zero**, performing automatic **feature selection** and creating **sparse models**. Useful when you have many irrelevant features.
---
---
card_id: axqKSiqY
---
When would you use the **tanh** activation function?
---
**Hidden layers** when you need zero-centered outputs (range -1 to 1).
---
---
card_id: NiVGVF0v
---
What is the **tanh** activation function formula?
---
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

Outputs values in range **-1 to 1** (zero-centered).
---
---
card_id: IBx0F6J0
---
What is the **drawback** of the tanh activation function?
---
Still suffers from the **vanishing gradient problem** in very deep networks, similar to sigmoid (gradients become very small for large positive or negative inputs).
---
---
card_id: NGJ3vvkZ
---
What are examples where **precision** should be prioritized over **recall**?
---
- Spam filtering (annoying to lose legitimate emails)
- Recommendation systems (bad recommendations hurt user experience)
- Criminal justice (wrongly convicting innocent people)
---
---
card_id: pMZK77F4
---
What is **translation invariance**?
---
**Translation invariance** is the ability to recognize a feature regardless of its position in the input.
---
---
card_id: uKMseWHY
---
What is an example of **translation invariance** in CNNs?
---
A CNN can detect a cat whether it's in the top-left or bottom-right of an image - the spatial location doesn't matter.
---
---
card_id: pfiGwlny
---
What do **convolutional layers** do in a neural network?
---
Apply learned filters (kernels) that slide across the input, detecting local patterns like edges, textures, and shapes.
---
---
card_id: ZF2WFN9M
---
What is **parameter sharing** in convolutional layers?
---
The same filter is applied across all positions in the input.
---
---
card_id: NptxfSGz
---
What is **local connectivity** in convolutional layers?
---
Each neuron connects only to a small region of the input (receptive field) rather than the entire input.
---
---
card_id: rA8ttBZx
---
How does **dropout** prevent **co-adaptation** of neurons?
---
By randomly dropping neurons, dropout prevents neurons from relying on specific other neurons always being present. Forces each neuron to learn robust features independently.
---
---
card_id: RiLlpkrC
---
How does **dropout** create an ensemble learning effect?
---
Each training iteration uses a different random subset of neurons, effectively training many different subnetworks. At test time, using all neurons approximates averaging these subnetworks' predictions.
---
---
card_id: QodC4mNG
---
Why does **dropout** force redundancy in neural networks?
---
Since any neuron might be dropped, multiple neurons must learn to detect the same important features. This redundancy makes the network more robust and less likely to overfit to specific neuron combinations.
---
---
card_id: t5IIfe3T
---
When would you use the **sigmoid** activation function?
---
**Output layer** for **binary classification**.
---
---
card_id: dBiU2RDW
---
What is the **sigmoid** activation function formula?
---
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Outputs values between **0 and 1**.
---
---
card_id: xZvfKOc7
---
What is the **drawback** of using sigmoid in hidden layers?
---
Suffers from **vanishing gradients** - gradients become very small for large positive or negative inputs, slowing or stopping learning in deep networks. Not recommended for hidden layers.
---
---
card_id: cuvmcyiq
---
What is the key effect of **L2 regularization** on model weights?
---
Shrinks all weights toward zero (but rarely to exactly zero), preventing any single feature from dominating. Creates **smoother, more stable models** with **dense** representations (all features retained).
---
---
card_id: 8lJAA6bT
---
When is **MAE** preferred as a loss metric?
---
**MAE** is preferred when:
- **Outliers** are present (MAE is less sensitive to outliers)
- All errors should be weighted **equally** (linear penalty)
- You want error in **same units** as target for easier interpretation

MAE treats all errors with equal weight: $|y - \hat{y}|$
---
---
card_id: kwcpMBRY
---
What is the **vanishing gradient problem**?
---
**Vanishing gradients** occur when gradients become extremely small during backpropagation, making weights update very slowly or stop learning entirely.
---
---
card_id: 2GhxVoqI
---
What is the difference between **parameters** and **hyperparameters**?
---
**Parameters**: Learned by the model during training (e.g., weights, biases in neural networks).

**Hyperparameters**: Set before training and control the learning process (e.g., learning rate, number of layers, regularization strength λ, number of trees in random forest).
---
---
card_id: n5F6zJ00
---
What is a **mini-batch** in training?
---
A **mini-batch** is a small subset of the training data used to compute one gradient update during training.
---
---
card_id: 2TU18FW3
---
What are typical **mini-batch** sizes?
---
32, 64, 128, 256 samples
---
---
card_id: azECjrm6
---
What are the advantages of **mini-batch** training?
---
- Faster than processing full dataset per update
- More stable than single-example updates
- Enables efficient parallel computation on GPUs
---
---
card_id: DZTZ6CM4
---
How does **L2 regularization** help with **multicollinearity**?
---
L2 regularization distributes weights across correlated features rather than putting all weight on one.
---
---
card_id: tLzMaETx
---
What is a **receptive field** in convolutional networks?
---
The **receptive field** is the region of the input that affects a particular neuron's activation.
---
---
card_id: P70AZoOs
---
How do **receptive fields** differ across CNN layers?
---
- Early layers: Small receptive fields (local patterns like edges)
- Deeper layers: Larger receptive fields (global patterns like shapes/objects)
---
---
card_id: vHPb54CB
---
How does stacking convolutional layers affect the **receptive field**?
---
Receptive field grows as you stack more convolutional layers, allowing neurons to "see" larger portions of the input.
---
---
card_id: zU2gnz2Z
---
What is the **softmax** activation function?
---
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

**Softmax** converts a vector of values into a probability distribution.
---
---
card_id: luDtAq8K
---
What are the properties of **softmax**?
---
- Outputs sum to 1
- All outputs between 0 and 1
- Emphasizes the largest value
---
---
card_id: REQNgnFU
---
When is **softmax** used in neural networks?
---
Output layer for **multi-class classification** (3+ classes).
---
---
card_id: QydEx1Yk
---
What is the **Leaky ReLU** activation function?
---
$$\text{Leaky ReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

where $\alpha$ is a small constant (e.g., 0.01).

**Leaky ReLU** allows a small negative slope instead of zero for negative inputs.
---
---
card_id: 7SSCNALh
---
What is the advantage of **Leaky ReLU** over regular ReLU?
---
Solves the dying ReLU problem - neurons can still learn even when inputs are negative.
---
---
card_id: IkstoadS
---
What is an **epoch** in training?
---
An **epoch** is one complete pass through the entire training dataset.
---
---
card_id: fogyKaEy
---
Why do models train for multiple **epochs**?
---
Model needs to see examples multiple times to learn patterns effectively. Number of epochs is a hyperparameter.
---
---
card_id: jTjjyJkj
---
What is **backpropagation**?
---
**Backpropagation** is the algorithm used to compute gradients of the loss function with respect to model parameters.
---
---
card_id: odVFhb1i
---
How does **backpropagation** work?
---
1. Forward pass: Compute predictions and loss
2. Backward pass: Compute gradients by applying chain rule backward through layers
3. Update weights using gradients
---
---
card_id: U8da8SZi
---
What is **gradient descent**?
---
**Gradient descent** is an optimization algorithm that iteratively adjusts parameters to minimize the loss function.

**Update rule**: $w_{new} = w_{old} - \alpha \nabla L(w)$

where:
- $\alpha$: learning rate
- $\nabla L(w)$: gradient of loss with respect to weights
---
---
card_id: 46tJr1fo
---
What are the main **gradient descent** variants?
---
Batch GD, Stochastic GD (SGD), Mini-batch GD.
---
---
card_id: vqpUWcWe
---
What is **learning rate** in gradient descent?
---
**Learning rate** ($\alpha$) controls the step size when updating parameters during optimization.
---
---
card_id: hWeUjPXO
---
What happens with **too high** vs **too low** learning rates?
---
**Too high**: Training is unstable, may overshoot minimum, may diverge

**Too low**: Training is very slow, may get stuck in local minima
---
---
card_id: a5L3wP0o
---
What are common **learning rate** values?
---
0.001, 0.01, 0.1
---
---
card_id: BbAmc3iV
---
What is **batch gradient descent**?
---
**Batch gradient descent** computes the gradient using the **entire training dataset** before making one parameter update (once per epoch).
---
---
card_id: QXDLMfNl
---
What are the advantages and disadvantages of **batch gradient descent**?
---
**Advantages**: Stable, smooth convergence

**Disadvantages**: Very slow for large datasets, requires all data in memory
---
---
card_id: H1xqU7vY
---
What is **stochastic gradient descent (SGD)**?
---
**Stochastic gradient descent** computes the gradient using **one random training example** at a time (once per training example).
---
---
card_id: PIRKmMCj
---
What are the advantages and disadvantages of **stochastic gradient descent (SGD)**?
---
**Advantages**: Fast updates, can escape local minima due to noise

**Disadvantages**: Noisy updates, erratic convergence path
---
---
card_id: acNElkYI
---
What is **mini-batch gradient descent**?
---
**Mini-batch gradient descent** computes the gradient using a **small batch** of training examples (e.g., 32, 64, 128).
---
---
card_id: mqENoiTI
---
What are the advantages of **mini-batch gradient descent**?
---
- Faster than batch GD
- More stable than SGD
- Efficient GPU utilization
- Most commonly used in practice
---
---
card_id: y8PrSSwI
---
What is an **activation function**?
---
An **activation function** introduces non-linearity into neural networks by transforming a neuron's weighted input.
---
---
card_id: tWLEiv2b
---
Why are **activation functions** needed in neural networks?
---
Without activation functions, even deep networks would only learn linear relationships.
---
---
card_id: zFsckFHr
---
What are common **activation function** examples?
---
ReLU, sigmoid, tanh, softmax, Leaky ReLU.
---
---
card_id: 0Q7OpQ43
---
What is an **optimizer** in machine learning?
---
An **optimizer** is an algorithm that adjusts model parameters to minimize the loss function.
---
---
card_id: fNyICQn2
---
What are common **optimizer** examples?
---
- SGD (Stochastic Gradient Descent)
- Adam (Adaptive Moment Estimation)
- RMSprop
- AdaGrad
---
---
card_id: e8KGqe0r
---
What is a **forward pass** in neural networks?
---
**Forward pass** (or forward propagation) is the process of computing predictions by passing input data through the network layers sequentially.
---
---
card_id: 8y1vGggu
---
What are the steps in a **forward pass** through a neural network?
---
1. Input enters the first layer
2. Each layer applies weights, biases, and activation functions
3. Output emerges from final layer
---
---
card_id: IbNlwYoW
---
What is the **chain rule** in backpropagation?
---
The **chain rule** from calculus allows computing gradients through composed functions.

**For neural networks**:
$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w_1}$$

Enables computing gradients by multiplying partial derivatives backward through layers.
---
---
card_id: bFIJbsU5
---
What is **average pooling**?
---
**Average pooling** downsamples by taking the **average value** within each pooling window (e.g., a 2×2 window takes the mean of 4 values).
---
---
card_id: nmog7z9T
---
How does **average pooling** compare to **max pooling**?
---
Average pooling preserves overall intensity, while max pooling preserves strongest features.
---
---
card_id: PfLQ4RzS
---
How does **boosting** work?
---
1. Train initial model on data
2. Identify misclassified examples
3. Give more weight to those examples
4. Train next model focusing on hard cases
5. Combine all models with weighted voting
---
---
card_id: tOmGjZ9r
---
What are examples of **low** and **high capacity** models?
---
- Low capacity: Linear regression
- High capacity: Deep neural network with many layers
---
---
card_id: w8C2e3CL
---
What is the **training set**?
---
The **training set** is the portion of data used to train the model - to learn parameters (weights, biases) by minimizing the loss function. The model sees and learns from this data directly.
---
---
card_id: 0n4KEKOP
---
What is the **validation set** used for?
---
Choosing learning rate, regularization strength, model architecture, early stopping.
---
---
card_id: TuGtEtRP
---
What is a **neural network layer**?
---
A **layer** is a collection of neurons that process inputs together and produce outputs. Each layer typically applies: linear transformation (weights + biases) → activation function.
---
---
card_id: 6ZDIFryu
---
What are the types of **neural network layers**?
---
- **Input layer**: Receives raw features
- **Hidden layers**: Intermediate transformations
- **Output layer**: Final predictions
---
---
card_id: VhOD3ekT
---
What is the **exploding gradient problem**?
---
**Exploding gradients** occur when gradients become extremely large during backpropagation, causing unstable training.
---
---
card_id: PmKl4R7g
---
What causes the **exploding gradient problem**?
---
- Deep networks with poor initialization
- Weights that amplify signals through layers
- Certain activation functions
---
---
card_id: r4HoxfKH
---
What are consequences of the **exploding gradient problem**?
---
- Weights update too drastically
- Training diverges (loss becomes NaN)
- Model fails to converge
---
---
card_id: Bm9pamfC
---
What are solutions to the **exploding gradient problem**?
---
Gradient clipping, proper initialization, batch normalization.
---
---
card_id: PnHbz8zJ
---
What is **gradient clipping**?
---
**Gradient clipping** limits the magnitude of gradients during training to prevent exploding gradients.
---
---
card_id: SIvSjQpC
---
What are the **gradient clipping** methods?
---
- **Clip by value**: Cap gradients at threshold (e.g., [-5, 5])
- **Clip by norm**: Scale gradient vector if its norm exceeds threshold
---
---
card_id: mqPH4KbK
---
When should you use **gradient clipping**?
---
Recurrent neural networks, very deep networks, when you observe exploding gradients.
---
---
card_id: p8NVrN3o
---
What are **weights** in neural networks?
---
**Weights** are learnable parameters that determine the strength of connections between neurons.
---
---
card_id: N4D5F4C9
---
What is **bias** (the parameter) in neural networks?
---
**Bias** is a learnable parameter added to the weighted sum before the activation function.

$$y = f(w_1x_1 + w_2x_2 + ... + w_nx_n + b)$$
---
---
card_id: 4Vn2Hdjs
---
What is the purpose of the **bias** parameter in neural networks?
---
Allows the activation function to shift left or right, increasing model flexibility. Without bias, a neuron with ReLU would always output 0 when all inputs are 0.
---
---
card_id: 7i0phiJv
---
What is the difference between **bias** (the parameter) and **bias** (the error)?
---
**Bias (parameter)**: A learnable value added to weighted sums in neural networks (notation: $b$).

**Bias (error)**: The systematic error when a model's average predictions miss the true function - a measure of underfitting.

Same word, completely different meanings - context determines which one is meant.
---
---
card_id: 0Wk1TIQr
---
When should you use **normalization** (Min-Max scaling)?
---
You need bounded values (e.g., for neural networks with sigmoid/tanh).
---
---
card_id: EuCNcuYm
---
What metric is best for detecting when a model is good for **ranking search results**, where top results matter more than lower-ranked ones?
---
**NDCG (Normalized Discounted Cumulative Gain)** - accounts for both relevance and position, with higher weight on top-ranked results.
---
---
card_id: 4kSFetKT
---
What metric is best for detecting when a model is good for **binary classification with imbalanced classes**, accounting for per-class accuracy?
---
**Balanced Accuracy** - averages recall across classes, giving equal weight to each class regardless of size.
---
---
card_id: ldCT7lDi
---
What is **harmonic mean**?
---
The **harmonic mean** is the reciprocal of the arithmetic mean of reciprocals, giving more weight to smaller values.
---
---
card_id: 33n5fG7G
---
When should you use **dropout**?
---
Use **dropout** when:
- Deep networks show signs of overfitting
- Training and validation error gap is large
- You have limited training data
- Training ensemble-like behavior is desired
---
---
card_id: 9dJ3I9aI
---
When should you use **batch normalization**?
---
Use **batch normalization** when:
- Training deep networks (10+ layers)
- You want faster convergence
- Experiencing internal covariate shift
- Need some regularization effect
---
---
card_id: FV2Pn2Kh
---
How is **batch normalization** applied differently at test time vs training time?
---
**Training**: Uses current mini-batch statistics (mean and variance)

**Test**: Uses running averages of mean and variance computed during training (fixed statistics)
---
---
card_id: 2hIwt9d2
---
How does **max pooling** work step-by-step?
---
1. Divide input into non-overlapping regions (e.g., 2×2 windows)
2. Take the maximum value from each region
3. Output the max values, reducing spatial dimensions
---
---
card_id: Xvi9k4mM
---
When should you use **max pooling** vs **average pooling**?
---
**Max pooling**: When you want to detect presence of features (dominant signal)

**Average pooling**: When you want to preserve overall information and reduce noise
---
---
card_id: dRyvIBOR
---
What is a **kernel (filter)** in convolutional layers?
---
A **kernel** is a small learnable weight matrix (e.g., 3×3) that slides across the input to detect specific patterns like edges, textures, or shapes.
---
---
card_id: naEM7AaV
---
How does a **convolutional filter** process an image?
---
1. Place filter (e.g., 3×3) at top-left of image
2. Compute element-wise multiplication and sum (dot product)
3. Store result in output feature map
4. Slide filter right by stride (e.g., 1 pixel), repeat
5. Continue until entire image is covered
---
---
card_id: CbV3ociH
---
Training **loss becomes NaN** during training. What are possible causes and solutions?
---
**Causes and solutions**:
- **Exploding gradients** → Use gradient clipping or lower learning rate
- **Learning rate too high** → Reduce learning rate by 10x
- **Poor weight initialization** → Use Xavier/He initialization
- **Numerical instability** → Add batch normalization or check for division by zero
---
---
card_id: kqu7riud
---
**Validation loss** stops decreasing at epoch 5 but **training loss** keeps dropping. What should you do?
---
**Overfitting detected** - use early stopping to stop training around epoch 5, or add regularization (dropout, L1/L2, data augmentation) and retrain.
---
---
card_id: la8zY9yL
---
**Training loss** oscillates wildly and never converges. What's the likely cause?
---
**Learning rate is too high** - the optimizer overshoots the minimum with each step. Solution: Reduce learning rate by 10x (e.g., 0.1 → 0.01).
---
---
card_id: qYAfqnzI
---
**Training loss** decreases extremely slowly (0.001 per epoch). What's likely wrong?
---
**Learning rate is too low** - steps are too small to reach the minimum efficiently. Solution: Increase learning rate by 10x.
---
---
card_id: tXoAhkAl
---
You have 1000 features but suspect only 50 are relevant. Should you use **L1** or **L2 regularization**?
---
**Use L1 (Lasso)** - it performs automatic feature selection by driving irrelevant feature weights to exactly zero, creating a sparse model with ~50 non-zero weights.
---
---
card_id: i4YfDU8Q
---
Why does **L1 regularization** drive weights to exactly zero while **L2** does not?
---
L1 uses absolute value $|w|$ which has constant gradient (±1), pushing weights by fixed amounts toward zero regardless of size. L2 uses $w^2$ with gradient proportional to $w$, so penalty weakens as weights approach zero.
---
---
card_id: wc8hG4Al
---
How do you implement **early stopping**?
---
1. Monitor validation loss each epoch
2. Track best validation loss seen so far
3. Set a **patience** parameter (e.g., 5 epochs)
4. Stop if validation loss doesn't improve for **patience** epochs
5. Restore weights from best epoch
---
---
card_id: ytrYDXIz
---
When should you use **normalization** vs **standardization**?
---
**Normalization (Min-Max)**: When you need bounded values in specific range (e.g., [0,1]) for neural networks with sigmoid/tanh

**Standardization (Z-score)**: When features have different units/scales and you want zero-centered data (more robust to outliers)
---
---
card_id: 6acnMN4c
---
Why is **ReLU** preferred over **sigmoid/tanh** for hidden layers?
---
**ReLU advantages**:
- No vanishing gradient problem (gradient is 0 or 1)
- Faster computation (simple threshold)
- Creates sparse activations (many zeros)

**Sigmoid/tanh problems**: Gradients vanish for large |x|, slowing learning in deep networks.
---
---
card_id: znraY4zM
---
What are the tradeoffs between **batch**, **stochastic**, and **mini-batch gradient descent**?
---
**Batch GD**: Stable but slow, requires all data in memory

**Stochastic GD**: Fast updates but noisy, erratic convergence

**Mini-batch GD**: Best balance - faster than batch, more stable than stochastic, efficient GPU usage
---
---
card_id: 36vzVgEA
---
What factors determine **model complexity**?
---
- Number of parameters (weights, biases)
- Network depth (number of layers)
- Network width (neurons per layer)
- Polynomial degree (for regression)
- Tree depth (for decision trees)
---