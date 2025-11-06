---
card_id: dJfAi5BL
---
What are the characteristics of **L1 (Lasso) regularization**?
---
- Produces **sparse models** (drives many weights to exactly zero)
- Performs automatic **feature selection** (eliminates irrelevant features)
- Less smooth optimization (not differentiable at zero)
---
card_id: Kkxdn8Ic
---
How do **convolutional layers** contribute to translation invariance?
---
Same filters (kernels) are applied across the entire image through **parameter sharing**.
---
card_id: wkozY6ck
---
How do **pooling layers** contribute to translation invariance?
---
By summarizing regions, pooling creates tolerance to small spatial shifts.
---
card_id: cOYkifcw
---
How does **hierarchical architecture** contribute to translation invariance in CNNs?
---
Learns features progressively from local patterns (edges, textures) to global structure (shapes, objects), building position-independent representations at each layer.
---
card_id: WbeBj01Z
---
How does **batch normalization** enable faster training?
---
Allows use of **higher learning rates** without diverging by preventing gradients from becoming too large or too small.
---
card_id: oXjqBXuV
---
How does **batch normalization** act as regularization?
---
Slight noise from using different batch statistics during training (each mini-batch has slightly different mean/variance) adds randomness that reduces overfitting.
---
card_id: 5hox1WhD
---
How does **batch normalization** reduce sensitivity to initialization?
---
By normalizing layer inputs, even poor initial weights can be quickly adjusted during training.
---
card_id: wJUHXCRc
---
When would you use the **ReLU** activation function?
---
**Default choice for hidden layers** in modern deep networks.
---
card_id: MKojEq9g
---
What are the **advantages** of the ReLU activation function?
---
- **Fast computation**: Simple threshold operation $\max(0, x)$
- **Avoids vanishing gradients**: Gradient is either 0 or 1
- **Sparse activation**: Many neurons output 0, creating efficient representations
---
card_id: JuOOADEc
---
What is the **"dying ReLU" problem**?
---
A drawback of ReLU where neurons can get stuck always outputting 0 (when input is always negative). Once "dead", these neurons stop learning because their gradient is always zero.
---
card_id: 237Yjdhr
---
What are solutions for training models on **imbalanced datasets**?
---
Use resampling techniques (oversampling minority/undersampling majority), assign higher class weights to minority class errors, or generate synthetic examples (e.g., SMOTE).
---
card_id: UOZ2QWyh
---
What is **dropout** in neural networks?
---
**Dropout** randomly deactivates neurons (sets outputs to zero) during training with probability $p$ (typically 0.5).
---
card_id: zoao2Kqo
---
What are the advantages of using **max pooling** layers?
---
- Reduces parameters
- Helps avoid overfitting
- Provides translation invariance
---
card_id: nemUn4IV
---
When should you choose **L1** over **L2 regularization**?
---
Choose **L1 (Lasso)** when:
- You want **feature selection** (automatic removal of irrelevant features)
- You believe many features are irrelevant
- You need a sparse, interpretable model
- Storage/computation efficiency matters (fewer non-zero weights)
---
card_id: 6e1DHybK
---
What is an example of a **high precision, low recall** classifier?
---
A spam filter that rarely marks legitimate emails as spam, but also misses catching many spam emails.
---
card_id: P8qPByl8
---
How can **reducing model complexity** help with overfitting?
---
Fewer parameters (layers, neurons, features) limit the model's capacity to memorize, forcing it to learn only the most important patterns.
---
card_id: XxjmB3Uk
---
How is **dropout** applied differently at test time vs training time?
---
**Training time**: Neurons randomly dropped with probability $p$

**Test time**: All neurons active, but outputs scaled by $(1-p)$ to account for more neurons being active than during training.
---
card_id: xv0xv2n3
---
What is the **ReLU** activation function formula?
---
$$\text{ReLU}(x) = \max(0, x)$$
---
card_id: TIBt6Kij
---
What are the characteristics of **L2 (Ridge) regularization**?
---
- Produces **dense models** (shrinks all weights toward zero, but rarely to exactly zero)
- Performs **feature shrinkage** (reduces impact of all features proportionally)
- Smoother optimization (differentiable everywhere)
---
card_id: DOg5LVX6
---
What is **batch normalization**?
---
**Batch normalization** normalizes layer inputs to have mean 0 and variance 1 within each mini-batch.
---
card_id: XU8s7gVe
---
What is the **batch normalization** formula?
---
$$\hat{x} = \frac{x - \mu_{batch}}{\sqrt{\sigma^2_{batch} + \epsilon}}$$

where $\mu_{batch}$ and $\sigma^2_{batch}$ are the mean and variance of the current mini-batch.
---
card_id: dSl1wQoa
---
What is the key effect of **L1 regularization** on model weights?
---
Drives some weights to **exactly zero**, performing automatic **feature selection** and creating **sparse models**. Useful when you have many irrelevant features.
---
card_id: 1tgI1IlY
---
When would you use the **tanh** activation function?
---
**Hidden layers** when you need zero-centered outputs (range -1 to 1).
---
card_id: b0D73lOF
---
What is the **tanh** activation function formula?
---
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

Outputs values in range **-1 to 1** (zero-centered).
---
card_id: 4LZA1ZRM
---
What is the **drawback** of the tanh activation function?
---
Still suffers from the **vanishing gradient problem** in very deep networks, similar to sigmoid (gradients become very small for large positive or negative inputs).
---
card_id: aohUIxP8
---
What are examples where **precision** should be prioritized over **recall**?
---
- Spam filtering (annoying to lose legitimate emails)
- Recommendation systems (bad recommendations hurt user experience)
- Criminal justice (wrongly convicting innocent people)
---
card_id: XhaxV63y
---
What is **translation invariance**?
---
**Translation invariance** is the ability to recognize a feature regardless of its position in the input.
---
card_id: Jdg5nD8y
---
What is an example of **translation invariance** in CNNs?
---
A CNN can detect a cat whether it's in the top-left or bottom-right of an image - the spatial location doesn't matter.
---
card_id: U870jcql
---
What do **convolutional layers** do in a neural network?
---
Apply learned filters (kernels) that slide across the input, detecting local patterns like edges, textures, and shapes.
---
card_id: kCL2LPnx
---
What is **parameter sharing** in convolutional layers?
---
The same filter is applied across all positions in the input.
---
card_id: s72vPzb3
---
What is **local connectivity** in convolutional layers?
---
Each neuron connects only to a small region of the input (receptive field) rather than the entire input.
---
card_id: lYIDPZGR
---
How does **dropout** prevent **co-adaptation** of neurons?
---
By randomly dropping neurons, dropout prevents neurons from relying on specific other neurons always being present. Forces each neuron to learn robust features independently.
---
card_id: kOdaJ6IH
---
How does **dropout** create an ensemble learning effect?
---
Each training iteration uses a different random subset of neurons, effectively training many different subnetworks. At test time, using all neurons approximates averaging these subnetworks' predictions.
---
card_id: R1EpWdjF
---
Why does **dropout** force redundancy in neural networks?
---
Since any neuron might be dropped, multiple neurons must learn to detect the same important features. This redundancy makes the network more robust and less likely to overfit to specific neuron combinations.
---
card_id: CT6AB2Ae
---
When would you use the **sigmoid** activation function?
---
**Output layer** for **binary classification**.
---
card_id: ke9msKlS
---
What is the **sigmoid** activation function formula?
---
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Outputs values between **0 and 1**.
---
card_id: GtRvs2XS
---
What is the **drawback** of using sigmoid in hidden layers?
---
Suffers from **vanishing gradients** - gradients become very small for large positive or negative inputs, slowing or stopping learning in deep networks. Not recommended for hidden layers.
---
card_id: 7cuUJfm8
---
What is the key effect of **L2 regularization** on model weights?
---
Shrinks all weights toward zero (but rarely to exactly zero), preventing any single feature from dominating. Creates **smoother, more stable models** with **dense** representations (all features retained).
---
card_id: rNlydhVH
---
When is **MAE** preferred as a loss metric?
---
**MAE** is preferred when:
- **Outliers** are present (MAE is less sensitive to outliers)
- All errors should be weighted **equally** (linear penalty)
- You want error in **same units** as target for easier interpretation

MAE treats all errors with equal weight: $|y - \hat{y}|$
---
card_id: IY4QfYAs
---
What is the **vanishing gradient problem**?
---
**Vanishing gradients** occur when gradients become extremely small during backpropagation, making weights update very slowly or stop learning entirely.
---
card_id: tcly6fKe
---
What is the difference between **parameters** and **hyperparameters**?
---
**Parameters**: Learned by the model during training (e.g., weights, biases in neural networks).

**Hyperparameters**: Set before training and control the learning process (e.g., learning rate, number of layers, regularization strength λ, number of trees in random forest).
---
card_id: M2G9osRf
---
What is a **mini-batch** in training?
---
A **mini-batch** is a small subset of the training data used to compute one gradient update during training.
---
card_id: gs35dYus
---
What are typical **mini-batch** sizes?
---
32, 64, 128, 256 samples
---
card_id: Dukz3eQI
---
What are the advantages of **mini-batch** training?
---
- Faster than processing full dataset per update
- More stable than single-example updates
- Enables efficient parallel computation on GPUs
---
card_id: AvHrfmxJ
---
How does **L2 regularization** help with **multicollinearity**?
---
L2 regularization distributes weights across correlated features rather than putting all weight on one.
---
card_id: 671CYxSO
---
What is a **receptive field** in convolutional networks?
---
The **receptive field** is the region of the input that affects a particular neuron's activation.
---
card_id: B7Z44ohh
---
How do **receptive fields** differ across CNN layers?
---
- Early layers: Small receptive fields (local patterns like edges)
- Deeper layers: Larger receptive fields (global patterns like shapes/objects)
---
card_id: 942oG5CG
---
How does stacking convolutional layers affect the **receptive field**?
---
Receptive field grows as you stack more convolutional layers, allowing neurons to "see" larger portions of the input.
---
card_id: bfOSNLVV
---
What is the **softmax** activation function?
---
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

**Softmax** converts a vector of values into a probability distribution.
---
card_id: vSJ60wnd
---
What are the properties of **softmax**?
---
- Outputs sum to 1
- All outputs between 0 and 1
- Emphasizes the largest value
---
card_id: d0cPtkJN
---
When is **softmax** used in neural networks?
---
Output layer for **multi-class classification** (3+ classes).
---
card_id: evjtR5MJ
---
What is the **Leaky ReLU** activation function?
---
$$\text{Leaky ReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

where $\alpha$ is a small constant (e.g., 0.01).

**Leaky ReLU** allows a small negative slope instead of zero for negative inputs.
---
card_id: kgwq8TEW
---
What is the advantage of **Leaky ReLU** over regular ReLU?
---
Solves the dying ReLU problem - neurons can still learn even when inputs are negative.
---
card_id: yckLMpI3
---
What is an **epoch** in training?
---
An **epoch** is one complete pass through the entire training dataset.
---
card_id: TEqx6Fkh
---
Why do models train for multiple **epochs**?
---
Model needs to see examples multiple times to learn patterns effectively. Number of epochs is a hyperparameter.
---
card_id: I06Ha3Sz
---
What is **backpropagation**?
---
**Backpropagation** is the algorithm used to compute gradients of the loss function with respect to model parameters.
---
card_id: lSl2ZBaN
---
How does **backpropagation** work?
---
1. Forward pass: Compute predictions and loss
2. Backward pass: Compute gradients by applying chain rule backward through layers
3. Update weights using gradients
---
card_id: aagLmYSb
---
What is **gradient descent**?
---
**Gradient descent** is an optimization algorithm that iteratively adjusts parameters to minimize the loss function.

**Update rule**: $w_{new} = w_{old} - \alpha \nabla L(w)$

where:
- $\alpha$: learning rate
- $\nabla L(w)$: gradient of loss with respect to weights
---
card_id: hAROxBpZ
---
What are the main **gradient descent** variants?
---
Batch GD, Stochastic GD (SGD), Mini-batch GD.
---
card_id: 1UU12dKZ
---
What is **learning rate** in gradient descent?
---
**Learning rate** ($\alpha$) controls the step size when updating parameters during optimization.
---
card_id: JTemgLOH
---
What happens with **too high** vs **too low** learning rates?
---
**Too high**: Training is unstable, may overshoot minimum, may diverge

**Too low**: Training is very slow, may get stuck in local minima
---
card_id: D5jCtumO
---
What are common **learning rate** values?
---
0.001, 0.01, 0.1
---
card_id: oWyJrOZ2
---
What is **batch gradient descent**?
---
**Batch gradient descent** computes the gradient using the **entire training dataset** before making one parameter update (once per epoch).
---
card_id: QFoJnUNe
---
What are the advantages and disadvantages of **batch gradient descent**?
---
**Advantages**: Stable, smooth convergence

**Disadvantages**: Very slow for large datasets, requires all data in memory
---
card_id: JR5Tdq8b
---
What is **stochastic gradient descent (SGD)**?
---
**Stochastic gradient descent** computes the gradient using **one random training example** at a time (once per training example).
---
card_id: ngSlxRRK
---
What are the advantages and disadvantages of **stochastic gradient descent (SGD)**?
---
**Advantages**: Fast updates, can escape local minima due to noise

**Disadvantages**: Noisy updates, erratic convergence path
---
card_id: 8BhsihNj
---
What is **mini-batch gradient descent**?
---
**Mini-batch gradient descent** computes the gradient using a **small batch** of training examples (e.g., 32, 64, 128).
---
card_id: Anl7v45h
---
What are the advantages of **mini-batch gradient descent**?
---
- Faster than batch GD
- More stable than SGD
- Efficient GPU utilization
- Most commonly used in practice
---
card_id: AZ7hwI1W
---
What is an **activation function**?
---
An **activation function** introduces non-linearity into neural networks by transforming a neuron's weighted input.
---
card_id: NKUpuhf6
---
Why are **activation functions** needed in neural networks?
---
Without activation functions, even deep networks would only learn linear relationships.
---
card_id: USCBwVDE
---
What are common **activation function** examples?
---
ReLU, sigmoid, tanh, softmax, Leaky ReLU.
---
card_id: 7d7lXuJP
---
What is an **optimizer** in machine learning?
---
An **optimizer** is an algorithm that adjusts model parameters to minimize the loss function.
---
card_id: ttcPbGGR
---
What are common **optimizer** examples?
---
- SGD (Stochastic Gradient Descent)
- Adam (Adaptive Moment Estimation)
- RMSprop
- AdaGrad
---
card_id: HgdguWIL
---
What is a **forward pass** in neural networks?
---
**Forward pass** (or forward propagation) is the process of computing predictions by passing input data through the network layers sequentially.
---
card_id: aHQPBzRE
---
What are the steps in a **forward pass** through a neural network?
---
1. Input enters the first layer
2. Each layer applies weights, biases, and activation functions
3. Output emerges from final layer
---
card_id: lqXeihx3
---
What is the **chain rule** in backpropagation?
---
The **chain rule** from calculus allows computing gradients through composed functions.

**For neural networks**:
$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w_1}$$

Enables computing gradients by multiplying partial derivatives backward through layers.
---
card_id: RzxtFO8M
---
What is **average pooling**?
---
**Average pooling** downsamples by taking the **average value** within each pooling window (e.g., a 2×2 window takes the mean of 4 values).
---
card_id: Di0YVyz7
---
How does **average pooling** compare to **max pooling**?
---
Average pooling preserves overall intensity, while max pooling preserves strongest features.
---
card_id: IkthhqIT
---
How does **boosting** work?
---
1. Train initial model on data
2. Identify misclassified examples
3. Give more weight to those examples
4. Train next model focusing on hard cases
5. Combine all models with weighted voting
---
card_id: zMuUP0Lp
---
What are examples of **low** and **high capacity** models?
---
- Low capacity: Linear regression
- High capacity: Deep neural network with many layers
---
card_id: d9lRIWeH
---
What is the **training set**?
---
The **training set** is the portion of data used to train the model - to learn parameters (weights, biases) by minimizing the loss function. The model sees and learns from this data directly.
---
card_id: dgglAJVW
---
What is the **validation set** used for?
---
Choosing learning rate, regularization strength, model architecture, early stopping.
---
card_id: 3Kk7apEf
---
What is a **neural network layer**?
---
A **layer** is a collection of neurons that process inputs together and produce outputs. Each layer typically applies: linear transformation (weights + biases) → activation function.
---
card_id: Cj78C9q0
---
What are the types of **neural network layers**?
---
- **Input layer**: Receives raw features
- **Hidden layers**: Intermediate transformations
- **Output layer**: Final predictions
---
card_id: eZazZLlr
---
What is the **exploding gradient problem**?
---
**Exploding gradients** occur when gradients become extremely large during backpropagation, causing unstable training.
---
card_id: dUkVOPEb
---
What causes the **exploding gradient problem**?
---
- Deep networks with poor initialization
- Weights that amplify signals through layers
- Certain activation functions
---
card_id: G7hYrBag
---
What are consequences of the **exploding gradient problem**?
---
- Weights update too drastically
- Training diverges (loss becomes NaN)
- Model fails to converge
---
card_id: Qs8WOUEV
---
What are solutions to the **exploding gradient problem**?
---
Gradient clipping, proper initialization, batch normalization.
---
card_id: 7pY5oINF
---
What is **gradient clipping**?
---
**Gradient clipping** limits the magnitude of gradients during training to prevent exploding gradients.
---
card_id: pzUU1qGr
---
What are the **gradient clipping** methods?
---
- **Clip by value**: Cap gradients at threshold (e.g., [-5, 5])
- **Clip by norm**: Scale gradient vector if its norm exceeds threshold
---
card_id: j3AL7SpI
---
When should you use **gradient clipping**?
---
Recurrent neural networks, very deep networks, when you observe exploding gradients.
---
card_id: btr0Ha9v
---
What are **weights** in neural networks?
---
**Weights** are learnable parameters that determine the strength of connections between neurons.
---
card_id: ChP9ltHS
---
What is **bias** (the parameter) in neural networks?
---
**Bias** is a learnable parameter added to the weighted sum before the activation function.

$$y = f(w_1x_1 + w_2x_2 + ... + w_nx_n + b)$$
---
card_id: Rwv1oMAW
---
What is the purpose of the **bias** parameter in neural networks?
---
Allows the activation function to shift left or right, increasing model flexibility. Without bias, a neuron with ReLU would always output 0 when all inputs are 0.
---
card_id: g7gMRCAo
---
What is the difference between **bias** (the parameter) and **bias** (the error)?
---
**Bias (parameter)**: A learnable value added to weighted sums in neural networks (notation: $b$).

**Bias (error)**: The systematic error when a model's average predictions miss the true function - a measure of underfitting.

Same word, completely different meanings - context determines which one is meant.
---
card_id: oDvMlJB4
---
When should you use **normalization** (Min-Max scaling)?
---
You need bounded values (e.g., for neural networks with sigmoid/tanh).
---
card_id: ane3AE8n
---
What metric is best for detecting when a model is good for **ranking search results**, where top results matter more than lower-ranked ones?
---
**NDCG (Normalized Discounted Cumulative Gain)** - accounts for both relevance and position, with higher weight on top-ranked results.
---
card_id: MmsRd8lv
---
What metric is best for detecting when a model is good for **binary classification with imbalanced classes**, accounting for per-class accuracy?
---
**Balanced Accuracy** - averages recall across classes, giving equal weight to each class regardless of size.
---
card_id: 43rSzwMS
---
What is **harmonic mean**?
---
The **harmonic mean** is the reciprocal of the arithmetic mean of reciprocals, giving more weight to smaller values.
---
card_id: hz9ln6UK
---
When should you use **dropout**?
---
Use **dropout** when:
- Deep networks show signs of overfitting
- Training and validation error gap is large
- You have limited training data
- Training ensemble-like behavior is desired
---
card_id: s0W6m3B4
---
When should you use **batch normalization**?
---
Use **batch normalization** when:
- Training deep networks (10+ layers)
- You want faster convergence
- Experiencing internal covariate shift
- Need some regularization effect
---
card_id: QL9ECzHa
---
How is **batch normalization** applied differently at test time vs training time?
---
**Training**: Uses current mini-batch statistics (mean and variance)

**Test**: Uses running averages of mean and variance computed during training (fixed statistics)
---
card_id: qSzjK6Q7
---
How does **max pooling** work step-by-step?
---
1. Divide input into non-overlapping regions (e.g., 2×2 windows)
2. Take the maximum value from each region
3. Output the max values, reducing spatial dimensions
---
card_id: kiUNdQ3r
---
When should you use **max pooling** vs **average pooling**?
---
**Max pooling**: When you want to detect presence of features (dominant signal)

**Average pooling**: When you want to preserve overall information and reduce noise
---
card_id: hP0bJwWg
---
What is a **kernel (filter)** in convolutional layers?
---
A **kernel** is a small learnable weight matrix (e.g., 3×3) that slides across the input to detect specific patterns like edges, textures, or shapes.
---
card_id: 0PezB8bq
---
How does a **convolutional filter** process an image?
---
1. Place filter (e.g., 3×3) at top-left of image
2. Compute element-wise multiplication and sum (dot product)
3. Store result in output feature map
4. Slide filter right by stride (e.g., 1 pixel), repeat
5. Continue until entire image is covered
---
card_id: SmwOgnIf
---
Training **loss becomes NaN** during training. What are possible causes and solutions?
---
**Causes and solutions**:
- **Exploding gradients** → Use gradient clipping or lower learning rate
- **Learning rate too high** → Reduce learning rate by 10x
- **Poor weight initialization** → Use Xavier/He initialization
- **Numerical instability** → Add batch normalization or check for division by zero
---
card_id: v8csueVk
---
**Validation loss** stops decreasing at epoch 5 but **training loss** keeps dropping. What should you do?
---
**Overfitting detected** - use early stopping to stop training around epoch 5, or add regularization (dropout, L1/L2, data augmentation) and retrain.
---
card_id: 7I3sAruH
---
**Training loss** oscillates wildly and never converges. What's the likely cause?
---
**Learning rate is too high** - the optimizer overshoots the minimum with each step. Solution: Reduce learning rate by 10x (e.g., 0.1 → 0.01).
---
card_id: WGcGLu3V
---
**Training loss** decreases extremely slowly (0.001 per epoch). What's likely wrong?
---
**Learning rate is too low** - steps are too small to reach the minimum efficiently. Solution: Increase learning rate by 10x.
---
card_id: V8op2qdG
---
You have 1000 features but suspect only 50 are relevant. Should you use **L1** or **L2 regularization**?
---
**Use L1 (Lasso)** - it performs automatic feature selection by driving irrelevant feature weights to exactly zero, creating a sparse model with ~50 non-zero weights.
---
card_id: Tdztjjec
---
Why does **L1 regularization** drive weights to exactly zero while **L2** does not?
---
L1 uses absolute value $|w|$ which has constant gradient (±1), pushing weights by fixed amounts toward zero regardless of size. L2 uses $w^2$ with gradient proportional to $w$, so penalty weakens as weights approach zero.
---
card_id: 5SQg8fqQ
---
How do you implement **early stopping**?
---
1. Monitor validation loss each epoch
2. Track best validation loss seen so far
3. Set a **patience** parameter (e.g., 5 epochs)
4. Stop if validation loss doesn't improve for **patience** epochs
5. Restore weights from best epoch
---
card_id: QQ6Jwy6h
---
When should you use **normalization** vs **standardization**?
---
**Normalization (Min-Max)**: When you need bounded values in specific range (e.g., [0,1]) for neural networks with sigmoid/tanh

**Standardization (Z-score)**: When features have different units/scales and you want zero-centered data (more robust to outliers)
---
card_id: 42NO4IhZ
---
Why is **ReLU** preferred over **sigmoid/tanh** for hidden layers?
---
**ReLU advantages**:
- No vanishing gradient problem (gradient is 0 or 1)
- Faster computation (simple threshold)
- Creates sparse activations (many zeros)

**Sigmoid/tanh problems**: Gradients vanish for large |x|, slowing learning in deep networks.
---
card_id: vextRIdg
---
What are the tradeoffs between **batch**, **stochastic**, and **mini-batch gradient descent**?
---
**Batch GD**: Stable but slow, requires all data in memory

**Stochastic GD**: Fast updates but noisy, erratic convergence

**Mini-batch GD**: Best balance - faster than batch, more stable than stochastic, efficient GPU usage
---
card_id: FG4Kh38D
---
What factors determine **model complexity**?
---
- Number of parameters (weights, biases)
- Network depth (number of layers)
- Network width (neurons per layer)
- Polynomial degree (for regression)
- Tree depth (for decision trees)
