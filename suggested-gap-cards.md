# Suggested Cards to Fill Gaps in aiml-dMZe0dy2.md

These cards fill gaps in existing concepts (not introducing new topics).

---
card_id: null
---
What does **covariance** measure?
---
Measures how two variables change together - whether they increase/decrease in tandem (positive covariance) or opposite directions (negative covariance).
---
card_id: null
---
How does **covariance** differ from **correlation**?
---
**Covariance**: Scale-dependent, units = (units of X) × (units of Y), unbounded range

**Correlation**: Scale-independent (normalized covariance), dimensionless, range -1 to +1
---
card_id: null
---
What does **standard error** measure?
---
Measures the variability of the **sample mean** - how much sample means would vary if you repeatedly sampled from the same population.
---
card_id: null
---
How does **standard error** differ from **standard deviation**?
---
**Standard deviation**: Measures spread of individual data points

**Standard error**: Measures precision of the sample mean estimate
---
card_id: null
---
What does **R-squared** measure in regression?
---
Measures the proportion of variance in the target variable explained by the model.
---
card_id: null
---
How is **R-squared** interpreted?
---
- **R² = 0**: Model explains no variance (predicts the mean)
- **R² = 0.7**: Model explains 70% of variance
- **R² = 1**: Perfect predictions

Higher R² indicates better fit.
---
card_id: null
---
What is **harmonic mean**?
---
The **harmonic mean** is the reciprocal of the arithmetic mean of reciprocals, giving more weight to smaller values.
---
card_id: null
---
When should you use **dropout**?
---
Use **dropout** when:
- Deep networks show signs of overfitting
- Training and validation error gap is large
- You have limited training data
- Training ensemble-like behavior is desired
---
card_id: null
---
When should you use **batch normalization**?
---
Use **batch normalization** when:
- Training deep networks (10+ layers)
- You want faster convergence
- Experiencing internal covariate shift
- Need some regularization effect
---
card_id: null
---
How is **batch normalization** applied differently at test time vs training time?
---
**Training**: Uses current mini-batch statistics (mean and variance)

**Test**: Uses running averages of mean and variance computed during training (fixed statistics)
---
card_id: null
---
How does **max pooling** work step-by-step?
---
1. Divide input into non-overlapping regions (e.g., 2×2 windows)
2. Take the maximum value from each region
3. Output the max values, reducing spatial dimensions
---
card_id: null
---
When should you use **max pooling** vs **average pooling**?
---
**Max pooling**: When you want to detect presence of features (dominant signal)

**Average pooling**: When you want to preserve overall information and reduce noise
---
card_id: null
---
What is a **kernel (filter)** in convolutional layers?
---
A **kernel** is a small learnable weight matrix (e.g., 3×3) that slides across the input to detect specific patterns like edges, textures, or shapes.
---
card_id: null
---
What is **information gain**?
---
**Information gain** measures the reduction in entropy (uncertainty) achieved by splitting data on a particular feature.
---
card_id: null
---
Training **loss becomes NaN** during training. What are possible causes and solutions?
---
**Causes and solutions**:
- **Exploding gradients** → Use gradient clipping or lower learning rate
- **Learning rate too high** → Reduce learning rate by 10x
- **Poor weight initialization** → Use Xavier/He initialization
- **Numerical instability** → Add batch normalization or check for division by zero
---
card_id: null
---
**Validation loss** stops decreasing at epoch 5 but **training loss** keeps dropping. What should you do?
---
**Overfitting detected** - use early stopping to stop training around epoch 5, or add regularization (dropout, L1/L2, data augmentation) and retrain.
---
card_id: null
---
**Training loss** oscillates wildly and never converges. What's the likely cause?
---
**Learning rate is too high** - the optimizer overshoots the minimum with each step. Solution: Reduce learning rate by 10x (e.g., 0.1 → 0.01).
---
card_id: null
---
**Training loss** decreases extremely slowly (0.001 per epoch). What's likely wrong?
---
**Learning rate is too low** - steps are too small to reach the minimum efficiently. Solution: Increase learning rate by 10x.
---
card_id: null
---
You have 1000 features but suspect only 50 are relevant. Should you use **L1** or **L2 regularization**?
---
**Use L1 (Lasso)** - it performs automatic feature selection by driving irrelevant feature weights to exactly zero, creating a sparse model with ~50 non-zero weights.
---
card_id: null
---
Your binary classifier has 90% **precision** but 20% **recall**. What does this mean about the model's behavior?
---
The model is **extremely conservative** - it only predicts positive when very confident (high precision), but misses most positive cases (low recall). It's producing very few positive predictions.
---
card_id: null
---
**Accuracy** is 95% but **F1 score** is 0.30. What does this tell you about your dataset?
---
**Severe class imbalance** - the model is likely predicting the majority class for almost everything, achieving high accuracy but performing poorly on the minority class (low F1).
---
card_id: null
---
What is the tradeoff between **precision** and **recall**?
---
**Precision-Recall Tradeoff**: Lowering the classification threshold increases recall (catches more positives) but decreases precision (more false alarms). They move inversely.
---
card_id: null
---
Why does **L1 regularization** drive weights to exactly zero while **L2** does not?
---
L1 uses absolute value $|w|$ which has constant gradient (±1), pushing weights by fixed amounts toward zero regardless of size. L2 uses $w^2$ with gradient proportional to $w$, so penalty weakens as weights approach zero.
---
card_id: null
---
What is **data augmentation**?
---
**Data augmentation** creates modified versions of training examples through transformations (rotation, flipping, cropping, color shifts) to artificially expand dataset size.
---
card_id: null
---
What are common **data augmentation** techniques for images?
---
- Geometric: rotation, flipping, cropping, scaling
- Color: brightness/contrast adjustment, color jittering
- Noise: adding Gaussian noise, blur
- Advanced: cutout, mixup
---
card_id: null
---
When is **data augmentation** most effective?
---
Most effective when:
- Limited training data
- Model shows overfitting
- Training data doesn't cover all variations (e.g., rotations, lighting conditions)
- Domain allows realistic transformations
---
card_id: null
---
What is **early stopping**?
---
**Early stopping** is a regularization technique that stops training when validation performance stops improving, preventing overfitting.
---
card_id: null
---
How do you implement **early stopping**?
---
1. Monitor validation loss each epoch
2. Track best validation loss seen so far
3. Set a **patience** parameter (e.g., 5 epochs)
4. Stop if validation loss doesn't improve for **patience** epochs
5. Restore weights from best epoch
---
card_id: null
---
When should you use **normalization** vs **standardization**?
---
**Normalization (Min-Max)**: When you need bounded values in specific range (e.g., [0,1]) for neural networks with sigmoid/tanh

**Standardization (Z-score)**: When features have different units/scales and you want zero-centered data (more robust to outliers)
---
card_id: null
---
Why is **ReLU** preferred over **sigmoid/tanh** for hidden layers?
---
**ReLU advantages**:
- No vanishing gradient problem (gradient is 0 or 1)
- Faster computation (simple threshold)
- Creates sparse activations (many zeros)

**Sigmoid/tanh problems**: Gradients vanish for large |x|, slowing learning in deep networks.
---
card_id: null
---
What are the tradeoffs between **batch**, **stochastic**, and **mini-batch gradient descent**?
---
**Batch GD**: Stable but slow, requires all data in memory

**Stochastic GD**: Fast updates but noisy, erratic convergence

**Mini-batch GD**: Best balance - faster than batch, more stable than stochastic, efficient GPU usage
---
card_id: null
---
What is **model complexity**?
---
**Model complexity** refers to the model's capacity to fit various functions - determined by number of parameters, depth, and flexibility.
---
card_id: null
---
What factors determine **model complexity**?
---
- Number of parameters (weights, biases)
- Network depth (number of layers)
- Network width (neurons per layer)
- Polynomial degree (for regression)
- Tree depth (for decision trees)
---
card_id: null
---
How is an **ROC curve** constructed?
---
1. Train classifier that outputs probabilities
2. Vary classification threshold from 0 to 1
3. At each threshold, compute TPR (y-axis) and FPR (x-axis)
4. Plot points and connect them to form curve
---
card_id: null
---
What is the tradeoff between **Type I** and **Type II errors**?
---
**Type I vs Type II Tradeoff**: Reducing significance level $\alpha$ (stricter threshold) decreases Type I errors (false positives) but increases Type II errors (false negatives). More conservative tests catch fewer true effects.
---
card_id: null
---
How does **convolutional filter** process an image?
---
1. Place filter (e.g., 3×3) at top-left of image
2. Compute element-wise multiplication and sum (dot product)
3. Store result in output feature map
4. Slide filter right by stride (e.g., 1 pixel), repeat
5. Continue until entire image is covered
---
card_id: null
---
What are the symptoms of **high bias** vs **high variance**?
---
**High Bias (Underfitting)**:
- High training error
- High test error (similar to training)
- Model too simple

**High Variance (Overfitting)**:
- Low training error
- High test error (large gap)
- Model too complex
---
