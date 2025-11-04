---
card_id: 00TVK0st
---
What is the **coefficient of variation** formula?
---
$$CV = \frac{\sigma}{\mu}$$

- $CV$: coefficient of variation
- $\sigma$: standard deviation
- $\mu$: mean

Often expressed as a percentage by multiplying by 100.
---
card_id: 0WTSpGKk
---
In the bias-variance U-shaped error curve, why do **complex models** (right side) have high error?
---
**High variance dominates** - the model is too flexible and fits noise in the training data, leading to **overfitting** and poor generalization to test data.
---
card_id: 22vcNLe1
---
How is the **Pearson correlation coefficient** interpreted?
---
Measures the **strength and direction** of linear relationship between variables.

- **Range**: -1 to +1
- **+1**: Perfect positive linear relationship
- **-1**: Perfect negative linear relationship
- **0**: No linear relationship
---
card_id: Pp0xCBFf
---
What is a key limitation of the **Pearson correlation coefficient**?
---
Only captures **linear** relationships. Can be zero or near-zero even when strong non-linear relationships exist (e.g., quadratic, exponential).
---
card_id: 2CskO98D
---
What is **stratified sampling**?
---
**Stratified sampling** maintains the same class distribution in train/validation/test splits as in the original dataset.

**Example**: If the full dataset is 80% class A and 20% class B, each split will also be 80% class A and 20% class B.
---
card_id: 2IIebhrF
---
What are the characteristics of **high-bias** models?
---
Models that are too simple and make systematic errors, missing important patterns in the data (**underfitting**).
---
card_id: 2kiLoeDd
---
What are the characteristics of **L1 (Lasso) regularization**?
---
- Produces **sparse models** (drives many weights to exactly zero)
- Performs automatic **feature selection** (eliminates irrelevant features)
- Less smooth optimization (not differentiable at zero)
- Uses absolute value penalty: $\lambda \sum |w_i|$
---
card_id: 2mOH3FWf
---
What is **KL divergence** (Kullback-Leibler divergence)?
---
$$D_{KL}(P \| Q) = \sum_{i} p(x_i) \log \frac{p(x_i)}{q(x_i)}$$

**KL divergence** measures how different probability distribution $Q$ is from reference distribution $P$.

- **Always ≥ 0**
- **= 0** only when P and Q are identical
- **Not symmetric**: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$

Used to measure distribution mismatch in ML.
---
card_id: 4RyDdLRL
---
What does the **bias-variance tradeoff** describe?
---
The balance between two sources of error: **bias** (underfitting from overly simple models) and **variance** (overfitting from overly complex models).

Increasing complexity lowers bias but raises variance, and vice versa. The goal is finding the sweet spot that minimizes total error on unseen data.
---
card_id: 6JnCbrfd
---
What does **precision** measure in classification?
---
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Precision** measures the fraction of predicted positives that are actually positive. It answers: "Of all the cases we predicted as positive, how many were actually positive?"

- $TP$: true positives
- $FP$: false positives
---
card_id: 7rNOkySl
---
What is the formula for **sample mean**?
---
$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

- $\bar{x}$: sample mean
- $n$: sample size
- $x_i$: individual value
---
card_id: 7tn65klh
---
Why use **k-fold cross-validation**?
---
**Benefits**:
- More reliable performance estimate than single train/test split
- Every sample used for both training and validation
- Reduces variance from random splitting
- Better utilizes limited data

Especially valuable when you have a small dataset.
---
card_id: 7wMm28H2
---
What are the characteristics of **high-variance** models?
---
Models that are too sensitive to noise in training data, fitting even random fluctuations (**overfitting**).
---
card_id: 8FN3v2uL
---
What are the solutions to fix a **high bias** (underfitting) problem?
---
Use a more complex model, add more features, or reduce regularization.
---
card_id: 96nV20bO
---
How do **convolutional layers** contribute to translation invariance?
---
Same filters (kernels) are applied across the entire image through **parameter sharing**. A filter that detects an edge at one location can detect it anywhere, making detection position-independent.
---
card_id: LN4bCVK5
---
How do **pooling layers** contribute to translation invariance?
---
Downsampling creates tolerance to small spatial shifts. By summarizing regions (e.g., taking max or average), pooling makes the representation robust to minor position changes.
---
card_id: Y1vuPj96
---
How does **hierarchical architecture** contribute to translation invariance in CNNs?
---
Learns features progressively from local patterns (edges, textures) to global structure (shapes, objects). This multi-level abstraction builds position-independent representations at each layer.
---
card_id: AY8mz6zj
---
What is the formula for **standard error**?
---
$$SE = \frac{\sigma}{\sqrt{n}}$$

**Standard error** measures the variability of the **sample mean** across different samples from the population.

- $SE$: standard error
- $\sigma$: population standard deviation
- $n$: sample size

It decreases as sample size increases.
---
card_id: AneKadwj
---
In the bias-variance U-shaped error curve, what happens at **optimal complexity** (bottom of the U)?
---
**Minimum total error** is achieved - the model has the right balance between bias and variance, neither underfitting nor overfitting.

This is the sweet spot we aim for when tuning model complexity.
---
card_id: Ap2G6Y1Z
---
How does **batch normalization** enable faster training?
---
Allows use of **higher learning rates** without diverging. By normalizing layer inputs, it prevents gradients from becoming too large or too small, enabling more aggressive optimization.
---
card_id: w2knuDZX
---
How does **batch normalization** act as regularization?
---
Slight noise from using different batch statistics during training (each mini-batch has slightly different mean/variance) adds randomness that reduces overfitting, similar to dropout.
---
card_id: 1iXPivNz
---
How does **batch normalization** reduce sensitivity to initialization?
---
Makes the network less dependent on careful weight initialization. By normalizing layer inputs, even poor initial weights can be quickly adjusted during training.
---
card_id: BMdy0vzZ
---
How is **information gain** used in decision trees?
---
$$\text{Information Gain} = H(\text{parent}) - \sum_{children} \frac{n_{child}}{n_{parent}} H(\text{child})$$

**Information gain** measures the reduction in entropy after splitting on a feature.

Decision trees choose splits that **maximize information gain** - i.e., create the most "pure" child nodes with lowest entropy.
---
card_id: BWXw16o8
---
Why does sample variance use **n-1** instead of **n** (Bessel's correction)?
---
Using $n$ would systematically **underestimate** the true population variance (biased estimator).

**Reason**: The sample mean $\bar{x}$ is calculated from the same data, making deviations $(x_i - \bar{x})$ artificially smaller than deviations from true mean $\mu$.

Dividing by $n-1$ corrects this bias, producing an **unbiased estimator** of population variance.
---
card_id: BwTjPP18
---
What is the formula for **population mean**?
---
$$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$$

- $\mu$: population mean
- $N$: population size
- $x_i$: individual value
---
card_id: CJntZtqj
---
What does the **F1 score** measure in classification?
---
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

The **F1 score** is the harmonic mean of **precision** and **recall**. It answers: "What's the overall performance when both precision and recall matter equally?"

- $TP$: true positives
- $FP$: false positives
- $FN$: false negatives
---
card_id: DtODRtgY
---
When is the **median** preferred over the **mean**?
---
**Preferred when**:
- Data has **outliers** (median is robust, mean is sensitive)
- Distribution is **skewed** (median represents "typical" value better)

**Examples**: Income, house prices, response times - all typically have outliers or skewed distributions.
---
card_id: E5FTs0pm
---
When would you use the **ReLU** activation function?
---
**Default choice for hidden layers** in modern deep networks because it enables fast training, avoids vanishing gradients, and provides computational efficiency.
---
card_id: otpBB2Jm
---
What are the **advantages** of the ReLU activation function?
---
- **Fast computation**: Simple threshold operation $\max(0, x)$
- **Avoids vanishing gradients**: Gradient is either 0 or 1
- **Sparse activation**: Many neurons output 0, creating efficient representations
---
card_id: pqrWuW8t
---
What is the **"dying ReLU" problem**?
---
A drawback of ReLU where neurons can get stuck always outputting 0 (when input is always negative). Once "dead", these neurons stop learning because their gradient is always zero.
---
card_id: FbTVAB1c
---
What is the formula for **standard deviation**?
---
$$\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}$$

- $\sigma$: standard deviation
- $N$: population size
- $x_i$: individual value
- $\mu$: population mean
---
card_id: GXeoR8nY
---
How does **sample size** affect **standard error**?
---
**Inverse relationship**: $SE = \frac{\sigma}{\sqrt{n}}$

**As sample size increases**:
- Standard error **decreases** (by factor of $\sqrt{n}$)
- Sample mean becomes more precise estimate of population mean
- Confidence intervals narrow

**Doubling sample size** reduces SE by factor of $\sqrt{2} \approx 1.41$ (not by factor of 2).
---
card_id: IEXXk7CH
---
What does **standard error** measure?
---
**Standard error** measures the **uncertainty of the sample mean** as an estimate of the population mean.

It describes estimation uncertainty. SE = $\frac{SD}{\sqrt{n}}$, so it decreases as sample size increases.
---
card_id: IjY7uZyN
---
Why does **cross-entropy** use logarithms like **Shannon entropy**?
---
Both measure **information content** and **uncertainty**.

**Shannon entropy**: Measures expected information in a distribution
**Cross-entropy**: Measures information when using predicted distribution $q$ to encode true distribution $p$

The log arises from **information theory**: unlikely events ($p \to 0$) carry more information. Cross-entropy = entropy + KL divergence, so minimizing cross-entropy minimizes distribution mismatch.
---
card_id: Io0WDLWv
---
If **precision = 0.8** and **recall = 0.6**, what's the **F1 score**?
---
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \cdot \frac{0.8 \cdot 0.6}{0.8 + 0.6} = 2 \cdot \frac{0.48}{1.4} \approx 0.686$$

**F1 ≈ 0.69**

The F1 score is closer to the lower value (recall) because it's the harmonic mean, which penalizes imbalanced precision and recall.
---
card_id: IzHjvUzB
---
What is the **median**?
---
The **median** is the middle value when data is sorted (50th percentile).

If there's an even number of values, it's the average of the two middle values.
---
card_id: JEhEaCg0
---
You have a dataset with **1% positive class**. A model predicts all negative and achieves **99% accuracy**. What's the problem?
---
**Accuracy is misleading with imbalanced classes**. The model learned nothing - it's just exploiting class imbalance by always predicting the majority class.
---
card_id: 5zcoLyjx
---
What metrics should you use instead of accuracy for **imbalanced datasets**?
---
- **Precision/Recall** for the minority class
- **F1 score** (harmonic mean of precision and recall)
- **AUC-ROC** or **AUC-PR** curves
- **Confusion matrix** (reveals class-specific performance)

These reveal whether the model actually learns to detect the minority class.
---
card_id: IXlVHAM3
---
What are solutions for training models on **imbalanced datasets**?
---
- **Resampling**: Oversample minority class or undersample majority class
- **Class weights**: Penalize minority class errors more heavily
- **Cost-sensitive learning**: Assign different misclassification costs
- **Synthetic data**: Generate synthetic minority examples (e.g., SMOTE)
---
card_id: KG5Xunz3
---
How do **training error** and **validation error** look in a **high variance** (overfitting) model?
---
**Low training error** with a **large gap** to validation error (validation error much higher).
---
card_id: KWgsQacj
---
What is **dropout** in neural networks?
---
**Dropout** randomly deactivates neurons (sets outputs to zero) during training with probability $p$ (typically 0.5).

Each training iteration uses a different random subset of neurons, creating many implicit sub-networks.
---
card_id: KfXgBQCx
---
What is the advantage of using **max pooling** layers?
---
Reduces parameters, helps avoid overfitting, helps with translation invariance.
---
card_id: M5ZWdgan
---
What does **low precision, high recall** indicate about a classifier?
---
The model is **aggressive** - it predicts positive liberally, catching most true positives but also generating many false positives.

Example: A security system that catches all threats but also has many false alarms.
---
card_id: M8a4MCxf
---
What is the formula for **specificity** (true negative rate)?
---
$$\text{Specificity} = \frac{TN}{TN + FP}$$

**Specificity** measures the fraction of actual negatives that were correctly predicted. It answers: "Of all the actual negative cases, how many did we correctly identify?"

- $TN$: true negatives
- $FP$: false positives
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
card_id: NFeLxVO1
---
What is the formula for **RMSE** (root mean squared error)?
---
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} = \sqrt{\text{MSE}}$$

Square root of MSE, bringing the error metric back to the **same units as the target variable** (more interpretable than MSE).
---
card_id: u9dIC0m7
---
How does **RMSE** compare to **MAE** in terms of error sensitivity?
---
RMSE penalizes large errors more heavily than MAE due to the squaring operation. Also more sensitive to outliers than MAE.
---
card_id: NLMXEux6
---
What is **preprocessing leakage** in machine learning?
---
Using test data in preprocessing steps like scaling or normalization (e.g., computing mean/std from full dataset). This leaks information from test set into training, causing overly optimistic performance estimates.
---
card_id: iff4M53K
---
What is **temporal leakage** in machine learning?
---
Features that contain information from the future that wouldn't be available at prediction time. For example, using tomorrow's stock price to predict today's price, or using event outcomes in the features.
---
card_id: RroKKXiM
---
What is **target leakage** in machine learning?
---
When features are derived from or contain information about the target variable. For example, using "total treatment cost" to predict "disease diagnosis" when the cost was calculated after diagnosis.
---
card_id: NnZdkxpc
---
What does **high precision, low recall** indicate about a classifier?
---
The model is **conservative** - it only predicts positive when very confident, resulting in few false positives but missing many true positives (high false negatives).

Example: A spam filter that rarely marks legitimate emails as spam, but also misses catching many spam emails.
---
card_id: O5CAOzaV
---
A model has **5% training error** and **40% validation error**. What's the problem?
---
**High variance / overfitting** - large gap between training and validation error indicates the model memorizes training data but doesn't generalize well.
---
card_id: BjC2X90B
---
How can **more training data** help reduce overfitting?
---
More diverse examples make it harder for the model to memorize individual cases. Forces the model to learn general patterns rather than specific training examples.
---
card_id: I94brlLK
---
How can **reducing model complexity** help with overfitting?
---
Fewer parameters (layers, neurons, features) limit the model's capacity to memorize. Simpler models are forced to learn only the most important patterns, improving generalization.
---
card_id: XPmZv2Tq
---
How does **early stopping** prevent overfitting?
---
Stop training when validation error starts increasing, even if training error is still decreasing. This catches the model before it starts memorizing training data.
---
card_id: 8fkbFdoB
---
How does **data augmentation** reduce overfitting?
---
Creates modified versions of training examples (e.g., rotated, flipped, or cropped images). This artificially increases dataset size and diversity, making it harder to memorize.
---
card_id: OOPt9KXD
---
When would you prioritize **recall** over **precision**?
---
When the **cost of missing a positive case (false negative) is very high**.

Examples:
- Cancer screening (missing a cancer diagnosis is catastrophic)
- Fraud detection (missing fraud can be very costly)
- Fire alarm systems (better to have false alarms than miss a real fire)
---
card_id: PcfSpiot
---
Why is the **coefficient of variation** useful?
---
It expresses variability **relative to the mean**, allowing comparison across datasets with different units or scales.

**Example**: You can compare the variability of heights (cm) vs weights (kg) because CV is dimensionless.

Useful when comparing datasets with different means or units.
---
card_id: PjCkzoUQ
---
How is **dropout** applied differently at test time vs training time?
---
**Training time**: Neurons randomly dropped with probability $p$

**Test time**: All neurons active, but outputs scaled by $(1-p)$ to account for more neurons being active than during training.

This ensures expected outputs remain consistent between training and testing.
---
card_id: RiAIDyWx
---
What is the **ReLU** activation function?
---
$$\text{ReLU}(x) = \max(0, x)$$

**ReLU (Rectified Linear Unit)** outputs the input if positive, zero otherwise.

Simple piecewise linear function: outputs $x$ when $x > 0$, and $0$ when $x \leq 0$.
---
card_id: RlsAv1wC
---
What is **low variance** in a model?
---
The model is stable; predictions don't change much with different data.
---
card_id: SSEMZoNl
---
What is the formula for **harmonic mean**?
---
$$H = \frac{n}{\sum_{i=1}^{n} \frac{1}{x_i}}$$

- $H$: harmonic mean
- $n$: number of values
- $x_i$: individual value
---
card_id: SeVj1fRI
---
What is **low bias** in a model?
---
The model's average predictions are close to the true underlying function.
---
card_id: Ua99vlsL
---
When should you choose **L2** over **L1 regularization**?
---
Choose **L2 (Ridge)** when:
- You want to **shrink all features** without eliminating any
- Handling **multicollinearity** (correlated features)
- All features are potentially relevant
- You need stable, smooth optimization
---
card_id: UyutiyAf
---
How do you prevent **preprocessing leakage**?
---
Fit preprocessors (scalers, encoders, imputers) on **training data only**, then apply the fitted transformation to test data. Never compute statistics from the combined train+test set.
---
card_id: zx2SiYw4
---
How do you prevent **temporal leakage** in time-series problems?
---
Ensure strict temporal ordering - training data must be from earlier time periods than test data. Carefully audit features to ensure they don't contain information from the future.
---
card_id: wEgsRGQn
---
Why should you remove duplicates **before** splitting train/test sets?
---
Duplicates across train/test splits cause data leakage - the model sees nearly identical examples in both training and testing, leading to overly optimistic performance estimates that won't generalize.
---
card_id: VJfaVUwg
---
What does a **z-score** represent?
---
**Z-score** measures how many standard deviations a value is from the mean.

**Examples**:
- $z = 0$: Value equals the mean
- $z = 2$: Value is 2 standard deviations above the mean
- $z = -1.5$: Value is 1.5 standard deviations below the mean
---
card_id: WQLc1i5o
---
What is the formula for **R-squared** (coefficient of determination)?
---
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

**R²** measures the **proportion of variance in the target explained by the model**.

- **Range**: 0 to 1 (can be negative for bad models)
- **R² = 1**: Perfect predictions
- **R² = 0**: Model no better than predicting the mean
---
card_id: X03z1kIe
---
What is the **total error decomposition** formula?
---
$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

- **Bias²**: Error from wrong model assumptions
- **Variance**: Error from sensitivity to training data
- **Irreducible Error**: Noise in the data that cannot be eliminated

The bias-variance tradeoff manages the first two terms.
---
card_id: XBOtRLIg
---
What are the characteristics of **L2 (Ridge) regularization**?
---
- Produces **dense models** (shrinks all weights toward zero, but rarely to exactly zero)
- Performs **feature shrinkage** (reduces impact of all features proportionally)
- Smoother optimization (differentiable everywhere)
- Uses squared penalty: $\lambda \sum w_i^2$
---
card_id: YLooMKtq
---
How does **increasing regularization** affect the bias-variance tradeoff?
---
**Increasing regularization**:
- Constrains the model → **increases bias** (simpler, more constrained model)
- Reduces sensitivity to training data → **decreases variance**
- Moves away from overfitting, toward underfitting

Stronger regularization penalties force simpler models.
---
card_id: YXfbAA9Q
---
What is the formula for **population variance**?
---
$$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$$

- $\sigma^2$: population variance
- $N$: population size
- $x_i$: individual value
- $\mu$: population mean
---
card_id: YcOEXDSR
---
What is the formula for **sample variance**?
---
$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

- $s^2$: sample variance
- $n$: sample size
- $x_i$: individual value
- $\bar{x}$: sample mean
---
card_id: Ywmf8Vez
---
Given **precision=0.80** and **recall=0.89**, calculate **F1 score**.
---
$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot 0.80 \cdot 0.89}{0.80 + 0.89} = \frac{1.424}{1.69} \approx 0.84$$

**F1 ≈ 0.84** (or 84%)
---
card_id: ZKmDv1w2
---
A linear model struggles to fit quadratic data, achieving only 70% accuracy on both train and test sets. What's the issue?
---
**Problem**: **High bias / underfitting** - the model is too simple (linear) to capture the true pattern (quadratic).

**Solution**: Use a more complex model (polynomial, neural network, etc.) that can represent curved decision boundaries.
---
card_id: ZxbODedV
---
In the dartboard analogy for machine learning, what does **variance** represent?
---
**Variance** represents the spread of arrows - how scattered the shots are around their average position.

Just like a model with high variance has inconsistent predictions across different training sets, arrows with high variance are spread out widely.
---
card_id: a2dKNgsc
---
What is **batch normalization**?
---
**Batch normalization** normalizes layer inputs to have mean 0 and variance 1 within each mini-batch.

$$\hat{x} = \frac{x - \mu_{batch}}{\sqrt{\sigma^2_{batch} + \epsilon}}$$

where $\mu_{batch}$ and $\sigma^2_{batch}$ are the mean and variance of the current mini-batch.
---
card_id: aEViSQBy
---
What is **L1 regularization** (Lasso) and what's its formula?
---
$$\text{Loss} = \text{Original Loss} + \lambda \sum_{i=1}^{n} |w_i|$$

**L1 regularization** adds the **sum of absolute values** of weights to the loss. $\lambda$ controls regularization strength.
---
card_id: jcGg9U6M
---
What is the key effect of **L1 regularization** on model weights?
---
Drives some weights to **exactly zero**, performing automatic **feature selection** and creating **sparse models**. Useful when you have many irrelevant features.
---
card_id: aK8uoMDN
---
When is the **mode** useful?
---
**Useful for**:
- **Categorical data** (where mean/median don't make sense)
- **Multimodal distributions** (reveals multiple peaks in the data)
- Finding the most common category/value

**Examples**: Most popular product, most common error type, most frequent customer segment.
---
card_id: aKE7NdOT
---
What is the **Shannon entropy** formula?
---
$$H(X) = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i)$$

- $H(X)$: entropy (information content)
- $p(x_i)$: probability of outcome $x_i$
- $n$: number of possible outcomes
---
card_id: axqKSiqY
---
When would you use the **tanh** activation function?
---
**Hidden layers** when you need zero-centered outputs (range -1 to 1). Better than sigmoid for hidden layers because zero-centering helps with gradient flow.
---
card_id: NiVGVF0v
---
What is the **tanh** activation function formula?
---
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

Outputs values in range **-1 to 1** (zero-centered, unlike sigmoid).
---
card_id: IBx0F6J0
---
What is the **drawback** of the tanh activation function?
---
Still suffers from the **vanishing gradient problem** in very deep networks, similar to sigmoid (gradients become very small for large positive or negative inputs).
---
card_id: b2hMhrnz
---
What is **bias** in machine learning?
---
The measure of how far the model's average predictions are from the true underlying function.
---
card_id: bOTPjqHx
---
What is the **cross-entropy** formula for binary classification?
---
$$H(p, q) = -\sum_{i} p(x_i) \log q(x_i)$$

For binary classification:
$$\text{CE} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

**Cross-entropy** measures how well predicted probabilities $q$ match true distribution $p$. Commonly used as a loss function for classification.

- $y$: true label (0 or 1)
- $\hat{y}$: predicted probability
---
card_id: bWmZd4uw
---
When does **Shannon entropy** reach its maximum value?
---
Entropy is **maximized** when the distribution is **uniform** (all outcomes equally likely).

For $n$ equally likely outcomes:
$$H_{max} = \log_2(n)$$

**Intuition**: Maximum uncertainty occurs when you have no information to favor any outcome over another.
---
card_id: bg0wcNu3
---
What is the formula for **accuracy**?
---
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- $TP$: true positives
- $TN$: true negatives
- $FP$: false positives
- $FN$: false negatives
---
card_id: cRoYSdrx
---
What is **k-fold cross-validation**?
---
Split data into $k$ equal folds. Train on $k-1$ folds, validate on the remaining fold. Repeat $k$ times, each fold used as validation once.

Average the $k$ validation scores for final performance estimate.

**Common choices**: $k = 5$ or $k = 10$
---
card_id: dEESfLjT
---
What happens when you **decrease** the regularization parameter **λ**?
---
**Decreasing λ** (weaker regularization):
- **Decreases bias** (model becomes more flexible)
- **Increases variance** (more sensitive to training data)
- Moves toward overfitting

**λ = 0**: No regularization at all (maximum variance, minimum bias)
---
card_id: eqJ4rBsS
---
How do **training error** and **validation error** look in a well-tuned model?
---
Both are **low** with a **small gap** between them.
---
card_id: f0s3zz8E
---
Why use a **train/validation/test** split (three sets)?
---
- **Training set**: Train the model
- **Validation set**: Tune hyperparameters and select models (used repeatedly during development)
- **Test set**: Final evaluation only (used once, never for training decisions)

**Without separate test set**: Validation performance becomes optimistic due to indirect overfitting through hyperparameter tuning.
---
card_id: fv6KTOih
---
What are the **bias-variance** characteristics of **low-complexity** models?
---
**High bias** and **low variance**, tending to **underfit** but generalizing better with noisy or limited data.
---
card_id: gpTDonuX
---
How does **MSE** relate to **bias and variance**?
---
$$\text{MSE} = \text{Bias}^2 + \text{Variance}$$

For a model's predictions:
- **Bias²**: How far the average prediction is from the true value
- **Variance**: How much predictions fluctuate across different training sets

MSE decomposes into these two fundamental sources of error.
---
card_id: hMnubTjV
---
What is **data leakage**?
---
**Data leakage** occurs when information from outside the training set is used during training, leading to overly optimistic performance that doesn't generalize.

The model learns from information it shouldn't have access to, causing inflated performance metrics.
---
card_id: hYt9lPto
---
What is **AUC** and how is it interpreted?
---
**Area Under the ROC Curve** - a single metric summarizing classifier performance across all thresholds.

- **AUC = 1.0**: Perfect classifier
- **AUC = 0.5**: Random guessing (diagonal line)
- **AUC < 0.5**: Worse than random (predictions are inverted)

Higher AUC means better separation between classes.
---
card_id: hjBWXkj3
---
When would you prioritize **precision** over **recall**?
---
When the **cost of a false positive is high** or you have limited resources to handle positive predictions.

Examples:
- Spam filtering (annoying to lose legitimate emails)
- Recommendation systems (bad recommendations hurt user experience)
- Criminal justice (wrongly convicting innocent people)
---
card_id: jFOdKDhb
---
What happens when you **increase** the regularization parameter **λ**?
---
**Increasing λ** (stronger regularization):
- **Increases bias** (model becomes simpler, more constrained)
- **Decreases variance** (less sensitive to training data)
- Moves toward underfitting

Stronger penalty on large weights forces simpler models.
---
card_id: mdlIumeg
---
Why can **accuracy** be misleading?
---
If dataset has class imbalance the model can have high global accuracy while completely failing at one or more classes.
---
card_id: o2mCdan7
---
What does **recall** measure in classification?
---
$$\text{Recall} = \frac{TP}{TP + FN}$$

**Recall** measures the fraction of actual positives that were correctly predicted. It answers: "Of all the actual positive cases, how many did we find?"

- $TP$: true positives
- $FN$: false negatives
---
card_id: oA6OP9w6
---
What is **variance** in machine learning?
---
The measure of how much the model's predictions fluctuate when trained on different samples of the data.
---
card_id: p1Hbz16b
---
How do **training error** and **validation error** look in a **high bias** (underfitting) model?
---
Both **training error** and **validation error** are **high**.
---
card_id: pMZK77F4
---
What is **translation invariance**?
---
**Translation invariance** is the ability to recognize a feature regardless of its position in the input.

**Example**: A CNN can detect a cat whether it's in the top-left or bottom-right of an image - the spatial location doesn't matter.
---
card_id: pfiGwlny
---
What do **convolutional layers** do in a neural network?
---
Apply learned filters (kernels) that slide across the input, detecting local patterns like edges, textures, and shapes. Fundamental building block of CNNs for image and spatial data.
---
card_id: ZF2WFN9M
---
What is **parameter sharing** in convolutional layers?
---
The same filter is applied across all positions in the input. This dramatically reduces parameters compared to fully connected layers, while enabling position-independent feature detection.
---
card_id: NptxfSGz
---
What is **local connectivity** in convolutional layers?
---
Each neuron connects only to a small region of the input (receptive field) rather than the entire input. This captures local patterns efficiently and reduces computational cost.
---
card_id: qHMwoyDJ
---
What is the **mean absolute error (MAE)** formula?
---
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

- $\text{MAE}$: mean absolute error
- $n$: number of samples
- $y_i$: true value for sample $i$
- $\hat{y}_i$: predicted value for sample $i$
---
card_id: qJtxNzq2
---
In the bias-variance U-shaped error curve, why do **simple models** (left side) have high error?
---
**High bias dominates** - the model is too simple to capture the underlying patterns, leading to **underfitting** and high error on both training and test data.
---
card_id: qo9zIkdR
---
In the dartboard analogy for machine learning, what does **bias** represent?
---
**Bias** represents systematic aiming error - arrows consistently miss the center in the same direction.

Just like a model with high bias systematically misses the true function, arrows with bias consistently miss the bullseye in one direction.
---
card_id: rA8ttBZx
---
How does **dropout** prevent **co-adaptation** of neurons?
---
By randomly dropping neurons, dropout prevents neurons from relying on specific other neurons always being present. Forces each neuron to learn robust features independently.
---
card_id: RiLlpkrC
---
How does **dropout** create an ensemble learning effect?
---
Each training iteration uses a different random subset of neurons, effectively training many different subnetworks. At test time, using all neurons approximates averaging these subnetworks' predictions.
---
card_id: QodC4mNG
---
Why does **dropout** force redundancy in neural networks?
---
Since any neuron might be dropped, multiple neurons must learn to detect the same important features. This redundancy makes the network more robust and less likely to overfit to specific neuron combinations.
---
card_id: rEopJVIo
---
What is the **mode**?
---
The **mode** is the most frequently occurring value in a dataset.

A dataset can have one mode (unimodal), two modes (bimodal), or multiple modes (multimodal).
---
card_id: rSiF58qj
---
What is the **Pearson correlation coefficient** formula?
---
$$r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

- $r$: Pearson correlation coefficient
- $\text{Cov}(X, Y)$: covariance between X and Y
- $\sigma_X$, $\sigma_Y$: standard deviations of X and Y

Unlike covariance, it's **normalized** to range from -1 to +1.
---
card_id: sXQ5x8Ri
---
What is the **mean squared error (MSE)** formula?
---
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- $\text{MSE}$: mean squared error
- $n$: number of samples
- $y_i$: true value for sample $i$
- $\hat{y}_i$: predicted value for sample $i$
---
card_id: t5IIfe3T
---
When would you use the **sigmoid** activation function?
---
**Output layer** for **binary classification** - it outputs probabilities between 0 and 1 that can be directly interpreted as class probabilities.
---
card_id: dBiU2RDW
---
What is the **sigmoid** activation function formula?
---
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Outputs values between **0 and 1**, making it suitable for probability interpretation.
---
card_id: xZvfKOc7
---
What is the **drawback** of using sigmoid in hidden layers?
---
Suffers from **vanishing gradients** - gradients become very small for large positive or negative inputs, slowing or stopping learning in deep networks. Not recommended for hidden layers.
---
card_id: tEJfmjf2
---
How does **decreasing regularization** affect the bias-variance tradeoff?
---
**Decreasing regularization**:
- More model freedom → **decreases bias** (more flexible model)
- More sensitivity to training data → **increases variance**
- Moves away from underfitting, toward overfitting

Less constraint allows the model to fit training data more closely.
---
card_id: tNzp58NI
---
What does **standard deviation** measure?
---
**Standard deviation** measures the **spread of individual data points** around the mean.

It describes the variability in the data itself and does not change with sample size (for a given population).
---
card_id: uXtcGBI0
---
Given a model with **99% training accuracy** but **60% test accuracy**, what's the problem?
---
**High variance / overfitting** - the model memorizes training data but doesn't generalize to new data.

The large gap between training and test accuracy is the key indicator of overfitting.
---
card_id: vWrhqLpm
---
When is **MSE/RMSE** preferred as a loss metric?
---
**MSE/RMSE** is preferred when:
- **Large errors are particularly costly** (quadratic penalty punishes them more)
- Mathematical convenience needed (MSE is differentiable everywhere)
- You want to penalize variance more heavily

MSE penalizes large errors more: $(y - \hat{y})^2$
---
card_id: uOGEawx7
---
What is the **ROC curve**?
---
**Receiver Operating Characteristic** curve plots **True Positive Rate (Recall)** vs **False Positive Rate** at various classification thresholds.

- X-axis: False Positive Rate = $\frac{FP}{FP + TN}$
- Y-axis: True Positive Rate = $\frac{TP}{TP + FN}$ (Recall)

Used to evaluate classifier performance across all thresholds, not just one.
---
card_id: BKxniY3t
---
What is the formula for **covariance**?
---
$$\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$$

**Covariance** measures how two variables change together:
- **Positive**: Variables increase together
- **Negative**: One increases as other decreases
- **Zero**: No linear relationship

Units depend on variable units, making interpretation difficult.
---
card_id: mSSyzDUL
---
Given **80 true positives (TP)** and **20 false positives (FP)**, calculate **precision**.
---
$$\text{Precision} = \frac{TP}{TP + FP} = \frac{80}{80 + 20} = \frac{80}{100} = 0.80$$

**Precision = 0.80** (or 80%)

80% of positive predictions were correct.
---
card_id: jYsXth6g
---
What is the formula for **z-score**?
---
$$z = \frac{x - \mu}{\sigma}$$

- $z$: z-score (standardized value)
- $x$: raw value
- $\mu$: mean
- $\sigma$: standard deviation
---
card_id: ANSlMJb3
---
What is **L2 regularization** (Ridge) and what's its formula?
---
$$\text{Loss} = \text{Original Loss} + \lambda \sum_{i=1}^{n} w_i^2$$

**L2 regularization** adds the **sum of squared weights** to the loss. $\lambda$ controls regularization strength.
---
card_id: cuvmcyiq
---
What is the key effect of **L2 regularization** on model weights?
---
Shrinks all weights toward zero (but rarely to exactly zero), preventing any single feature from dominating. Creates **smoother, more stable models** with **dense** representations (all features retained).
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
card_id: aLII3Y0k
---
When should you use **stratified sampling**?
---
**Use when**:
- **Imbalanced classes** (e.g., 95% negative, 5% positive)
- Small datasets (ensures each split has examples from all classes)
- Rare categories that might be missed in random splits

Prevents situations where a class might be missing from validation/test sets.
