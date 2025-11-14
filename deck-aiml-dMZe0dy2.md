---
card_id: 00TVK0st
---
What is the **coefficient of variation** formula?
---
$$CV = \frac{\sigma}{\mu}$$

- $CV$: coefficient of variation
- $\sigma$: standard deviation
- $\mu$: mean
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
---
card_id: yy3vuZcw
---
What is an example of **stratified sampling**?
---
If the full dataset is 80% class A and 20% class B, each split will also be 80% class A and 20% class B.
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
---
card_id: 2mOH3FWf
---
What is the **KL divergence** formula?
---
$$D_{KL}(P \| Q) = \sum_{i} p(x_i) \log \frac{p(x_i)}{q(x_i)}$$
---
card_id: wf5wpU98
---
What does **KL divergence** measure?
---
Measures how different probability distribution $Q$ is from reference distribution $P$.
---
card_id: E2XqXHrR
---
What are the key properties of **KL divergence**?
---
- **Always ≥ 0**
- **= 0** only when P and Q are identical
- **Not symmetric**: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$
---
card_id: MJFAqEXp
---
How is **KL divergence** used in machine learning?
---
- **Loss function component** (e.g., in VAEs, measuring latent distribution vs prior)
- **Comparing model outputs** to target distributions (related to cross-entropy)
- **Model evaluation** and distribution alignment
---
card_id: 4RyDdLRL
---
What does the **bias-variance tradeoff** describe?
---
The balance between two sources of error: **bias** (underfitting from overly simple models) and **variance** (overfitting from overly complex models).
---
card_id: 6JnCbrfd
---
What is the **precision (PPV, Positive Predictive Value)** formula?
---
$$\text{Precision} = \frac{TP}{TP + FP}$$

- $TP$: true positives
- $FP$: false positives
---
card_id: rTz7miPH
---
What does **precision (PPV, Positive Predictive Value)** measure in classification?
---
**Precision** measures the fraction of predicted positives that are actually positive.
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
Provides more reliable performance estimates than single train/test split by reducing variance from random splitting and utilizing all data for both training and validation.
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
Same filters (kernels) are applied across the entire image through **parameter sharing**.
---
card_id: LN4bCVK5
---
How do **pooling layers** contribute to translation invariance?
---
By summarizing regions, pooling creates tolerance to small spatial shifts.
---
card_id: Y1vuPj96
---
How does **hierarchical architecture** contribute to translation invariance in CNNs?
---
Learns features progressively from local patterns (edges, textures) to global structure (shapes, objects), building position-independent representations at each layer.
---
card_id: AY8mz6zj
---
What is the formula for **standard error**?
---
$$SE = \frac{\sigma}{\sqrt{n}}$$

- $SE$: standard error
- $\sigma$: population standard deviation
- $n$: sample size
---
card_id: AneKadwj
---
In the bias-variance U-shaped error curve, what happens at **optimal complexity** (bottom of the U)?
---
**Minimum total error** is achieved - the model has the right balance between bias and variance, neither underfitting nor overfitting.
---
card_id: Ap2G6Y1Z
---
How does **batch normalization** enable faster training?
---
Allows use of **higher learning rates** without diverging by preventing gradients from becoming too large or too small.
---
card_id: w2knuDZX
---
How does **batch normalization** act as regularization?
---
Slight noise from using different batch statistics during training (each mini-batch has slightly different mean/variance) adds randomness that reduces overfitting.
---
card_id: 1iXPivNz
---
How does **batch normalization** reduce sensitivity to initialization?
---
By normalizing layer inputs, even poor initial weights can be quickly adjusted during training.
---
card_id: BMdy0vzZ
---
What is the **information gain** formula?
---
$$\text{Information Gain} = H(\text{parent}) - \sum_{children} \frac{n_{child}}{n_{parent}} H(\text{child})$$
---
card_id: uTKVGbxs
---
How is **information gain** used in decision trees?
---
Decision trees choose splits that **maximize information gain** - i.e., create the most "pure" child nodes with lowest entropy.
---
card_id: BWXw16o8
---
Why does sample variance use **n-1** instead of **n** (Bessel's correction)?
---
Using $n$ would systematically underestimate the population variance because the sample mean $\bar{x}$ is calculated from the same data, making deviations artificially smaller. Dividing by $n-1$ corrects this bias, producing an unbiased estimator.
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
What is the **F1 score** formula?
---
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

- $TP$: true positives
- $FP$: false positives
- $FN$: false negatives
---
card_id: zicKRF6O
---
What does the **F1 score** measure in classification?
---
The **F1 score** is the harmonic mean of **precision** and **recall**.
---
card_id: DtODRtgY
---
When is the **median** preferred over the **mean**?
---
**Preferred when**:
- Data has **outliers** (median is robust, mean is sensitive)
- Distribution is **skewed** (median represents "typical" value better)
---
card_id: KOqfLBVF
---
What are examples of data where **median** is preferred over **mean**?
---
Income, house prices, response times.
---
card_id: E5FTs0pm
---
When would you use the **ReLU** activation function?
---
**Default choice for hidden layers** in modern deep networks.
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
**Inverse relationship**: As sample size increases, standard error **decreases** (by factor of $\sqrt{n}$).
---
card_id: 2UUY5jSm
---
What are the implications of increasing sample size?
---
Estimates become more precise - the sample mean approaches the population mean and confidence intervals narrow.
---
card_id: IjY7uZyN
---
Why does **cross-entropy** use logarithms like **Shannon entropy**?
---
The log arises from **information theory**: unlikely events ($p \to 0$) carry more information.
---
card_id: IzHjvUzB
---
What is the **median**?
---
The **median** is the middle value when data is sorted (50th percentile).
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
Use class-specific metrics (precision, recall, F1 score) or threshold-independent metrics (AUC-ROC, AUC-PR) that account for per-class performance rather than overall accuracy.
---
card_id: IXlVHAM3
---
What are solutions for training models on **imbalanced datasets**?
---
Use resampling techniques (oversampling minority/undersampling majority), assign higher class weights to minority class errors, or generate synthetic examples (e.g., SMOTE).
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
---
card_id: KfXgBQCx
---
What are the advantages of using **max pooling** layers?
---
- Reduces parameters
- Helps avoid overfitting
- Provides translation invariance
---
card_id: M5ZWdgan
---
What does **low precision, high recall** indicate about a classifier?
---
The model is **aggressive** - it predicts positive liberally.
---
card_id: tvbpWodL
---
What is an example of a **low precision, high recall** classifier?
---
A security system that catches all threats but also has many false alarms.
---
card_id: M8a4MCxf
---
What is the formula for **specificity** (true negative rate)?
---
$$\text{Specificity} = \frac{TN}{TN + FP}$$

- $TN$: true negatives
- $FP$: false positives
---
card_id: LdzlSXj1
---
What does **specificity** measure?
---
**Specificity** measures the fraction of actual negatives that were correctly predicted.
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
The model is **conservative** - it only predicts positive when very confident.
---
card_id: Y5gL62EZ
---
What is an example of a **high precision, low recall** classifier?
---
A spam filter that rarely marks legitimate emails as spam, but also misses catching many spam emails.
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
More diverse examples make it harder for the model to memorize individual cases, forcing it to learn general patterns.
---
card_id: I94brlLK
---
How can **reducing model complexity** help with overfitting?
---
Fewer parameters (layers, neurons, features) limit the model's capacity to memorize, forcing it to learn only the most important patterns.
---
card_id: XPmZv2Tq
---
How does **early stopping** prevent overfitting?
---
Stops training when validation error starts increasing, even if training error is still decreasing, catching the model before it memorizes training data.
---
card_id: 8fkbFdoB
---
How does **data augmentation** reduce overfitting?
---
Creates modified versions of training examples (e.g., rotated, flipped, or cropped images). This artificially increases dataset size and diversity, making it harder to memorize.
---
card_id: OOPt9KXD
---
When would you prioritize **recall (Sensitivity)** over **precision (PPV)**?
---
When the **cost of missing a positive case (false negative) is very high**.
---
card_id: YzMsMOqE
---
What are examples where **recall** should be prioritized over **precision**?
---
- Cancer screening (missing a cancer diagnosis is catastrophic)
- Fraud detection (missing fraud can be very costly)
- Fire alarm systems (better to have false alarms than miss a real fire)
---
card_id: PcfSpiot
---
Why is the **coefficient of variation** useful?
---
It expresses variability **relative to the mean**, allowing comparison across datasets with different units or scales (CV is dimensionless).
---
card_id: PjCkzoUQ
---
How is **dropout** applied differently at test time vs training time?
---
**Training time**: Neurons randomly dropped with probability $p$

**Test time**: All neurons active, but outputs scaled by $(1-p)$ to account for more neurons being active than during training.
---
card_id: RiAIDyWx
---
What is the **ReLU** activation function formula?
---
$$\text{ReLU}(x) = \max(0, x)$$
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
---
card_id: Lttgwt7j
---
What do different **z-score** values indicate?
---
- $z = 0$: Value equals the mean
- $z = 2$: Value is 2 standard deviations above the mean
- $z = -1.5$: Value is 1.5 standard deviations below the mean
---
card_id: WQLc1i5o
---
What is the formula for **R-squared** (coefficient of determination)?
---
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$
---
card_id: X03z1kIe
---
What is the **total error decomposition** formula?
---
$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

- **Bias²**: Error from wrong model assumptions
- **Variance**: Error from sensitivity to training data
- **Irreducible Error**: Noise in the data that cannot be eliminated
---
card_id: XBOtRLIg
---
What are the characteristics of **L2 (Ridge) regularization**?
---
- Produces **dense models** (shrinks all weights toward zero, but rarely to exactly zero)
- Performs **feature shrinkage** (reduces impact of all features proportionally)
- Smoother optimization (differentiable everywhere)
---
card_id: YLooMKtq
---
How does **increasing regularization** affect the bias-variance tradeoff?
---
**Increasing regularization**:
- Constrains the model → **increases bias** (simpler, more constrained model)
- Reduces sensitivity to training data → **decreases variance**
- Moves away from overfitting, toward underfitting
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
**High bias / underfitting** - the model is too simple (linear) to capture the true pattern (quadratic).
---
card_id: ZxbODedV
---
In the dartboard analogy for machine learning, what does **variance** represent?
---
**Variance** represents the spread of arrows - how scattered the shots are around their average position.
---
card_id: a2dKNgsc
---
What is **batch normalization**?
---
**Batch normalization** normalizes layer inputs to have mean 0 and variance 1 within each mini-batch.
---
card_id: 8AeZNfcx
---
What is the **batch normalization** formula?
---
$$\hat{x} = \frac{x - \mu_{batch}}{\sqrt{\sigma^2_{batch} + \epsilon}}$$

where $\mu_{batch}$ and $\sigma^2_{batch}$ are the mean and variance of the current mini-batch.
---
card_id: aEViSQBy
---
What is the **L1 regularization** (Lasso) formula?
---
$$\text{Loss} = \text{Original Loss} + \lambda \sum_{i=1}^{n} |w_i|$$
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
**Hidden layers** when you need zero-centered outputs (range -1 to 1).
---
card_id: NiVGVF0v
---
What is the **tanh** activation function formula?
---
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

Outputs values in range **-1 to 1** (zero-centered).
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
card_id: VpK7Xt6p
---
What is the **bias formula** in machine learning?
---
$$\text{Bias}[\hat{f}(x)] = E[\hat{f}(x)] - f(x)$$

or equivalently:

$$\text{Bias} = E[\hat{y}] - y$$

- $E[\hat{f}(x)]$: expected prediction of the model
- $f(x)$: true function value
- $E[\hat{y}]$: expected predicted value
- $y$: true value
---
card_id: bOTPjqHx
---
What is the **cross-entropy** formula?
---
$$H(p, q) = -\sum_{i} p(x_i) \log q(x_i)$$
---
card_id: mnAhXudy
---
What is the **cross-entropy** formula for binary classification?
---
$$\text{CE} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

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

- **Bias²**: How far the average prediction is from the true value
- **Variance**: How much predictions fluctuate across different training sets
---
card_id: hMnubTjV
---
What is **data leakage**?
---
**Data leakage** occurs when information from outside the training set is used during training, leading to overly optimistic performance that doesn't generalize.
---
card_id: hYt9lPto
---
What is **AUC** (Area Under the ROC Curve)?
---
**AUC** is the area under the ROC curve - a single metric summarizing classifier performance across all thresholds.
---
card_id: 3WiiBpVX
---
How is **AUC** interpreted?
---
- **AUC = 1.0**: Perfect classifier
- **AUC = 0.5**: Random guessing (diagonal line)
- **AUC < 0.5**: Worse than random (predictions are inverted)

Higher AUC means better separation between classes.
---
card_id: hjBWXkj3
---
When would you prioritize **precision (PPV)** over **recall (Sensitivity)**?
---
When the **cost of a false positive is high** or you have limited resources to handle positive predictions.
---
card_id: NGJ3vvkZ
---
What are examples where **precision** should be prioritized over **recall**?
---
- Spam filtering (annoying to lose legitimate emails)
- Recommendation systems (bad recommendations hurt user experience)
- Criminal justice (wrongly convicting innocent people)
---
card_id: mdlIumeg
---
Why can **accuracy** be misleading?
---
If dataset has class imbalance the model can have high global accuracy while completely failing at one or more classes.
---
card_id: o2mCdan7
---
What is the **recall (Sensitivity, True Positive Rate)** formula?
---
$$\text{Recall} = \frac{TP}{TP + FN}$$

- $TP$: true positives
- $FN$: false negatives
---
card_id: 8RwDlaTy
---
What does **recall (Sensitivity, True Positive Rate)** measure in classification?
---
**Recall** measures the fraction of actual positives that were correctly predicted.
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
---
card_id: uKMseWHY
---
What is an example of **translation invariance** in CNNs?
---
A CNN can detect a cat whether it's in the top-left or bottom-right of an image - the spatial location doesn't matter.
---
card_id: pfiGwlny
---
What do **convolutional layers** do in a neural network?
---
Apply learned filters (kernels) that slide across the input, detecting local patterns like edges, textures, and shapes.
---
card_id: ZF2WFN9M
---
What is **parameter sharing** in convolutional layers?
---
The same filter is applied across all positions in the input.
---
card_id: NptxfSGz
---
What is **local connectivity** in convolutional layers?
---
Each neuron connects only to a small region of the input (receptive field) rather than the entire input.
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
---
card_id: wenO5IJa
---
How many modes can a dataset have?
---
A dataset can have:
- **One mode** (unimodal)
- **Two modes** (bimodal)
- **Multiple modes** (multimodal)
---
card_id: rSiF58qj
---
What is the **Pearson correlation coefficient** formula?
---
$$r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

- $r$: Pearson correlation coefficient
- $\text{Cov}(X, Y)$: covariance between X and Y
- $\sigma_X$, $\sigma_Y$: standard deviations of X and Y
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
**Output layer** for **binary classification**.
---
card_id: dBiU2RDW
---
What is the **sigmoid** activation function formula?
---
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Outputs values between **0 and 1**.
---
card_id: xZvfKOc7
---
What is the **drawback** of using sigmoid in hidden layers?
---
Suffers from **vanishing gradients** - gradients become very small for large positive or negative inputs, slowing or stopping learning in deep networks. Not recommended for hidden layers.
---
card_id: tNzp58NI
---
What does **standard deviation** measure?
---
**Standard deviation** measures the **spread of individual data points** around the mean.
---
card_id: uXtcGBI0
---
Given a model with **99% training accuracy** but **60% test accuracy**, what's the problem?
---
**High variance / overfitting** - the model memorizes training data but doesn't generalize to new data.
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
**Receiver Operating Characteristic** curve plots **True Positive Rate (Recall, Sensitivity)** vs **False Positive Rate** at various classification thresholds.
---
card_id: SJ62qEWG
---
What are the axes of the **ROC curve**?
---
- X-axis: False Positive Rate = $\frac{FP}{FP + TN}$
- Y-axis: True Positive Rate = $\frac{TP}{TP + FN}$ (Recall, Sensitivity)
---
card_id: BKxniY3t
---
What is the formula for **covariance**?
---
$$\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$$
---
card_id: mSSyzDUL
---
Given **80 true positives (TP)** and **20 false positives (FP)**, calculate **precision (PPV)**.
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
What is the **L2 regularization** (Ridge) formula?
---
$$\text{Loss} = \text{Original Loss} + \lambda \sum_{i=1}^{n} w_i^2$$
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
---
card_id: MHu0Xv79
---
What is **overfitting**?
---
**Overfitting** occurs when a model learns the training data too well, including noise and random fluctuations, rather than just the underlying patterns.
---
card_id: k4yrW9xb
---
What are the characteristics of **overfitting**?
---
- High performance on training data
- Poor performance on new/test data
- Model has memorized rather than learned general patterns
---
card_id: kbW0MgoN
---
What is **underfitting**?
---
**Underfitting** occurs when a model is too simple to capture the underlying patterns in the data.
---
card_id: ozCnitK8
---
What are the characteristics of **underfitting**?
---
- Poor performance on both training and test data
- Model lacks the capacity to represent the true relationship
- Systematic errors due to oversimplification
---
card_id: o6esIpT3
---
What is **regularization** in machine learning?
---
**Regularization** adds a penalty term to the loss function to constrain model complexity and prevent overfitting.
---
card_id: KlEA3qfZ
---
What is a **confusion matrix**?
---
A **confusion matrix** is a table showing the four possible outcomes of binary classification: TP, FP, TN, and FN.
---
card_id: yLulG0eu
---
What is a **True Positive (TP)**?
---
A case where the model **correctly predicted the positive class**.
---
card_id: MMqxawJc
---
What is an example of a **True Positive (TP)**?
---
Model predicts "has disease" and the patient actually has the disease.
---
card_id: awlbwGll
---
What is a **False Positive (FP)**?
---
A case where the model **incorrectly predicted positive** when it was actually negative (Type I error). Also called a **false alarm**.
---
card_id: 3zUHQBdR
---
What is an example of a **False Positive (FP)**?
---
Model predicts "has disease" but the patient is healthy.
---
card_id: 54gRuV3l
---
What is a **True Negative (TN)**?
---
A case where the model **correctly predicted the negative class**.
---
card_id: 0dpgE0Qs
---
What is an example of a **True Negative (TN)**?
---
Model predicts "no disease" and the patient is indeed healthy.
---
card_id: 0pPYcDB3
---
What is a **False Negative (FN)**?
---
A case where the model **incorrectly predicted negative** when it was actually positive (Type II error). Also called a **miss**.
---
card_id: CM3V3D54
---
What is an example of a **False Negative (FN)**?
---
Model predicts "no disease" but the patient actually has the disease.
---
card_id: llct1zZ3
---
What is the **False Positive Rate** formula?
---
$$\text{FPR} = \frac{FP}{FP + TN} = \frac{FP}{\text{Total Actual Negatives}}$$

- $FP$: false positives
- $TN$: true negatives
---
card_id: gmpKQkfS
---
What does **False Positive Rate** measure?
---
**False Positive Rate** measures the fraction of actual negatives that were incorrectly predicted as positive.
---
card_id: dA69gJE7
---
What does **Shannon entropy** measure about a distribution?
---
**Entropy measures uncertainty or randomness** in a distribution.
---
card_id: I10LiBvY
---
Why does the **F1 score** use the **harmonic mean** instead of arithmetic mean?
---
The harmonic mean penalizes extreme imbalances between precision and recall.
---
card_id: kwcpMBRY
---
What is the **vanishing gradient problem**?
---
**Vanishing gradients** occur when gradients become extremely small during backpropagation, making weights update very slowly or stop learning entirely.
---
card_id: zQXVOCyB
---
What is the difference between **population** and **sample**?
---
**Population**: The complete set of all items/observations you're interested in studying (often impossible to measure fully).

**Sample**: A subset of the population actually observed/measured.
---
card_id: MW5HaGms
---
What is an example of **population** vs **sample**?
---
- Population: All voters in a country
- Sample: 1,000 voters surveyed
---
card_id: 4PojHsim
---
What is an **outlier**?
---
An **outlier** is a data point that differs significantly from other observations - unusually high or low compared to the rest of the data.
---
card_id: y86QRD9C
---
What are common causes of **outliers**?
---
- Natural variation (legitimate extreme values)
- Measurement errors
- Data entry errors
---
card_id: my3dqVrH
---
What is the impact of **outliers** on statistics?
---
Can strongly influence statistics like mean and variance, but not median.
---
card_id: OfGtTf7D
---
What is **generalization** in machine learning?
---
**Generalization** is the model's ability to perform well on new, unseen data - not just the training data.
---
card_id: 2GhxVoqI
---
What is the difference between **parameters** and **hyperparameters**?
---
**Parameters**: Learned by the model during training (e.g., weights, biases in neural networks).

**Hyperparameters**: Set before training and control the learning process (e.g., learning rate, number of layers, regularization strength λ, number of trees in random forest).
---
card_id: n5F6zJ00
---
What is a **mini-batch** in training?
---
A **mini-batch** is a small subset of the training data used to compute one gradient update during training.
---
card_id: 2TU18FW3
---
What are typical **mini-batch** sizes?
---
32, 64, 128, 256 samples
---
card_id: azECjrm6
---
What are the advantages of **mini-batch** training?
---
- Faster than processing full dataset per update
- More stable than single-example updates
- Enables efficient parallel computation on GPUs
---
card_id: q2OQkYy4
---
What is the relationship between **cross-entropy**, **entropy**, and **KL divergence**?
---
$$H(p, q) = H(p) + D_{KL}(p \| q)$$

**Cross-entropy** = **Shannon entropy** + **KL divergence**

**Minimizing cross-entropy** ≡ **minimizing KL divergence** (since $H(p)$ is constant during training).
---
card_id: OY9wn6JJ
---
What is **multicollinearity**?
---
**Multicollinearity** occurs when predictor variables are highly correlated with each other.
---
card_id: q6g3n0k9
---
What problems does **multicollinearity** cause?
---
- Makes it hard to determine individual feature importance
- Causes unstable coefficient estimates (small data changes → large coefficient changes)
- Inflates variance of coefficient estimates
---
card_id: DZTZ6CM4
---
How does **L2 regularization** help with **multicollinearity**?
---
L2 regularization distributes weights across correlated features rather than putting all weight on one.
---
card_id: QE3CgMAI
---
What is **ensemble learning**?
---
**Ensemble learning** combines predictions from multiple models to make better predictions than any single model.
---
card_id: tweun77n
---
What is the key idea behind **ensemble learning**?
---
Different models make different errors; averaging reduces overall error.
---
card_id: EUCiccyR
---
What are common **ensemble learning** methods?
---
- Bagging (e.g., Random Forest)
- Boosting (e.g., XGBoost, AdaBoost)
- Stacking
---
card_id: tLzMaETx
---
What is a **receptive field** in convolutional networks?
---
The **receptive field** is the region of the input that affects a particular neuron's activation.
---
card_id: P70AZoOs
---
How do **receptive fields** differ across CNN layers?
---
- Early layers: Small receptive fields (local patterns like edges)
- Deeper layers: Larger receptive fields (global patterns like shapes/objects)
---
card_id: vHPb54CB
---
How does stacking convolutional layers affect the **receptive field**?
---
Receptive field grows as you stack more convolutional layers, allowing neurons to "see" larger portions of the input.
---
card_id: zU2gnz2Z
---
What is the **softmax** activation function?
---
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

**Softmax** converts a vector of values into a probability distribution.
---
card_id: luDtAq8K
---
What are the properties of **softmax**?
---
- Outputs sum to 1
- All outputs between 0 and 1
- Emphasizes the largest value
---
card_id: REQNgnFU
---
When is **softmax** used in neural networks?
---
Output layer for **multi-class classification** (3+ classes).
---
card_id: QydEx1Yk
---
What is the **Leaky ReLU** activation function?
---
$$\text{Leaky ReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

where $\alpha$ is a small constant (e.g., 0.01).

**Leaky ReLU** allows a small negative slope instead of zero for negative inputs.
---
card_id: 7SSCNALh
---
What is the advantage of **Leaky ReLU** over regular ReLU?
---
Solves the dying ReLU problem - neurons can still learn even when inputs are negative.
---
card_id: IkstoadS
---
What is an **epoch** in training?
---
An **epoch** is one complete pass through the entire training dataset.
---
card_id: fogyKaEy
---
Why do models train for multiple **epochs**?
---
Model needs to see examples multiple times to learn patterns effectively. Number of epochs is a hyperparameter.
---
card_id: jTjjyJkj
---
What is **backpropagation**?
---
**Backpropagation** is the algorithm used to compute gradients of the loss function with respect to model parameters.
---
card_id: odVFhb1i
---
How does **backpropagation** work?
---
1. Forward pass: Compute predictions and loss
2. Backward pass: Compute gradients by applying chain rule backward through layers
3. Update weights using gradients
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
card_id: 46tJr1fo
---
What are the main **gradient descent** variants?
---
Batch GD, Stochastic GD (SGD), Mini-batch GD.
---
card_id: vqpUWcWe
---
What is **learning rate** in gradient descent?
---
**Learning rate** ($\alpha$) controls the step size when updating parameters during optimization.
---
card_id: hWeUjPXO
---
What happens with **too high** vs **too low** learning rates?
---
**Too high**: Training is unstable, may overshoot minimum, may diverge

**Too low**: Training is very slow, may get stuck in local minima
---
card_id: a5L3wP0o
---
What are common **learning rate** values?
---
0.001, 0.01, 0.1
---
card_id: BbAmc3iV
---
What is **batch gradient descent**?
---
**Batch gradient descent** computes the gradient using the **entire training dataset** before making one parameter update (once per epoch).
---
card_id: QXDLMfNl
---
What are the advantages and disadvantages of **batch gradient descent**?
---
**Advantages**: Stable, smooth convergence

**Disadvantages**: Very slow for large datasets, requires all data in memory
---
card_id: H1xqU7vY
---
What is **stochastic gradient descent (SGD)**?
---
**Stochastic gradient descent** computes the gradient using **one random training example** at a time (once per training example).
---
card_id: PIRKmMCj
---
What are the advantages and disadvantages of **stochastic gradient descent (SGD)**?
---
**Advantages**: Fast updates, can escape local minima due to noise

**Disadvantages**: Noisy updates, erratic convergence path
---
card_id: acNElkYI
---
What is **mini-batch gradient descent**?
---
**Mini-batch gradient descent** computes the gradient using a **small batch** of training examples (e.g., 32, 64, 128).
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
card_id: UI9T8OSk
---
When does **Shannon entropy** equal zero?
---
Entropy equals **zero** when the distribution is **deterministic** - one outcome has probability 1, all others have probability 0.
---
card_id: NfsaxG5y
---
What is a **Type I error**?
---
**Type I error** (false positive) occurs when you reject a true null hypothesis - incorrectly detecting an effect that doesn't exist. Often denoted by $\alpha$ (significance level).
---
card_id: 8W66UYWY
---
What is an example of a **Type I error**?
---
A medical test says a healthy patient has a disease.
---
card_id: krF3IVXZ
---
What is a **Type II error**?
---
**Type II error** (false negative) occurs when you fail to reject a false null hypothesis - missing an effect that does exist. Often denoted by $\beta$. Power = $1 - \beta$.
---
card_id: xnL5TQ3p
---
What is an example of a **Type II error**?
---
A medical test says a sick patient is healthy.
---
card_id: O6dyoh2N
---
What is a **loss function**?
---
A **loss function** (or cost function) quantifies how wrong the model's predictions are compared to the true values.
---
card_id: C365ec1B
---
What is the purpose of a **loss function** in training?
---
Provides a single numerical value to minimize during training.
---
card_id: lhNODfQa
---
What are common **loss function** examples?
---
- Mean Squared Error (MSE) for regression
- Cross-entropy for classification
- Mean Absolute Error (MAE) for regression
---
card_id: y8PrSSwI
---
What is an **activation function**?
---
An **activation function** introduces non-linearity into neural networks by transforming a neuron's weighted input.
---
card_id: tWLEiv2b
---
Why are **activation functions** needed in neural networks?
---
Without activation functions, even deep networks would only learn linear relationships.
---
card_id: zFsckFHr
---
What are common **activation function** examples?
---
ReLU, sigmoid, tanh, softmax, Leaky ReLU.
---
card_id: 0Q7OpQ43
---
What is an **optimizer** in machine learning?
---
An **optimizer** is an algorithm that adjusts model parameters to minimize the loss function.
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
card_id: e8KGqe0r
---
What is a **forward pass** in neural networks?
---
**Forward pass** (or forward propagation) is the process of computing predictions by passing input data through the network layers sequentially.
---
card_id: 8y1vGggu
---
What are the steps in a **forward pass** through a neural network?
---
1. Input enters the first layer
2. Each layer applies weights, biases, and activation functions
3. Output emerges from final layer
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
card_id: bFIJbsU5
---
What is **average pooling**?
---
**Average pooling** downsamples by taking the **average value** within each pooling window (e.g., a 2×2 window takes the mean of 4 values).
---
card_id: nmog7z9T
---
How does **average pooling** compare to **max pooling**?
---
Average pooling preserves overall intensity, while max pooling preserves strongest features.
---
card_id: KER0dp8z
---
What is **bagging** in ensemble learning?
---
**Bagging** (Bootstrap Aggregating) trains multiple models on different random subsets of training data, then averages their predictions.
---
card_id: AFB2UYzc
---
How does **bagging** work?
---
1. Create bootstrap samples (random sampling with replacement)
2. Train one model on each sample
3. Average predictions (regression) or vote (classification)
---
card_id: PZhGvfXU
---
What is an example of **bagging**?
---
Random Forest uses bagging with decision trees.
---
card_id: 4cZEhE2o
---
What is the benefit of **bagging**?
---
Reduces variance by averaging diverse models.
---
card_id: gbTk1KMQ
---
What is **boosting** in ensemble learning?
---
**Boosting** trains models sequentially, where each new model focuses on correcting errors made by previous models.
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
card_id: 3Kx0iMdb
---
What are examples of **boosting** algorithms?
---
AdaBoost, XGBoost, Gradient Boosting.
---
card_id: ZQKGKnlH
---
What is the benefit of **boosting**?
---
Reduces both bias and variance by building strong learners from weak ones.
---
card_id: C0EWjq3B
---
What is the difference between **bagging** and **boosting**?
---
**Bagging**: Models trained **in parallel** on independent samples, reduces **variance**

**Boosting**: Models trained **sequentially** where later models correct earlier mistakes, reduces **bias** and **variance**
---
card_id: 8znuzNfd
---
How do **bagging** and **boosting** compare in terms of overfitting?
---
Bagging is more robust to overfitting; boosting can achieve higher accuracy but may overfit if not careful.
---
card_id: yiJaLVoL
---
What is **feature engineering**?
---
**Feature engineering** is the process of creating, transforming, or selecting features to improve model performance.
---
card_id: 71ghEQCy
---
What are examples of **feature engineering** techniques?
---
- Creating interaction terms (e.g., $x_1 \times x_2$)
- Polynomial features (e.g., $x^2$, $x^3$)
- Domain-specific features (e.g., extracting day-of-week from timestamps)
- Normalization/scaling
- Encoding categorical variables
---
card_id: BTfmEblr
---
What is **one-hot encoding**?
---
**One-hot encoding** converts categorical variables into binary vectors where only one element is 1 and the rest are 0.
---
card_id: lvhkdPE5
---
What is an example of **one-hot encoding**?
---
For colors {red, green, blue}:
- Red → [1, 0, 0]
- Green → [0, 1, 0]
- Blue → [0, 0, 1]
---
card_id: Akrf5AmB
---
What is **model capacity**?
---
**Model capacity** refers to the model's ability to fit a variety of functions - its flexibility and complexity.
---
card_id: 2MVCTUPZ
---
What characterizes **high capacity** vs **low capacity** models?
---
**High capacity**: Many parameters, can fit complex patterns (risk overfitting)

**Low capacity**: Few parameters, limited flexibility (risk underfitting)
---
card_id: tOmGjZ9r
---
What are examples of **low** and **high capacity** models?
---
- Low capacity: Linear regression
- High capacity: Deep neural network with many layers
---
card_id: w8C2e3CL
---
What is the **training set**?
---
The **training set** is the portion of data used to train the model - to learn parameters (weights, biases) by minimizing the loss function. The model sees and learns from this data directly.
---
card_id: XYXFvqS6
---
What is a typical **training set** split percentage?
---
60-80% of total data
---
card_id: Cwdj2N2S
---
What is the **validation set**?
---
The **validation set** is data used to tune hyperparameters and make model selection decisions during development. Can be used repeatedly during development.
---
card_id: 5WFLPHyv
---
What is a typical **validation set** split percentage?
---
10-20% of total data
---
card_id: 0n4KEKOP
---
What is the **validation set** used for?
---
Choosing learning rate, regularization strength, model architecture, early stopping.
---
card_id: YRMA3HJw
---
What is the **test set**?
---
The **test set** is data held out for final evaluation only - never used during training or model selection. Used once after all development is complete, to get an unbiased performance estimate.
---
card_id: MxO1HdII
---
What is a typical **test set** split percentage?
---
10-20% of total data
---
card_id: fAcjI5tK
---
Why must the **test set** never be used for training decisions?
---
Using it for training decisions causes you to overfit to it, making performance estimates biased.
---
card_id: TuGtEtRP
---
What is a **neural network layer**?
---
A **layer** is a collection of neurons that process inputs together and produce outputs. Each layer typically applies: linear transformation (weights + biases) → activation function.
---
card_id: 6ZDIFryu
---
What are the types of **neural network layers**?
---
- **Input layer**: Receives raw features
- **Hidden layers**: Intermediate transformations
- **Output layer**: Final predictions
---
card_id: VhOD3ekT
---
What is the **exploding gradient problem**?
---
**Exploding gradients** occur when gradients become extremely large during backpropagation, causing unstable training.
---
card_id: PmKl4R7g
---
What causes the **exploding gradient problem**?
---
- Deep networks with poor initialization
- Weights that amplify signals through layers
- Certain activation functions
---
card_id: r4HoxfKH
---
What are consequences of the **exploding gradient problem**?
---
- Weights update too drastically
- Training diverges (loss becomes NaN)
- Model fails to converge
---
card_id: Bm9pamfC
---
What are solutions to the **exploding gradient problem**?
---
Gradient clipping, proper initialization, batch normalization.
---
card_id: PnHbz8zJ
---
What is **gradient clipping**?
---
**Gradient clipping** limits the magnitude of gradients during training to prevent exploding gradients.
---
card_id: SIvSjQpC
---
What are the **gradient clipping** methods?
---
- **Clip by value**: Cap gradients at threshold (e.g., [-5, 5])
- **Clip by norm**: Scale gradient vector if its norm exceeds threshold
---
card_id: mqPH4KbK
---
When should you use **gradient clipping**?
---
Recurrent neural networks, very deep networks, when you observe exploding gradients.
---
card_id: p8NVrN3o
---
What are **weights** in neural networks?
---
**Weights** are learnable parameters that determine the strength of connections between neurons.
---
card_id: N4D5F4C9
---
What is **bias** (the parameter) in neural networks?
---
**Bias** is a learnable parameter added to the weighted sum before the activation function.

$$y = f(w_1x_1 + w_2x_2 + ... + w_nx_n + b)$$
---
card_id: 4Vn2Hdjs
---
What is the purpose of the **bias** parameter in neural networks?
---
Allows the activation function to shift left or right, increasing model flexibility. Without bias, a neuron with ReLU would always output 0 when all inputs are 0.
---
card_id: 7i0phiJv
---
What is the difference between **bias** (the parameter) and **bias** (the error)?
---
**Bias (parameter)**: A learnable value added to weighted sums in neural networks (notation: $b$).

**Bias (error)**: The systematic error when a model's average predictions miss the true function - a measure of underfitting.

Same word, completely different meanings - context determines which one is meant.
---
card_id: nPBdbFwA
---
What is **feature scaling**?
---
**Feature scaling** transforms features to similar ranges to improve training performance.
---
card_id: exsTnOs7
---
What are common **feature scaling** methods?
---
Normalization (Min-Max scaling) and Standardization (Z-score normalization).
---
card_id: Wj8qng64
---
Why is **feature scaling** needed?
---
Features with large ranges can dominate gradient updates and slow convergence.
---
card_id: ZDwcXyr0
---
What is **normalization** in feature scaling?
---
**Normalization** (Min-Max scaling) scales features to a fixed range, typically [0, 1].

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$
---
card_id: 0Wk1TIQr
---
When should you use **normalization** (Min-Max scaling)?
---
You need bounded values (e.g., for neural networks with sigmoid/tanh).
---
card_id: tiSod4EC
---
What is a disadvantage of **normalization** (Min-Max scaling)?
---
Sensitive to outliers (they determine min/max).
---
card_id: Lj7tR0qb
---
What is **standardization** in feature scaling?
---
**Standardization** (Z-score normalization) transforms features to have mean=0 and standard deviation=1.

$$x' = \frac{x - \mu}{\sigma}$$

where $\mu$ is mean and $\sigma$ is standard deviation.
---
card_id: 9pCunREq
---
What is a **decision boundary**?
---
A **decision boundary** is the surface that separates different classes in the feature space.
---
card_id: tWnFYWpm
---
What are examples of **decision boundaries** in different dimensions?
---
**Linear classifiers**:
- 2D space: Straight line
- 3D space: Plane
- Higher dimensions: Hyperplane

**Non-linear classifiers**: Curved boundaries
---
card_id: ZJsw6FAA
---
How does **model complexity** affect decision boundary shape?
---
- **Simple models** create simple boundaries
- **Complex models** create intricate boundaries
---
card_id: kpCPzJdm
---
What is a **hyperplane**?
---
A **hyperplane** is a flat (linear) subspace that divides a higher-dimensional space.
---
card_id: ylaQJBy0
---
What are **hyperplane** dimensions in different spaces?
---
- 1D space: Point
- 2D space: Line
- 3D space: Plane
- N-D space: Hyperplane (N-1 dimensions)
---
card_id: HNzOigX9
---
How are **hyperplanes** used in machine learning?
---
Linear classifiers (like SVM) create hyperplane decision boundaries.
---
card_id: 3NrwoKQG
---
What is **Recall** also known as in AI/ML?
---
**Recall** is also known as **Sensitivity** or **True Positive Rate (TPR)**.
---
card_id: NBmcO83a
---
What is **Sensitivity** also known as in AI/ML?
---
**Sensitivity** is also known as **Recall** or **True Positive Rate (TPR)**.
---
card_id: HAgpg5UB
---
What is **True Positive Rate (TPR)** also known as in AI/ML?
---
**True Positive Rate (TPR)** is also known as **Recall** or **Sensitivity**.
---
card_id: 0JgCdm9w
---
What is **Precision** also known as in AI/ML?
---
**Precision** is also known as **PPV (Positive Predictive Value)**.
---
card_id: b5SI03Z6
---
What is **PPV (Positive Predictive Value)** also known as in AI/ML?
---
**PPV (Positive Predictive Value)** is also known as **Precision**.
---
card_id: frs3nlIG
---
What metric is best for detecting when a model is good for **credit card fraud detection**, where both false positives (blocking legitimate transactions) and false negatives (missing fraud) have significant costs?
---
**F1-score** - provides a balanced measure between precision and recall, appropriate when both types of errors matter.
---
card_id: EuCNcuYm
---
What metric is best for detecting when a model is good for **ranking search results**, where top results matter more than lower-ranked ones?
---
**NDCG (Normalized Discounted Cumulative Gain)** - accounts for both relevance and position, with higher weight on top-ranked results.
---
card_id: xwvUf1wh
---
What metric is best for detecting when a **language model** produces coherent and accurate text?
---
**Perplexity** - measures how well the probability distribution predicts the sample, with lower values indicating better predictive performance.
---
card_id: Ny9AunSu
---
What metric is best for detecting when a model is good for **medical screening** where false positives lead to expensive unnecessary follow-up procedures?
---
**Precision** - minimizes false positives to reduce unnecessary costly procedures while ensuring positive predictions are reliable.
---
card_id: qVAjG5ZN
---
What metric is best for detecting when a model is good for **image segmentation**, measuring how well predicted regions overlap with ground truth?
---
**IoU (Intersection over Union)** - measures the overlap between predicted and actual regions, ranging from 0 (no overlap) to 1 (perfect match).
---
card_id: RHAd6hj7
---
What metric is best for detecting when a model is good for **predicting house prices** when you want the error in the original dollar units?
---
**RMSE (Root Mean Squared Error)** - provides error in the same units as the target variable (dollars), making it interpretable and emphasizing larger errors.
---
card_id: EkSzU21a
---
What metric is best for detecting when a model is good for **binary classification with severe class imbalance** (e.g., 99% negative, 1% positive)?
---
**MCC (Matthews Correlation Coefficient)** - ranges from -1 to +1 and provides a balanced measure even with extreme class imbalance, using all confusion matrix elements.
---
card_id: UPI9tbWF
---
What metric is best for detecting when a model is good for **comparing classifier performance across different decision thresholds**?
---
**AUC-ROC (Area Under ROC Curve)** - threshold-independent metric that evaluates performance across all possible classification thresholds.
---
card_id: qSvHXJ4Y
---
What metric is best for detecting when a model is good for **rare disease detection**, where correctly identifying healthy patients is important for resource allocation?
---
**Specificity (True Negative Rate)** - measures the model's ability to correctly identify negative cases, crucial when most cases are negative.
---
card_id: 4kSFetKT
---
What metric is best for detecting when a model is good for **binary classification with imbalanced classes**, accounting for per-class accuracy?
---
**Balanced Accuracy** - averages recall across classes, giving equal weight to each class regardless of size.
---
card_id: oBLYRI4i
---
What metric is best for detecting when a model is good for **probabilistic predictions**, where calibrated probabilities matter more than just class labels?
---
**Log Loss (Cross-Entropy)** - directly evaluates the quality of predicted probabilities, heavily penalizing confident wrong predictions.
---
card_id: TvwkOqHv
---
What metric is best for detecting when a **machine translation model** produces quality translations?
---
**BLEU (Bilingual Evaluation Understudy)** - measures n-gram overlap between predicted and reference translations.
---
card_id: twHZY5bV
---
What metric is best for detecting when a model is good for **predicting house prices** when you want to be robust to outliers (extreme price anomalies)?
---
**MAE (Mean Absolute Error)** - treats all errors equally without squaring, making it less sensitive to outliers than MSE.
---
card_id: wuNF9xk4
---
What metric is best for detecting when a model is good for **predicting stock prices** where large errors should be penalized more heavily than small ones?
---
**MSE (Mean Squared Error)** - squares the errors, giving disproportionately higher penalty to larger deviations.
---
card_id: UuQpx1Pc
---
What metric is best for detecting when a regression model is good for **explaining variance** in the target variable?
---
**R² (Coefficient of Determination)** - measures the proportion of variance explained by the model, ranging from 0 to 1.
---
card_id: 0OwFILIh
---
What metric is best for detecting when a model is good for **object detection**, measuring bounding box prediction accuracy?
---
**IoU (Intersection over Union)** - computes overlap between predicted and ground truth bounding boxes, with threshold (e.g., IoU > 0.5) determining detection success.
---
card_id: gGruzl4g
---
What metric is best for detecting when a model is good for **recommender system ranking**, where retrieving relevant items near the top matters?
---
**MAP (Mean Average Precision)** - averages precision at each relevant item position, emphasizing early retrieval of relevant items.
---
card_id: JbroXq1F
---
What metric is best for detecting when a model is good for **multi-class classification with balanced classes** and no differential cost between error types?
---
**Accuracy** - simple and interpretable proportion of correct predictions, appropriate when all classes have similar size and importance.
---
card_id: uw2701Cr
---
What metric is best for detecting when a model is good for **anomaly detection in network security**, where catching attacks is critical but normal traffic vastly outnumbers attacks?
---
**Recall** combined with **Precision-Recall AUC** - prioritizes catching attacks (high recall) while PR-AUC evaluates performance across thresholds for imbalanced data.
---
card_id: 6FnoQTcG
---
What does **covariance** measure?
---
Measures how two variables change together - whether they increase/decrease in tandem (positive covariance) or opposite directions (negative covariance).
---
card_id: j6PIg78N
---
How does **covariance** differ from **correlation**?
---
**Covariance**: Scale-dependent, unbounded range, units = (units of X) × (units of Y)

**Correlation**: Scale-independent (normalized covariance), dimensionless, range -1 to +1
---
card_id: PkpGz6fh
---
What does **standard error** measure?
---
Measures the variability of the **sample mean** - how much sample means would vary if you repeatedly sampled from the same population.
---
card_id: V3WPCUOB
---
How does **standard error** differ from **standard deviation**?
---
**Standard deviation**: Measures spread of individual data points

**Standard error**: Measures precision of the sample mean estimate
---
card_id: KDnFOba5
---
What does **R-squared** measure in regression?
---
Measures the proportion of variance in the target variable explained by the model.
---
card_id: UmHtIgEm
---
How is **R-squared** interpreted?
---
- **R² = 0**: Model explains no variance (predicts the mean)
- **R² = 0.7**: Model explains 70% of variance
- **R² = 1**: Perfect predictions

Higher R² indicates better fit.
---
card_id: ldCT7lDi
---
What is **harmonic mean**?
---
The **harmonic mean** is the reciprocal of the arithmetic mean of reciprocals, giving more weight to smaller values.
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
card_id: FV2Pn2Kh
---
How is **batch normalization** applied differently at test time vs training time?
---
**Training**: Uses current mini-batch statistics (mean and variance)

**Test**: Uses running averages of mean and variance computed during training (fixed statistics)
---
card_id: 2hIwt9d2
---
How does **max pooling** work step-by-step?
---
1. Divide input into non-overlapping regions (e.g., 2×2 windows)
2. Take the maximum value from each region
3. Output the max values, reducing spatial dimensions
---
card_id: Xvi9k4mM
---
When should you use **max pooling** vs **average pooling**?
---
**Max pooling**: When you want to detect presence of features (dominant signal)

**Average pooling**: When you want to preserve overall information and reduce noise
---
card_id: dRyvIBOR
---
What is a **kernel (filter)** in convolutional layers?
---
A **kernel** is a small learnable weight matrix (e.g., 3×3) that slides across the input to detect specific patterns like edges, textures, or shapes.
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
card_id: FhreUGhm
---
What is **information gain**?
---
**Information gain** measures the reduction in entropy (uncertainty) achieved by splitting data on a particular feature.
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
card_id: kqu7riud
---
**Validation loss** stops decreasing at epoch 5 but **training loss** keeps dropping. What should you do?
---
**Overfitting detected** - use early stopping to stop training around epoch 5, or add regularization (dropout, L1/L2, data augmentation) and retrain.
---
card_id: la8zY9yL
---
**Training loss** oscillates wildly and never converges. What's the likely cause?
---
**Learning rate is too high** - the optimizer overshoots the minimum with each step. Solution: Reduce learning rate by 10x (e.g., 0.1 → 0.01).
---
card_id: qYAfqnzI
---
**Training loss** decreases extremely slowly (0.001 per epoch). What's likely wrong?
---
**Learning rate is too low** - steps are too small to reach the minimum efficiently. Solution: Increase learning rate by 10x.
---
card_id: tXoAhkAl
---
You have 1000 features but suspect only 50 are relevant. Should you use **L1** or **L2 regularization**?
---
**Use L1 (Lasso)** - it performs automatic feature selection by driving irrelevant feature weights to exactly zero, creating a sparse model with ~50 non-zero weights.
---
card_id: YpXEpl2q
---
Your binary classifier has 90% **precision** but 20% **recall**. What does this mean about the model's behavior?
---
The model is **extremely conservative** - it only predicts positive when very confident (high precision), but misses most positive cases (low recall). It's producing very few positive predictions.
---
card_id: qvG1VoHc
---
**Accuracy** is 95% but **F1 score** is 0.30. What does this tell you about your dataset?
---
**Severe class imbalance** - the model is likely predicting the majority class for almost everything, achieving high accuracy but performing poorly on the minority class (low F1).
---
card_id: vs8j84yd
---
What is the tradeoff between **precision** and **recall**?
---
**Precision-Recall Tradeoff**: Lowering the classification threshold increases recall (catches more positives) but decreases precision (more false alarms). They move inversely.
---
card_id: i4YfDU8Q
---
Why does **L1 regularization** drive weights to exactly zero while **L2** does not?
---
L1 uses absolute value $|w|$ which has constant gradient (±1), pushing weights by fixed amounts toward zero regardless of size. L2 uses $w^2$ with gradient proportional to $w$, so penalty weakens as weights approach zero.
---
card_id: 3FMKUz7B
---
What is **data augmentation**?
---
**Data augmentation** creates modified versions of training examples through transformations (rotation, flipping, cropping, color shifts) to artificially expand dataset size.
---
card_id: o9sVPuyT
---
What are common **data augmentation** techniques for images?
---
- Geometric: rotation, flipping, cropping, scaling
- Color: brightness/contrast adjustment, color jittering
- Noise: adding Gaussian noise, blur
- Advanced: cutout, mixup
---
card_id: ryru7Jwu
---
When is **data augmentation** most effective?
---
Most effective when:
- Limited training data
- Model shows overfitting
- Training data doesn't cover all variations (e.g., rotations, lighting conditions)
- Domain allows realistic transformations
---
card_id: ZH3j98Ju
---
What is **early stopping**?
---
**Early stopping** is a regularization technique that stops training when validation performance stops improving, preventing overfitting.
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
card_id: ytrYDXIz
---
When should you use **normalization** vs **standardization**?
---
**Normalization (Min-Max)**: When you need bounded values in specific range (e.g., [0,1]) for neural networks with sigmoid/tanh

**Standardization (Z-score)**: When features have different units/scales and you want zero-centered data (more robust to outliers)
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
card_id: znraY4zM
---
What are the tradeoffs between **batch**, **stochastic**, and **mini-batch gradient descent**?
---
**Batch GD**: Stable but slow, requires all data in memory

**Stochastic GD**: Fast updates but noisy, erratic convergence

**Mini-batch GD**: Best balance - faster than batch, more stable than stochastic, efficient GPU usage
---
card_id: bo9ChDN9
---
What is **model complexity**?
---
**Model complexity** refers to the model's capacity to fit various functions - determined by number of parameters, depth, and flexibility.
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
card_id: 4ADzRBcr
---
How is an **ROC curve** constructed?
---
1. Train classifier that outputs probabilities
2. Vary classification threshold from 0 to 1
3. At each threshold, compute TPR (y-axis) and FPR (x-axis)
4. Plot points and connect them to form curve
---
card_id: kDzie0bq
---
What is the tradeoff between **Type I** and **Type II errors**?
---
**Type I vs Type II Tradeoff**: Reducing significance level $\alpha$ (stricter threshold) decreases Type I errors (false positives) but increases Type II errors (false negatives). More conservative tests catch fewer true effects.
---
card_id: iJ4AzCoM
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
