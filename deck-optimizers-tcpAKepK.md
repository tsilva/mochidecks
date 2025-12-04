---
card_id: EGmg9Tqk
---
What is **Stochastic Gradient Descent (SGD)**?
---
An optimizer that updates parameters by taking steps in the opposite direction of the gradient.
---
card_id: PIZ1zBHj
---
What is the **SGD update rule**?
---
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

where `$\theta_t$`: parameters at step t, `$\eta$`: learning rate, `$\nabla L(\theta_t)$`: gradient of loss
---
card_id: Ily6DnXZ
---
What are **advantages of SGD**?
---
Simple and reliable
Low memory usage (O(1))
Works well with proper learning rate
---
card_id: 2BJYMigy
---
When does **SGD oscillate** during optimization?
---
In elongated valleys where different dimensions have different scales, causing zigzag patterns perpendicular to the optimal direction.
---
card_id: qZ1c4pXk
---
What are **disadvantages of SGD**?
---
Sensitive to learning rate choice
Slow in ravines
Oscillates in narrow valleys
Uses same learning rate for all parameters
---
card_id: mopTWWJ3
---
What problem does **momentum** solve in optimization?
---
SGD has no memory of previous steps, causing oscillations and slow progress.
---
card_id: avGzXDB0
---
What is **momentum** in neural network optimization?
---
An optimizer enhancement that accumulates velocity in persistent gradient directions to smooth trajectories.
---
card_id: 0MwbfDUS
---
What are the **update equations for momentum**?
---
$$v_{t+1} = \beta v_t + \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

where `$v_t$`: velocity, `$\beta$`: momentum coefficient (typically 0.9)
---
card_id: QRNCkHJk
---
How does **momentum accelerate** optimization?
---
Accumulates velocity in consistent gradient directions while dampening oscillations in varying directions.
---
card_id: Hs2TrYzp
---
What is the typical value for the **momentum coefficient** β?
---
0.9 (retains 90% of previous velocity, effectively remembering the last ~10 gradients).
---
card_id: ndY3oL5o
---
How does **high momentum** (β close to 1) affect optimization?
---
Creates very smooth trajectories but can overshoot minima due to slow changes in direction.
---
card_id: 0Fv99pes
---
What is the **memory complexity** of momentum compared to SGD?
---
O(n) for momentum (stores velocity for each parameter) vs O(1) for SGD (no state maintained).
---
card_id: lFZimfhv
---
What problem does **RMSprop** solve?
---
SGD and momentum use the same learning rate for all parameters, but different parameters need different update scales.
---
card_id: rHzpclLW
---
What is **RMSprop** (Root Mean Square Propagation)?
---
An optimizer that adapts learning rate per parameter based on recent gradient magnitudes.
---
card_id: JDNsYRj5
---
What are the **update equations for RMSprop**?
---
$$s_{t+1} = \beta s_t + (1 - \beta) (\nabla L(\theta_t))^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_{t+1} + \epsilon}} \nabla L(\theta_t)$$

where `$s_t$`: running average of squared gradients, `$\beta$`: decay rate (0.9 or 0.99), `$\epsilon$`: stability constant (1e-8)
---
card_id: blPLk7xy
---
How does **RMSprop handle different parameter scales**?
---
Divides each gradient by the square root of its recent squared average, giving smaller effective learning rates to parameters with large gradients.
---
card_id: g6v0FoDn
---
Why does **RMSprop work well** on elongated loss landscapes?
---
Automatically reduces updates in steep directions and increases them in flat directions to navigate ravines directly.
---
card_id: TQ0IlU9Y
---
What is **Adam** optimizer?
---
An optimizer combining momentum and RMSprop with bias correction for robust optimization.
---
card_id: artI1K0b
---
What are the **update equations for Adam**?
---
$$m_{t+1} = \beta_1 m_t + (1 - \beta_1) \nabla L(\theta_t)$$
$$v_{t+1} = \beta_2 v_t + (1 - \beta_2) (\nabla L(\theta_t))^2$$
$$\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}, \hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \hat{m}_{t+1}$$

where `$m_t$`: first moment (momentum), `$v_t$`: second moment (variance)
---
card_id: 5IxV4knB
---
What are the **typical hyperparameters** for Adam?
---
Learning rate: 0.001, β₁: 0.9 (momentum), β₂: 0.999 (RMSprop), ε: 1e-8 (stability).
---
card_id: 3x0XSSF8
---
Why does **Adam use bias correction** ($$\hat{m}$$ and $$\hat{v}$$)?
---
Both moments initialize to zero, causing bias toward zero in early iterations; correction terms amplify early estimates to compensate.
---
card_id: 9HVQisw2
---
How does the **bias correction factor** change over time in Adam?
---
It starts high (amplifying early estimates) and quickly approaches 1 as the optimizer accumulates history, then has negligible effect.
---
card_id: icbq4ayC
---
What is the **memory complexity** of Adam?
---
O(2n) - stores both momentum and squared gradient running averages for each parameter.
---
card_id: xFr6be5u
---
When should you use **SGD** over adaptive optimizers?
---
**SGD**: Simple problems, reproducibility needs
**SGD + LR schedule**: Final performance tuning
**SGD**: Low memory constraints
---
card_id: Y3ijmFDO
---
When should you use **momentum** optimization?
---
Noisy gradients
Loss landscapes with ravines
Need faster convergence with smooth trajectories
---
card_id: TlWhcv0L
---
When should you use **RMSprop**?
---
Training RNNs
Non-stationary objectives
Sparse gradients
Parameters with vastly different scales
---
card_id: RKtltmdr
---
When should you use **Adam** optimizer?
---
Default choice for most deep learning tasks due to robust performance and good out-of-the-box results.
---
card_id: aEwvd8mN
---
How does **SGD compare to Adam** on elongated loss landscapes?
---
SGD oscillates perpendicular to the valley direction, while Adam adapts learning rates to navigate directly toward the minimum.
---
card_id: YdY1WaJq
---
What effect does **learning rate scheduling** have on SGD?
---
Makes SGD competitive with adaptive methods by allowing large early steps and fine-tuning with smaller steps later.
---
card_id: uVBtfrZK
---
What makes the **Rosenbrock function** challenging for optimizers?
---
Narrow banana-shaped valley where one dimension changes much more slowly than the other.
---
card_id: XWf4njT6
---
What is an **elongated bowl** (ravine) in optimization?
---
A loss landscape where different dimensions have vastly different curvatures, causing oscillations in steeper directions.
---
card_id: dSJcARor
---
What is an example of an **elongated bowl** loss function?
---
$$x^2 + 10y^2$$ where the y-direction is 10 times steeper than x-direction.
---
card_id: R2VIbpDb
---
What is a **saddle point** in loss landscapes?
---
A point where gradient is zero but it's a minimum in some directions and maximum in others.
---
card_id: 0mgdLH6w
---
What is an example of a **saddle point** function?
---
$$x^2 - y^2$$ which curves up in x-direction and down in y-direction.
---
card_id: JDAMJA7y
---
How does **momentum help escape saddle points**?
---
Accumulated velocity carries the optimizer through flat regions where gradients are small.
---
card_id: cM2XRrKI
---
What is the recommended **initial learning rate** for Adam?
---
0.001 (default works well across most problems).
---
card_id: Vx2PxXeM
---
What is the recommended **initial learning rate** for SGD?
---
0.01 to 0.1 (problem-dependent, often requires tuning and scheduling).
---
card_id: kPbE2XGj
---
What is **AdamW**?
---
A variant of Adam with decoupled weight decay for better regularization.
---
card_id: 2zS3Ljna
---
What advantage does **RMSprop have for RNNs**?
---
Handles non-stationary objectives and varying gradient scales across time steps common in sequential data.
---
card_id: x6SmF13b
---
Why might **SGD with momentum** outperform Adam in some cases?
---
Can achieve better generalization with proper learning rate scheduling while using less memory.
---
card_id: STBOaDYN
---
What is the **effective learning rate** in RMSprop?
---
$$\frac{\eta}{\sqrt{s_t + \epsilon}}$$ where each parameter gets its own effective rate based on its gradient history.
---
card_id: O4xUQlVs
---
How does **batch size affect** optimizer performance?
---
**Larger batches**: More stable gradients, can use larger learning rates
**Smaller batches**: Noisier gradients, can help escape local minima
---
card_id: 9Utq74nE
---
What is the purpose of **ε (epsilon)** in RMSprop and Adam?
---
Small constant (typically 1e-8) for numerical stability to prevent division by zero.
---
card_id: 4EMPziRi
---
How does **momentum reduce oscillations**?
---
Accumulates velocity in consistent directions and cancels out components that change sign frequently.
---
card_id: HZrHTwZQ
---
What happens with **too high a learning rate** in SGD?
---
Optimizer oscillates wildly or diverges, overshooting minima.
---
card_id: YgJJnSoG
---
What happens with **too low a learning rate** in SGD?
---
Very slow convergence, may get stuck in local minima or saddle points.
---
card_id: H4WI2rM9
---
Why does **Adam work well out-of-the-box**?
---
Combines momentum for smooth trajectories, adaptive learning rates for different scales, and bias correction.
---
card_id: jeMXU2Le
---
Model oscillates in a narrow valley during training. Which optimizer helps most?
---
Momentum or adaptive methods (RMSprop/Adam) that dampen oscillations perpendicular to the valley direction.
---
card_id: YK0N1Vae
---
Training loss stuck on a plateau with small gradients. Which optimizer feature helps?
---
Momentum, which accumulates velocity to carry through flat regions and escape saddle points.
---
card_id: yaIISLOu
---
Different parameters have vastly different gradient scales. Which optimizer adapts best?
---
RMSprop or Adam, which normalize gradients per parameter based on their individual histories.
---
card_id: NkWec2O2
---
Training an RNN with varying gradient scales across time. Which optimizer is preferred?
---
RMSprop, designed for non-stationary objectives and handling sparse or varying gradients well.
---
card_id: ATWHf1FJ
---
Need fastest convergence with minimal hyperparameter tuning. Which optimizer to choose?
---
Adam with default hyperparameters (lr=0.001, β₁=0.9, β₂=0.999).
---
card_id: E8zEGRkZ
---
What is **velocity** in momentum optimization?
---
A running average of gradients that accumulates direction, like a heavy ball rolling downhill building up speed.
---
card_id: Iqyxn3DF
---
What are **non-stationary objectives** in optimization?
---
Loss landscapes where the optimal direction changes over time, common in RNNs processing sequential data.
---
card_id: 9fxSgcww
---
What is the **Beale function**?
---
A multimodal test function with multiple local minima, used to test optimizer robustness on challenging landscapes.
---
card_id: HmZCYos1
---
What is the **first moment** in Adam?
---
Running average of gradients (momentum component) that tracks the mean gradient direction.
---
card_id: 7II5e3QM
---
What is the **second moment** in Adam?
---
Running average of squared gradients (RMSprop component) that tracks gradient variance for adaptive learning rates.
---
card_id: FM4vLoEZ
---
What is the **bias correction formula** in Adam?
---
$$\hat{m} = \frac{m}{1 - \beta_1^t}$$ and $$\hat{v} = \frac{v}{1 - \beta_2^t}$$ where correction factor approaches 1 as t increases.
---
card_id: oQmy375u
---
What does β=0.9 mean for **effective window** in momentum?
---
The optimizer effectively "remembers" approximately the last 10 gradients.
---
card_id: f5uQabrO
---
What is **StepLR** learning rate scheduling?
---
Decays learning rate by a factor (gamma) every fixed number of epochs (step_size).
---
card_id: AsHrfzqf
---
Why does **RMSprop give smaller updates** to steep dimensions?
---
Steep dimensions have large gradients, so large squared gradient average in denominator reduces effective learning rate.
---
card_id: NxBBoNTE
---
Training starts well but loss becomes NaN after a few epochs. Likely cause?
---
Learning rate too high causing exploding gradients, or poor initialization leading to numerical instability.
---
card_id: KQ1SuHEB
---
Model trains very slowly, loss barely decreases after many epochs. Likely cause?
---
Learning rate too low, or optimizer stuck in saddle point/flat region without momentum.
---
card_id: LcWu886k
---
SGD trains well but Adam converges to worse test accuracy. Why?
---
Adam's adaptive rates may lead to sharp minima with poor generalization; SGD finds flatter minima.
---
card_id: qAxotQI2
---
Different layers have vastly different loss gradients. Which optimizer component helps?
---
Adaptive learning rates (RMSprop/Adam component) that normalize per-parameter updates.
---
card_id: JGox7hj7
---
What happens to **bias correction** after many training steps?
---
Correction factor $$\frac{1}{1-\beta^t}$$ approaches 1, so correction has negligible effect after warmup.
---
card_id: 0nhSxlz2
---
How do optimizers **navigate ravines** differently?
---
**SGD**: Zigzags perpendicular to valley
**Momentum**: Smooths zigzags
**RMSprop/Adam**: Adapts per-dimension, navigates directly
