---
card_id: 7QNjyIIj
---
What is a **learning rate schedule**?
---
Strategy that adjusts learning rate during training to balance fast progress and precise convergence.
---
card_id: K7JsH7IC
---
What problem do **learning rate schedules** solve?
---
Constant learning rates either converge slowly or oscillate without achieving fine convergence.
---
card_id: Y4EsY2uR
---
How do **learning rate schedules** improve training?
---
Start high for fast exploration, decrease gradually for fine-grained optimization.
---
card_id: M29fQhZM
---
What are the main types of **learning rate schedules**?
---
Step decay, exponential decay, cosine annealing, warmup.
---
card_id: 54Qf5qoW
---
What is **step decay**?
---
Reduces learning rate by a multiplicative factor at fixed epoch intervals.
---
card_id: jozAXTeh
---
What is the formula for **step decay**?
---
$$\text{lr} = \text{lr}_0 \times \gamma^{\lfloor \text{epoch} / \text{step\_size} \rfloor}$$

$\text{lr}_0$: initial LR, $\gamma$: decay factor, $\text{step\_size}$: epochs between reductions
---
card_id: aht5TwMy
---
When should you use **step decay**?
---
Classic CNNs, older vision models, or replicating papers that use this schedule.
---
card_id: T8LtZ5xR
---
What is **exponential decay**?
---
Reduces learning rate smoothly by a constant factor each epoch.
---
card_id: 44iFlYEI
---
What is the formula for **exponential decay**?
---
$$\text{lr} = \text{lr}_0 \times \gamma^{\text{epoch}}$$

$\text{lr}_0$: initial LR, $\gamma$: decay rate (typically 0.95-0.99)
---
card_id: JNQ5brSq
---
How does **exponential decay** differ from **step decay**?
---
**Exponential**: Smooth continuous reduction / **Step**: Abrupt reductions at fixed intervals
---
card_id: Va9cof5k
---
What is **cosine annealing**?
---
Schedule following a cosine curve from high to low learning rate.
---
card_id: YwZdVdVh
---
What does **cosine annealing** optimize for?
---
Maintains high LR longer for exploration, then drops aggressively for fine-tuning.
---
card_id: r7yOIwxL
---
What is the formula for **cosine annealing**?
---
$$\text{lr} = \text{lr}_{\min} + \frac{1}{2}(\text{lr}_{\max} - \text{lr}_{\min}) \left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\text{max}}} \pi\right)\right)$$

$T_{\text{cur}}$: current epoch, $T_{\text{max}}$: total epochs
---
card_id: 1ZXHMwDH
---
When should you use **cosine annealing**?
---
Modern CNNs, ResNets, Vision Transformers, or as default for most deep learning tasks.
---
card_id: eiAzC9Go
---
What are advantages of **cosine annealing**?
---
Smooth decay, minimal hyperparameter tuning, state-of-the-art results across domains.
---
card_id: mybyRvyn
---
How does **cosine annealing** differ from **exponential decay**?
---
**Cosine**: Maintains higher LR longer, aggressive late drop / **Exponential**: Steady conservative decay
---
card_id: EGEVyMWT
---
What is **warmup** in learning rate schedules?
---
Gradually increasing learning rate from near-zero to target over initial epochs.
---
card_id: xzmKsfEc
---
What problem does **warmup** solve?
---
Prevents divergence from large learning rates applied to randomly initialized unstable weights.
---
card_id: NFeeAKNH
---
How does **warmup** stabilize early training?
---
Allows stable gradient patterns to form before applying full learning rate.
---
card_id: lAr9Wv5L
---
When should you use **warmup** for Transformers?
---
Always; Transformers have unstable early gradients requiring gradual LR ramp-up.
---
card_id: wT4E4eaB
---
When should you use **warmup** for large batch training?
---
Batch sizes above 256, where initial gradient estimates are noisier.
---
card_id: CoakV5zr
---
What is typical **warmup** duration?
---
5-10% of total training epochs.
---
card_id: OfHf71IW
---
What is **warmup + cosine annealing**?
---
Linear LR increase during warmup followed by cosine decay to minimum.
---
card_id: F247XMRD
---
When should you use **warmup + cosine annealing**?
---
Training Transformers, large models, or large batch sizes requiring early stability.
---
card_id: fZj53sOZ
---
What is **cosine annealing with warm restarts** (SGDR)?
---
Periodically restarts learning rate to high values following cosine curves.
---
card_id: DycvRczo
---
When should you use **SGDR**?
---
Experimental research, snapshot ensembles, or escaping poor local minima.
---
card_id: BgxcYIHx
---
Model trains with constant LR=0.01 but loss plateaus and oscillates. Solution?
---
Apply learning rate schedule (cosine or step decay) to enable finer optimization.
---
card_id: qiaUktc5
---
Training Transformer from scratch with batch size 256. Which LR schedule?
---
Warmup + cosine annealing.
---
card_id: KCctdXWF
---
Training ResNet-50 for 200 epochs. Default LR schedule?
---
Cosine annealing with T_max=200, eta_min=0.
---
card_id: mVnAWB3E
---
Large batch training (size 512) shows loss spikes in first 10 epochs. Solution?
---
Add warmup phase for 5-10% of total epochs.
---
card_id: I2HMt1Ov
---
Model using step decay shows loss spike after LR reduction at epoch 50. Why?
---
Abrupt LR reduction causes temporary instability as optimizer adjusts.
---
card_id: E3PsA4jf
---
Training CNN but unsure of total epochs. Which schedule to avoid?
---
Cosine annealing (requires T_max upfront); use exponential or step decay.
---
card_id: h42smCal
---
Loss decreases steadily with constant LR=0.001 but never fully converges. Issue?
---
Learning rate too small for efficient exploration; start higher with decay.
---
card_id: CqAQlNeZ
---
Training loss explodes in first epoch with LR=0.01. Solution?
---
Add warmup to gradually increase from near-zero, or reduce initial LR.
---
card_id: 827akBE7
---
Following 2015 vision paper for ResNet. Which LR schedule likely used?
---
Step decay (reduce by 0.1 every 30-50 epochs).
---
card_id: kYa1HvEz
---
Small model (50K params), 100 epochs, batch size 32. Need warmup?
---
No; warmup less critical for small models and batch sizes.
---
card_id: 8lJykM3d
---
Model with constant LR shows good train loss but poor validation loss. Add schedule?
---
No; this is overfitting, not an LR schedule problem. Add regularization instead.
---
card_id: ZUtOal8a
---
Training stalls at epoch 150 with exponential decay (gamma=0.97). What's happening?
---
Learning rate decayed too low; increase gamma or use cosine for slower early decay.
