---
card_id: oRhG8R26
---
What types of functions can a single linear neuron represent?
---
Linear functions such as linear regression, addition, and subtraction.
---
card_id: 5deHzVGh
---
What activation must the output layer use for exact arithmetic?
---
A linear activation.
---
card_id: mi1v8ZFr
---
Why do multi-operation arithmetic networks require a hidden layer?
---
To model nonlinear functions like multiplication and division.
---
card_id: mBJe8zqm
---
Are multiplication and division linearly separable?
---
No, they require nonlinear representations.
---
card_id: 2ieOu1LH
---
Is the parity (odd/even) problem linearly separable?
---
No, parity is not linearly separable.
---
card_id: NEaJf1oZ
---
Does a single perceptron work for odd/even classification?
---
No, at least one hidden layer is needed.
---
card_id: QXz9My5C
---
Why is sparse categorical cross-entropy memory efficient?
---
It uses integer labels instead of one-hot encodings.
---
card_id: wkB1hrw6
---
Binary cross-entropy is a special case of which loss?
---
Categorical cross-entropy.
---
card_id: nWN51TVy
---
Why does binary cross-entropy perform well for classification?
---
It heavily penalizes confident incorrect predictions.
---
card_id: tQQARqQN
---
What is sigmoid mathematically equal to?
---
The logistic function.
---
card_id: 6zSDBqW4
---
A single sigmoid neuron is equivalent to what traditional model?
---
Logistic regression.
---
card_id: RAfFb5Sf
---
What advantage do large batch sizes provide?
---
Higher compute efficiency.
---
card_id: KBW1939G
---
What advantage do small batch sizes provide?
---
Better generalization due to gradient noise.
---
card_id: abVaqJZv
---
What batch size is a commonly recommended starting point?
---
32.
---
card_id: SrUMFYaT
---
L2 regularization is equivalent to what technique?
---
Weight decay.
---
card_id: yyAgOcjz
---
What does dropout accomplish in training?
---
Robustness through an ensemble-like effect.
---
card_id: hzzXg7KH
---
Where is dropout most effective in neural networks?
---
In fully connected layers or attention blocks.
---
card_id: 5TPDDdkb
---
What is a drawback of max pooling?
---
It discards spatial information.
---
card_id: 1wS1nZDd
---
What commonly replaces pooling in modern CNNs?
---
Strided convolutions.
---
card_id: u8SgiEpx
---
Q-learning is what category of RL method?
---
A value-based method.
---
card_id: mXh5x8Bh
---
How should time-series validation splits be structured?
---
Train on past, validate on future data.
---
card_id: oTQyeM6o
---
How should validation be structured with grouped entities like boats or patients?
---
Use unseen groups in validation.
---
card_id: ZOUd1lyy
---
What is the best metric for reliable early stopping?
---
Validation loss.
---
card_id: GIKJejMr
---
Why might validation accuracy also be used for early stopping?
---
It may stop earlier, useful for ensembles.
---
card_id: yICuwYi4
---
Why increase the number of filters in deeper CNN layers?
---
To capture more complex features.
---
card_id: YIcbuiP3
---
What does varying kernel size or stride achieve in CNNs?
---
Multiscale structural representation.
---
card_id: cRIeq0Rj
---
Why should inputs be scaled/normalized for neural networks?
---
It stabilizes and improves training, especially for ReLU.
---
card_id: oWHKH5Nm
---
What optimizer is a common starting point with neural nets?
---
Adam with learning rate around 1e-3.
---
card_id: zqhZuzBK
---
How should you respond to underfitting?
---
Increase model capacity or reduce regularization.
---
card_id: vyGKreyd
---
How should you respond to overfitting?
---
Increase regularization or reduce model capacity.
---
card_id: 9QIkipa1
---
When should learning rate scheduling and early stopping be applied?
---
During fine-tuning after initial convergence.
---
card_id: tiRgOIb7
---
What is the purpose of a final held-out test set?
---
To estimate generalization performance.
---
card_id: lbJ9OVBr
---
What input distribution helps ReLU networks learn effectively?
---
Zero-centered inputs with mean ≈ 0 and std ≈ 1.
---
card_id: QSCL98Es
---
What happens if ReLU networks only see positive activations?
---
The network becomes locally linear and may fail to learn nonlinear behaviors.
