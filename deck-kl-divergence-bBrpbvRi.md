---
card_id: 2xirm7wV
---
What is **information content** (surprisal)?
---
The amount of information conveyed by observing an event, measured as $$I(x) = -\log_2 p(x)$$ bits.
---
card_id: TUvCNeCu
---
What does **information content** measure about an event?
---
How surprising or unexpected an event is based on its probability.
---
card_id: OVg4M6Rw
---
Why is the **negative logarithm** used for information content?
---
- Inverse relationship with probability
- Additive for independent events
- Zero for certain events, infinity for impossible events
---
card_id: xxz63dN2
---
Why do rare events carry more **information** than common events?
---
Because rare events (low probability) are more surprising and thus more informative when they occur.
---
card_id: EiRN86PW
---
What is **entropy** in information theory?
---
The expected information content, measuring the average unpredictability in a probability distribution.
---
card_id: fUkn6nfq
---
What does **entropy** measure in a probability distribution?
---
The average surprise, unpredictability, or minimum bits needed to encode outcomes from the distribution.
---
card_id: PXdTtRPV
---
When does a discrete distribution achieve **maximum entropy**?
---
When all outcomes are equally likely (uniform distribution).
---
card_id: tPR0LlR7
---
How is **entropy** calculated for a discrete distribution?
---
$$H(P) = -\sum_{x} p(x) \log_2 p(x)$$ with $$p(x)$$: probability of outcome $$x$$
---
card_id: wyOr7b2d
---
How does **entropy** differ between a deterministic and a uniform distribution?
---
**Deterministic**: Zero entropy (perfectly predictable). **Uniform**: Maximum entropy $$\log_2(n)$$ for $$n$$ outcomes (maximally unpredictable).
---
card_id: nC6jtJcc
---
What is **cross-entropy** H(P,Q)?
---
The expected surprise when events follow distribution P but we measure surprise using distribution Q.
---
card_id: GKzH8sFe
---
What does **cross-entropy** H(P,Q) measure?
---
The average number of bits needed to encode events from P using an encoding optimized for Q.
---
card_id: nQo3tFVo
---
How is **cross-entropy** H(P,Q) calculated?
---
$$H(P, Q) = -\sum_{x} p(x) \log_2 q(x)$$ with $$p(x)$$: true probability, $$q(x)$$: model probability
---
card_id: cLcJ0YZQ
---
How does **cross-entropy** H(P,Q) relate to **entropy** H(P)?
---
$$H(P,Q) \geq H(P)$$ with equality only when $$Q = P$$. Cross-entropy equals entropy plus KL divergence.
---
card_id: hlFdeOFD
---
Why is **cross-entropy** used as a loss function in classification?
---
It measures how well model predictions Q match true labels P. Lower cross-entropy means better predictions.
---
card_id: fAbn1WDa
---
What is **KL divergence** $$D_{KL}(P \| Q)$$?
---
The extra bits needed when using distribution Q to encode events from distribution P, beyond the optimal encoding.
---
card_id: fLSv9l4w
---
What does **KL divergence** measure between two distributions?
---
How different distribution Q is from distribution P, or the cost of using the wrong distribution.
---
card_id: ls7v2wDf
---
How is **KL divergence** $$D_{KL}(P \| Q)$$ calculated?
---
$$D_{KL}(P \| Q) = \sum_{x} p(x) \log_2 \frac{p(x)}{q(x)}$$ with $$p(x)$$: true probability, $$q(x)$$: model probability
---
card_id: DWsYKq6s
---
How does **KL divergence** relate to **cross-entropy** and **entropy**?
---
$$D_{KL}(P \| Q) = H(P, Q) - H(P)$$. KL divergence is the gap between cross-entropy and entropy.
---
card_id: ANevjsYy
---
What is the value of **$$D_{KL}(P \| Q)$$** when P = Q?
---
Zero. KL divergence is zero if and only if the distributions are identical.
---
card_id: dJwjNvaz
---
Is **KL divergence** symmetric?
---
No. $$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$$ in general. This is why we use the notation $$\|$$ instead of a comma.
---
card_id: eFw5gbhH
---
Is **KL divergence** always non-negative?
---
Yes. $$D_{KL}(P \| Q) \geq 0$$ for all distributions P and Q, with equality only when $$P = Q$$.
---
card_id: 4P6uqdEB
---
What's the difference between **$$D_{KL}(P \| Q)$$** and **$$D_{KL}(Q \| P)$$**?
---
$$D_{KL}(P \| Q)$$: Extra bits when Q encodes events from P. $$D_{KL}(Q \| P)$$: Extra bits when P encodes events from Q.
---
card_id: 5g04Ujgv
---
What factors affect **KL divergence** between two Gaussians?
---
Both mean difference and variance ratio. KL divergence increases with larger mean shifts or variance mismatches.
---
card_id: R6XZOcmi
---
How is **KL divergence** used in Variational Autoencoders (VAEs)?
---
Regularizes the learned latent distribution $$q(z|x)$$ to stay close to the prior $$p(z)$$ via the term $$D_{KL}(q(z|x) \| p(z))$$.
---
card_id: a9m4boPv
---
How is **KL divergence** used in RL policy optimization (TRPO/PPO)?
---
Constrains policy updates to prevent large changes: $$D_{KL}(\pi_{old} \| \pi_{new}) < \delta$$ for stable learning.
---
card_id: vOtwrChu
---
How is **KL divergence** used in knowledge distillation?
---
Minimizes $$D_{KL}(P_{teacher} \| P_{student})$$ to make the student model's output distribution match the teacher's.
---
card_id: Nza294Pv
---
Why does minimizing **cross-entropy** minimize **KL divergence** in classification?
---
Because $$H(P,Q) = H(P) + D_{KL}(P \| Q)$$, and $$H(P)$$ is constant for fixed labels. Minimizing $$H(P,Q)$$ minimizes $$D_{KL}(P \| Q)$$.
---
card_id: 1bU7fhD3
---
What does **KL divergence** represent in terms of compression?
---
The extra storage cost (in bits) when using a suboptimal compression scheme Q instead of the optimal scheme P.
---
card_id: 3s0CJLx8
---
What are the four key properties of **KL divergence**?
---
- Non-negative: $$D_{KL}(P \| Q) \geq 0$$
- Zero iff $$P = Q$$
- Asymmetric: $$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$$
- Not a distance metric (fails triangle inequality)
---
card_id: xjLECWXC
---
Model predicts [0.9, 0.05, 0.05] for true label [1, 0, 0]. Is **cross-entropy** low or high?
---
Low. The model is confident and correct, so the cross-entropy loss is low (approximately 0.105).
---
card_id: V8Avm5dM
---
Model predicts [0.05, 0.9, 0.05] for true label [1, 0, 0]. Is **cross-entropy** low or high?
---
High. The model is confident but wrong, resulting in high cross-entropy loss (approximately 2.996).
---
card_id: qlCMWU4K
---
Two distributions have the same mean and variance. Is their **KL divergence** zero?
---
Not necessarily. KL divergence compares entire distributions, not just moments. Different shapes can have the same mean and variance.
---
card_id: tQd8F8Hj
---
What is **differential entropy** for a Gaussian $$N(\mu, \sigma^2)$$?
---
$$H = \frac{1}{2} \log_2(2\pi e \sigma^2)$$ bits. Wider Gaussians (larger $$\sigma$$) have higher entropy.
---
card_id: 5EOzUBPI
---
How does mean shift affect **Gaussian KL divergence**?
---
KL divergence grows quadratically with the distance between means. Larger mean shifts cause proportionally larger divergence.
---
card_id: sYPfW8Kl
---
A loaded die always shows 6. What is its **entropy**?
---
Zero bits. It's perfectly predictable with no uncertainty.
---
card_id: klARsh33
---
A fair 6-sided die is rolled. What is its **entropy**?
---
$$\log_2(6) \approx 2.585$$ bits. This is maximum entropy for 6 equally likely outcomes.
---
card_id: iBQFL7gR
---
Why is **cross-entropy** always greater than or equal to **entropy**?
---
Because using the wrong distribution Q to encode events from P always costs at least as many bits as using P itself.
---
card_id: OJXcS0uK
---
What does it mean when **$$D_{KL}(P \| Q)$$** is large?
---
Distribution Q is very different from P, or Q is a poor model of P, resulting in many wasted bits.
---
card_id: pHq0jqEe
---
In VAE training, what happens if the **KL divergence** term is too high?
---
The encoder deviates too far from the prior, potentially making generation from random latent samples fail.
---
card_id: qTYZ7hw9
---
In RL, what happens if **$$D_{KL}(\pi_{old} \| \pi_{new})$$** is too large?
---
The policy changes too drastically, potentially causing catastrophic forgetting of good behaviors and unstable learning.
---
card_id: Tu2nFTve
---
What is **mode-seeking behavior** in KL divergence?
---
When using $$D_{KL}(P \| Q)$$, Q is penalized for missing mass where P has mass, causing Q to focus on modes of P.
---
card_id: CXH10hVp
---
What is **mean-seeking behavior** in KL divergence?
---
When using $$D_{KL}(Q \| P)$$, Q is penalized for having mass where P doesn't, causing Q to cover all of P's support.
---
card_id: 5zioMJqg
---
When should you use **$$D_{KL}(P \| Q)$$** vs **$$D_{KL}(Q \| P)$$**?
---
**$$D_{KL}(P \| Q)$$**: Mode-seeking, when you want Q to focus on high-probability regions of P. **$$D_{KL}(Q \| P)$$**: Mean-seeking, when you want Q to cover all regions where P has mass.
---
card_id: ZytQb6XY
---
What is the difference between **bits** and **nats** in information theory?
---
**Bits**: Use $$\log_2$$, measure information in binary digits. **Nats**: Use $$\ln$$ (natural log), measure information in natural units.
---
card_id: 4tpihOAE
---
What is the **0 log 0 convention** in entropy calculations?
---
By convention, $$0 \cdot \log(0) = 0$$, since events with zero probability contribute nothing to entropy.
---
card_id: qwRl3uQA
---
What is the **KL divergence formula for two Gaussians** $$N(\mu_1, \sigma_1^2)$$ and $$N(\mu_2, \sigma_2^2)$$?
---
$$D_{KL}(P \| Q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$ (in nats)
---
card_id: UnJlKI07
---
Why does the **uniform distribution** have maximum entropy?
---
Because entropy is maximized when all outcomes are equally likely, representing maximum uncertainty and unpredictability.
---
card_id: 7mkpu6IW
---
How does **cross-entropy** relate to **log-likelihood**?
---
Minimizing cross-entropy is equivalent to maximizing log-likelihood. $$H(P,Q) = -\mathbb{E}_P[\log q(x)]$$ is the negative log-likelihood.
---
card_id: 6LqRXApG
---
What is the **ELBO** in Variational Autoencoders?
---
Evidence Lower BOund: $$\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$. Reconstruction term minus KL regularization.
---
card_id: tf2HaBrx
---
What does the **KL term** do in the VAE ELBO?
---
Regularizes the encoder distribution $$q(z|x)$$ to stay close to the prior $$p(z)$$, preventing overfitting and ensuring smooth latent space.
---
card_id: TGyN7L6Z
---
What is **$$\beta$$-VAE**?
---
A VAE variant that weights the KL term by $$\beta$$: $$\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \beta \cdot D_{KL}(q(z|x) \| p(z))$$. Controls reconstruction vs regularization trade-off.
---
card_id: AuCBKRc4
---
Why does **knowledge distillation** capture more than just the correct class?
---
Because it matches the entire probability distribution from the teacher, capturing uncertainty and relationships between classes, not just the argmax.
---
card_id: cU5RRjvn
---
Coin lands on edge ($$p = 0.0001$$). How much **information** in bits?
---
$$-\log_2(0.0001) \approx 13.29$$ bits. Like flipping 13 coins and getting all heads.
---
card_id: 48MYQXVi
---
Winning Powerball lottery ($$p \approx 3.42 \times 10^{-9}$$). How much **information** in bits?
---
$$-\log_2(3.42 \times 10^{-9}) \approx 28.12$$ bits. Like flipping 28 coins and getting all heads.
---
card_id: PHJ48oeL
---
Why is **KL divergence** called a "divergence" and not a "distance"?
---
Because it's asymmetric ($$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$$) and doesn't satisfy the triangle inequality, so it's not a true metric.
---
card_id: hkrs1YXj
---
What does the notation **$$\|$$** in $$D_{KL}(P \| Q)$$ emphasize?
---
The asymmetry of KL divergence. It's not commutative like a distance function, so we use $$\|$$ instead of a comma.
---
card_id: vlyDTxgf
---
When **minimizing cross-entropy** during training, what are you implicitly doing?
---
Minimizing KL divergence between the true distribution P and the model distribution Q, since $$H(P)$$ is constant.
---
card_id: ReJqJyEI
---
VAE: High reconstruction quality but poor generation. Which term in ELBO is likely the problem?
---
The KL term is likely too low (encoder deviates from prior), causing the latent space to be poorly structured for generation.
---
card_id: v0zmQNMI
---
Why does **wider variance** in a Gaussian increase its entropy?
---
Because probability mass is more spread out, making outcomes less predictable and increasing average surprise.
---
card_id: 2BdsHtAJ
---
What happens to **entropy** as a distribution becomes more concentrated?
---
Entropy decreases. More concentrated distributions are more predictable with less average surprise.
---
card_id: d7z5sL7k
---
In practice, how do you choose between **forward** and **reverse KL** for variational inference?
---
**Forward** $$D_{KL}(P \| Q)$$: When P is known and you want Q to cover modes. **Reverse** $$D_{KL}(Q \| P)$$: When P is intractable and you optimize Q (standard in VAEs).
---
card_id: Qw8SIwrX
---
Why can't we compute **$$D_{KL}(P \| Q)$$** directly in VAEs?
---
Because the true posterior $$P(z|x)$$ is intractable. We use reverse KL $$D_{KL}(Q \| P)$$ which only requires evaluating Q and the prior P.
---
card_id: QLjS4pXE
---
Rewrite **entropy** $$H(P) = -\sum_{x} p(x) \log p(x)$$ without the negative sign.
---
$$H(P) = \sum_{x} p(x) \log \frac{1}{p(x)}$$
---
card_id: pkEZh2TH
---
Rewrite **entropy** $$H(P) = \sum_{x} p(x) \log \frac{1}{p(x)}$$ using a negative sign.
---
$$H(P) = -\sum_{x} p(x) \log p(x)$$
---
card_id: Lpi55Oc1
---
Express **entropy** $$H(P) = -\sum_{x} p(x) \log p(x)$$ using expectation notation.
---
$$H(P) = \mathbb{E}_{x \sim P}[-\log p(x)]$$ or $$\mathbb{E}[\log \frac{1}{p(x)}]$$
---
card_id: AuA49JzG
---
Expand **entropy** $$H(P) = \mathbb{E}[-\log p(x)]$$ as a summation formula.
---
$$H(P) = -\sum_{x} p(x) \log p(x)$$ or $$\sum_{x} p(x) \log \frac{1}{p(x)}$$
