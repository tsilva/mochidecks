---
card_id: yRFFIvgE
---
What are **Siamese networks**?
---
Neural network architectures with shared weights that learn to determine whether two inputs are similar or dissimilar.
---
card_id: xcIyUCEY
---
What problem do **Siamese networks** solve?
---
Learning similarity between inputs when labeled data is scarce, enabling recognition from few examples.
---
card_id: RDhbQILL
---
When should you use **Siamese networks**?
---
- One-shot or few-shot learning tasks
- Face or signature verification
- Similarity search in large databases
- When labeled training data is limited
---
card_id: YNi1UKOG
---
How do **Siamese networks** process two inputs?
---
1. Pass both inputs through the same network with shared weights
2. Generate embeddings for each input
3. Compute distance between embeddings
4. Use distance to determine similarity
---
card_id: xYaDlkwR
---
How do **Siamese networks** differ from traditional classification networks?
---
**Siamese**: Learn similarity/dissimilarity between pairs, output distance / **Traditional**: Predict class labels, output probabilities
---
card_id: zanwlTS8
---
What are **shared weights** in Siamese networks?
---
Identical network parameters used to process both inputs, ensuring consistent representations across the twin networks.
---
card_id: 7Mb977gz
---
What is the purpose of **shared weights** in Siamese networks?
---
Ensure differences in embeddings reflect differences in inputs rather than differences in processing.
---
card_id: y7DlgzDK
---
How are **shared weights** implemented in a Siamese network?
---
Both inputs pass through the same embedding network (same layers and parameters) in separate forward passes.
---
card_id: 2GyX8aX8
---
Scenario: You build a Siamese network with two separate networks instead of shared weights. What problem occurs?
---
Embeddings may differ due to network variations rather than input differences, preventing meaningful similarity comparisons.
---
card_id: SyzUviwC
---
What is **contrastive loss**?
---
Loss function that pulls similar pairs together by minimizing distance and pushes dissimilar pairs apart up to a margin.
---
card_id: F5qZ0AG9
---
What does **contrastive loss** optimize for?
---
Creating an embedding space where similar items are close together and dissimilar items are separated by at least a margin.
---
card_id: PzxpLQjt
---
What is the **contrastive loss** formula?
---
$$\mathcal{L} = \frac{1}{2N} \sum_{i=1}^N \left[ y_i \cdot d_i^2 + (1-y_i) \cdot \max(0, m - d_i)^2 \right]$$

with $y_i$: 1 for similar, 0 for dissimilar; $d_i$: distance; $m$: margin
---
card_id: qcvOfQHG
---
How does **contrastive loss** handle similar pairs?
---
Loss equals $d^2$ (squared distance), creating strong pressure to minimize distance toward zero.
---
card_id: xqJof30f
---
How does **contrastive loss** handle dissimilar pairs?
---
Loss equals $\max(0, m - d)^2$, penalizing distances below margin but zero when distance exceeds margin.
---
card_id: NOffpqhW
---
What is the role of the **margin parameter** in contrastive loss?
---
Defines minimum distance for dissimilar pairs; prevents wasted effort pushing already-separated pairs further apart.
---
card_id: MKcKM3o7
---
Scenario: Your Siamese network produces distance 0.3 for similar pairs and 1.8 for dissimilar pairs with margin=2.0. What happens to loss?
---
Similar pairs contribute loss; dissimilar pairs also contribute loss since 1.8 < 2.0 margin, pushing them further apart.
---
card_id: 4XdvLUZ4
---
What is **Euclidean distance**?
---
Straight-line distance between two points, computed as $d(x_1, x_2) = \sqrt{\sum_{i=1}^n (x_{1,i} - x_{2,i})^2}$.
---
card_id: agCbZkCf
---
What is **cosine similarity**?
---
Measure of angle between two vectors, computed as $\frac{x_1 \cdot x_2}{\|x_1\| \|x_2\|}$, ranging from -1 (opposite) to 1 (identical).
---
card_id: 6VYUKOsc
---
How does **Euclidean distance** differ from **cosine similarity**?
---
**Euclidean**: Sensitive to both direction and magnitude / **Cosine**: Only sensitive to direction, ignores magnitude
---
card_id: Qglh431l
---
What is an **embedding space** in Siamese networks?
---
Learned vector space where distance between embeddings corresponds to semantic similarity of inputs.
---
card_id: drHCAwES
---
What property does the **embedding space** have after training?
---
Similar items cluster together with small distances; dissimilar items are separated with large distances.
---
card_id: IQghSMkc
---
How do **Siamese networks** create embeddings?
---
Pass inputs through convolutional or fully connected layers that map to fixed-dimensional vectors in embedding space.
---
card_id: 1xll2YK6
---
What is **one-shot learning**?
---
Recognizing new classes from just a single example per class.
---
card_id: XB9zVgnE
---
What enables **one-shot learning** in Siamese networks?
---
Learned similarity function allows classification by comparing query to single support examples via nearest-neighbor search.
---
card_id: EMzXUKul
---
How does **one-shot classification** work with Siamese networks?
---
1. Create support set with one example per class
2. Compute distances from query to all support examples
3. Predict class of closest support example
---
card_id: 4D0Jknzl
---
When is **one-shot learning** useful?
---
- Recognizing new classes with minimal labeled data
- Face verification with few examples per person
- Scenarios where collecting many examples is expensive
- Rapidly adapting to new categories
---
card_id: 1l3BPVQI
---
Scenario: You have 1 example each of 10 new digits. How would a Siamese network classify a query digit?
---
Compute embedding distances from query to all 10 support examples; predict class of nearest support example.
---
card_id: bRVm89g9
---
What is **triplet loss**?
---
Loss function using anchor-positive-negative triplets that encourages positive closer to anchor than negative by a margin.
---
card_id: aFy3nRRz
---
What is the **triplet loss** formula?
---
$$\mathcal{L} = \max(0, d(a, p) - d(a, n) + m)$$

with $a$: anchor, $p$: positive, $n$: negative, $m$: margin
---
card_id: 66qUv5gf
---
How does **triplet loss** differ from **contrastive loss**?
---
**Triplet**: Uses anchor-positive-negative triplets, directly optimizes relative distances / **Contrastive**: Uses pairs, optimizes absolute distances
---
card_id: brXUxibD
---
What are **positive pairs** in Siamese network training?
---
Pairs of inputs from the same class with label 1, used to train the network to minimize their distance.
---
card_id: wcKYwf0U
---
What are **negative pairs** in Siamese network training?
---
Pairs of inputs from different classes with label 0, used to train the network to maximize their distance.
---
card_id: y3WWA4JS
---
What are common applications of **Siamese networks**?
---
Face verification, signature verification, similarity search, one-shot learning, metric learning, product matching, image retrieval.
