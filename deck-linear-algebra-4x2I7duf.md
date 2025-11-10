---
card_id: RxLVZ3ft
---
What is **linear algebra**?
---
The study of vectors, matrices, and linear transformations.
---
card_id: l1ZyDLOb
---
How is data structured in **machine learning**?
---
As vectors (samples/features) and matrices (datasets/transformations).
---
card_id: KYP2Rcdb
---
What is a **vector** in linear algebra?
---
An ordered list of numbers that can represent a point in space or direction with magnitude.
---
card_id: null
---
What operation implements a **neural network layer**?
---
Matrix multiplication: weights transform input vectors to output vectors.
---
card_id: CA7nlbE8
---
How is data represented as **vectors** in ML?
---
- Images: pixel values flattened to vector
- Text: word embeddings (e.g., 300-dimensional)
- Features: measurements as components
---
card_id: l1o1OaK1
---
What is **vector addition** geometrically?
---
Placing the tail of the second vector at the head of the first, forming a parallelogram.
---
card_id: JSS0T54N
---
What does **scalar multiplication** do to a vector?
---
Scales its length; negative scalars also reverse direction.
---
card_id: fxId1pWl
---
What is the **dot product** algebraically?
---
$$\mathbf{a} \cdot \mathbf{b} = \sum_{i} a_i b_i$$
where $a_i$ and $b_i$ are corresponding elements.
---
card_id: gsa5W107
---
What is the **dot product** geometrically?
---
$$\mathbf{a} \cdot \mathbf{b} = ||\mathbf{a}|| \, ||\mathbf{b}|| \cos(\theta)$$
where $\theta$ is the angle between vectors.
---
card_id: NY8CAtcF
---
What does the **dot product** measure?
---
Directional similarity: positive (same direction), zero (perpendicular), negative (opposite).
---
card_id: xpBAo6Hr
---
When are two vectors **orthogonal**?
---
When their dot product equals zero, meaning they are perpendicular.
---
card_id: mmRIlpED
---
What is the **projection** of vector a onto vector b?
---
$$\text{proj}_{\mathbf{b}}(\mathbf{a}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{b}||^2} \mathbf{b}$$
The component of $\mathbf{a}$ along $\mathbf{b}$ direction.
---
card_id: f4pNM7JW
---
What is the **L2 norm** (Euclidean norm)?
---
$$||\mathbf{v}||_2 = \sqrt{\sum_{i} v_i^2}$$
The straight-line distance from origin to point.
---
card_id: rPgBRjLl
---
What is the **L1 norm** (Manhattan norm)?
---
$$||\mathbf{v}||_1 = \sum_{i} |v_i|$$
The sum of absolute values of components.
---
card_id: ySJpR0t3
---
What is the **L∞ norm** (max norm)?
---
$$||\mathbf{v}||_\infty = \max_i |v_i|$$
The largest absolute value among all components.
---
card_id: DZIYuTvD
---
What shape is the **L2 norm** unit circle?
---
Perfect circle (all points equidistant from origin using Euclidean distance).
---
card_id: z0gz16hT
---
What shape is the **L1 norm** unit circle?
---
Diamond or rotated square (all points with sum of absolute values = 1).
---
card_id: 8l761Hmh
---
What shape is the **L∞ norm** unit circle?
---
Axis-aligned square (all points where max component = 1).
---
card_id: SBuvJyN9
---
Why does **L1 regularization** produce sparse models?
---
The diamond-shaped L1 unit ball has sharp corners at the axes, creating pressure toward zero weights.
---
card_id: PiypMEWT
---
Why does **L2 regularization** produce dense models?
---
The circular L2 unit ball is smooth and uniform, shrinking all weights proportionally without preference for zeros.
---
card_id: xBs6PfE7
---
What is a **unit vector**?
---
A vector with norm equal to 1.
---
card_id: Nfi6lpdO
---
How do you **normalize** a vector?
---
$$\mathbf{v}_{\text{normalized}} = \frac{\mathbf{v}}{||\mathbf{v}||}$$
Divide by its norm.
---
card_id: UVKoefOH
---
Why normalize **word embeddings** in ML?
---
To measure similarity using dot product independent of magnitude, focusing only on direction.
---
card_id: PpzqMBPO
---
What does the **king - man + woman ≈ queen** example demonstrate?
---
Word embeddings capture semantic relationships where vector arithmetic reflects meaning relationships.
---
card_id: YEIMAZLY
---
What is a **matrix** in linear algebra?
---
A rectangular array of numbers organized in rows and columns.
---
card_id: ciP3fLTV
---
What are three ways to interpret a **matrix**?
---
1. Collection of vectors (rows or columns)
2. Lookup table (row/column indices)
3. Linear transformation (function transforming vectors)
---
card_id: 5thV5ejU
---
In **matrix-vector multiplication**, if A is (m×n) and v is (n×1), what is the result shape?
---
(m×1) - the result is a column vector with m elements.
---
card_id: FVJLuTFr
---
What does each element represent in **matrix-vector multiplication**?
---
The dot product of a matrix row with the input vector.
---
card_id: EcN10zlL
---
What do the **columns of a matrix** represent geometrically?
---
Where the basis vectors go after the transformation; they define the transformed coordinate system.
---
card_id: WN1TAQ2O
---
In **matrix-matrix multiplication**, if A is (m×n) and B is (n×p), what is the result shape?
---
(m×p) - rows from A, columns from B.
---
card_id: fJqyzjGb
---
What does **matrix-matrix multiplication** represent?
---
Composition of transformations: A @ B means "first apply B, then apply A" (right-to-left order).
---
card_id: aC3e3ZCS
---
Is **matrix multiplication commutative**?
---
No - in general A @ B ≠ B @ A. Order matters for transformations.
---
card_id: yoWYOB6w
---
What is the **transpose** of a matrix?
---
Flipping the matrix over its diagonal so rows become columns and columns become rows. Notation: $\mathbf{A}^T$
---
card_id: lEeFt8NC
---
What is the transpose property for **matrix products**?
---
$$(\mathbf{AB})^T = \mathbf{B}^T \mathbf{A}^T$$
Transpose reverses the order of multiplication.
---
card_id: h5xutEFH
---
What is the **identity matrix**?
---
A square matrix with 1s on the diagonal and 0s elsewhere; the multiplicative identity ($\mathbf{IA} = \mathbf{AI} = \mathbf{A}$).
---
card_id: jEwtWYfv
---
What is the **matrix inverse**?
---
Matrix $\mathbf{A}^{-1}$ that "undoes" transformation A, satisfying $\mathbf{A} \mathbf{A}^{-1} = \mathbf{A}^{-1} \mathbf{A} = \mathbf{I}$.
---
card_id: lroOZ22B
---
What conditions must a matrix satisfy to have an **inverse**?
---
1. Must be square (same rows and columns)
2. Must have full rank (linearly independent rows/columns)
---
card_id: 7ynvdQGY
---
What is the **rank** of a matrix?
---
The number of linearly independent rows or columns; the number of truly independent directions.
---
card_id: Dxt5zino
---
What is a **singular matrix**?
---
A matrix with rank less than its dimensions, meaning it squashes space into fewer dimensions and has no inverse (determinant = 0).
---
card_id: 09jSFy6b
---
How is a dataset represented as a **matrix** in ML?
---
Rows represent samples (data points), columns represent features (dimensions).
---
card_id: 6bbjk7xR
---
What does a **neural network layer** compute?
---
$$\mathbf{Y} = \mathbf{XW} + \mathbf{b}$$
Matrix multiplication of input by weights plus bias broadcast across batch.
---
card_id: Z9TJe9Us
---
What is a **linear transformation**?
---
A function that preserves vector addition and scalar multiplication: $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$ and $T(c\mathbf{v}) = cT(\mathbf{v})$.
---
card_id: kz49mm7r
---
What do **linear transformations** preserve?
---
Lines remain lines, origin stays fixed, parallel lines stay parallel.
---
card_id: CnoVXkGr
---
What can **linear transformations** change?
---
Distances, angles, areas, and volumes can all change under linear transformations.
---
card_id: MiAaE4eF
---
What is the matrix form for a **scaling transformation**?
---
Diagonal matrix: $\begin{bmatrix} a & 0 \\ 0 & b \end{bmatrix}$
Stretches x-axis by a, y-axis by b.
---
card_id: miGT3TsX
---
What is the matrix for **rotation** by angle θ?
---
$$\mathbf{R}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$
Counterclockwise rotation.
---
card_id: w7wud9Oj
---
Why does the **rotation matrix** have that specific form?
---
Columns show where basis vectors go: $[1,0] \to [\cos\theta, \sin\theta]$ and $[0,1] \to [-\sin\theta, \cos\theta]$ (perpendicular on unit circle).
---
card_id: QrHg8rb7
---
What does a **shearing transformation** do?
---
Slants space while keeping one axis fixed, like a stack of cards sliding.
---
card_id: 4sPEOypK
---
Why can't you just stack linear layers in a **neural network**?
---
Two linear transformations compose to a single linear transformation ($\mathbf{W}_2 \mathbf{W}_1$ is one matrix), so activation functions are essential for non-linearity.
---
card_id: DpmSyanr
---
What is an **eigenvector**?
---
A special vector that only gets scaled (not rotated) when a matrix is applied: $\mathbf{Av} = \lambda \mathbf{v}$.
---
card_id: B5YjJKOe
---
What is an **eigenvalue**?
---
The scaling factor $\lambda$ by which an eigenvector is multiplied: $\mathbf{Av} = \lambda \mathbf{v}$.
---
card_id: aJbpNAUn
---
What do **eigenvectors** reveal about a transformation?
---
The natural directions that only get scaled, not rotated.
---
card_id: null
---
What are three applications of **eigenvectors** in ML?
---
PCA (dimensionality reduction), PageRank (web ranking), stability analysis (dynamics).
---
card_id: szw7l28k
---
For a **diagonal matrix**, what are the eigenvectors?
---
The standard basis vectors (e.g., $[1,0]$, $[0,1]$ in 2D).
---
card_id: Wcr3ulM6
---
For a **diagonal matrix**, what are the eigenvalues?
---
The diagonal elements themselves.
---
card_id: fWe5we5B
---
What is the **power iteration method**?
---
Repeatedly apply matrix and normalize to find the dominant eigenvector.
---
card_id: gr42Q1p7
---
Why does **power iteration** work?
---
Each application stretches the largest eigenvalue direction more, making it dominate over iterations.
---
card_id: Rxt7H3SR
---
What is the **Rayleigh quotient** for computing eigenvalues?
---
$$\lambda = \mathbf{v}^T \mathbf{A} \mathbf{v}$$
Given eigenvector v, computes corresponding eigenvalue.
---
card_id: TxUAEILC
---
What does **PCA** (Principal Component Analysis) do?
---
Finds directions of maximum variance in data by computing eigenvectors of the covariance matrix.
---
card_id: 2KKvfNB6
---
What does the **covariance matrix** capture?
---
How variables vary together: diagonal has variances, off-diagonal has covariances between features.
---
card_id: ygF8xn74
---
What are the steps in the **PCA algorithm**?
---
1. Center data (subtract mean)
2. Compute covariance matrix
3. Find eigenvectors and eigenvalues
4. Project onto top eigenvectors to reduce dimensions
---
card_id: FGJgULdw
---
What does the **first principal component** (PC1) represent?
---
The direction of maximum variance in the data (eigenvector with largest eigenvalue).
---
card_id: ICzMkHSV
---
What is the **span** of a set of vectors?
---
All possible linear combinations of those vectors; the "space" they can reach.
---
card_id: uMgYamnB
---
What does **linear independence** mean?
---
No vector can be written as a combination of the others; each adds a new direction.
---
card_id: UDOlRQtk
---
What is a **basis** for a vector space?
---
A set of linearly independent vectors that span the entire space; minimal directions to reach anywhere.
---
card_id: sq7ANthp
---
How many vectors are needed for a **basis** in n-dimensional space?
---
Exactly n linearly independent vectors.
---
card_id: lJgnhO5U
---
What is the **standard basis** in 2D?
---
$$\mathbf{e}_1 = [1, 0], \mathbf{e}_2 = [0, 1]$$
Any vector $[x, y] = x\mathbf{e}_1 + y\mathbf{e}_2$
---
card_id: cecV1CiR
---
What is an **orthonormal basis**?
---
A basis where vectors are orthogonal to each other (dot product = 0) and have unit length (norm = 1).
---
card_id: lirGn2dR
---
What does the **Gram-Schmidt process** do?
---
Converts linearly independent vectors into an orthonormal basis.
---
card_id: ZdGFubOO
---
How does **Gram-Schmidt** work?
---
1. Normalize first vector
2. For each subsequent vector: subtract projections onto all previous orthonormal vectors, then normalize
---
card_id: 9n7FYRGt
---
What is the formula for **Gram-Schmidt** projection removal?
---
$$\mathbf{u}_i = \frac{\mathbf{v}_i - \sum_j (\mathbf{v}_i \cdot \mathbf{u}_j)\mathbf{u}_j}{||\mathbf{v}_i - \sum_j (\mathbf{v}_i \cdot \mathbf{u}_j)\mathbf{u}_j||}$$
Subtract all projections to remove overlap.
---
card_id: adHIpgtZ
---
What is the **normal equation** for linear regression?
---
$$\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
Solves $\mathbf{y} = \mathbf{Xw}$ for weights w.
---
card_id: 5gHkqjdl
---
When does the **normal equation** fail?
---
When $\mathbf{X}^T \mathbf{X}$ is singular (not invertible), typically due to linearly dependent features.
---
card_id: OXu3u9fH
---
What is the **attention mechanism** formula?
---
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}}\right) \mathbf{V}$$
Computes weighted combination based on query-key similarity.
---
card_id: rla3KEtr
---
What are **Q, K, V** in the attention mechanism?
---
Q (queries): what to look for; K (keys): what's available; V (values): what to return.
---
card_id: cbSsBjKu
---
What does **QK^T** compute in attention?
---
All pairwise similarity scores between queries and keys (relationship matrix).
---
card_id: I2OrxUfT
---
What do the **attention weights** represent?
---
How much each query attends to each key; each row sums to 1 after softmax.
---
card_id: null
---
Model: In attention, $\mathbf{QK}^T$ has very large values before softmax. What happens?
---
Softmax saturates to near 0 or 1, losing gradients - that's why we divide by $\sqrt{d_k}$.
---
card_id: hRUSAR1u
---
Why is **matrix multiplication** ubiquitous in ML?
---
Enables layers, batch processing, attention, and GPU acceleration.
---
card_id: 8h1CKoTR
---
Model: You have shape (100, 784) data matrix and (784, 128) weight matrix. What operation and result shape?
---
Matrix multiplication: data @ weights. Result shape: (100, 128) - 100 samples with 128 output features.
---
card_id: 1UW1k6GB
---
Model: Feature matrix has rank 5 but 10 columns. What does this mean?
---
Only 5 truly independent features; remaining 5 are linear combinations of others, indicating redundancy or multicollinearity.
---
card_id: 6GRnClhP
---
Model: Which regularization for feature selection with many irrelevant features?
---
L1 (Lasso) regularization - diamond shape creates sparsity, driving irrelevant weights to exactly zero.
---
card_id: jHIcF2Ff
---
Model: PCA shows first component explains 95% variance. What to do?
---
Can reduce to just first principal component, capturing most information while drastically reducing dimensionality.
---
card_id: 2QeS6GR2
---
Model: Stacking 3 linear layers without activation. What's the effective model?
---
Single linear layer - composition of linear transformations is still linear: $\mathbf{W}_3 \mathbf{W}_2 \mathbf{W}_1 = \mathbf{W}_{\text{single}}$.
---
card_id: K5yynhw5
---
Model: Covariance matrix eigenvectors are not orthogonal. What's wrong?
---
Error - eigenvectors of symmetric matrices (like covariance) are always orthogonal; check computation.
---
card_id: d6hI177e
---
Model: Batch size 64, embedding dim 512. What's the shape of embeddings?
---
(64, 512) - 64 samples in batch, each with 512-dimensional embedding vector.
---
card_id: wO8VXnVi
---
Given **matrices** A (3×4) and B (5×3), can you compute A @ B?
---
No - incompatible shapes. For A @ B, inner dimensions must match (4 ≠ 5).
---
card_id: WOuAhDRg
---
Given **matrices** A (3×4) and B (4×5), what is the shape of A @ B?
---
(3×5) - outer dimensions: 3 rows from A, 5 columns from B.
---
card_id: dkP2tU4r
---
Model: Computing $(\mathbf{AB})^T$. Can you compute $\mathbf{A}^T \mathbf{B}^T$?
---
No - incorrect order. Must use $(\mathbf{AB})^T = \mathbf{B}^T \mathbf{A}^T$ (reverses multiplication order).
---
card_id: AJMREAEs
---
Model: Vector dot product is negative. What does this mean?
---
Vectors point in generally opposite directions (angle > 90°).
---
card_id: ZIozYxC0
---
Model: Two vectors have dot product zero. Are they orthogonal?
---
Yes - zero dot product means perpendicular (90° angle).
---
card_id: null
---
Model: Word embeddings for "king" and "queen" have dot product 0.85. What does this indicate?
---
High similarity - they point in nearly the same semantic direction.
---
card_id: z21uxY7i
---
When does repeated **matrix multiplication** make eigenvector direction dominate?
---
When applying the same matrix repeatedly ($\mathbf{A}^n$), the direction of the largest eigenvalue grows fastest and dominates.
---
card_id: ONjluYN8
---
How do **batch operations** improve efficiency?
---
Matrix operations process all samples simultaneously, leveraging GPU parallelism for 10-100x speedup over loops.
---
card_id: Yhhdwzql
---
Why is the **Euclidean norm** called "as the crow flies"?
---
It measures straight-line distance, the shortest path between two points.
---
card_id: 4fWE4ZgW
---
Why is the **Manhattan norm** called "city-block distance"?
---
It measures distance along grid lines (like city streets), summing movement in each axis direction.
---
card_id: PtbRMzH2
---
What is **cosine similarity** for word embeddings?
---
$$\text{similarity} = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \, ||\mathbf{b}||} = \cos(\theta)$$
Dot product of normalized vectors; measures angular similarity.
---
card_id: VXPlHBGc
---
Model: Scaling transformation $\begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}$. What happens to unit square?
---
Rectangle: width doubled (2x in x), height halved (0.5x in y).
---
card_id: zroMoRLW
---
Model: Rotation by 90°. What's the matrix?
---
$$\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$
Since $\cos(90°) = 0, \sin(90°) = 1$.
---
card_id: 5ojjQOvN
---
Model: Matrix has determinant = 0. Can you find its inverse?
---
No - determinant = 0 means singular matrix with no inverse.
---
card_id: null
---
Model: Solving $\mathbf{Ax} = \mathbf{b}$ but $\mathbf{A}$ has rank 3 with 5 columns. What's the issue?
---
System is underdetermined - infinite solutions exist due to redundant features.
---
card_id: kmlMo8qe
---
What property connects **determinant** and **matrix invertibility**?
---
Matrix is invertible if and only if determinant ≠ 0; det = 0 means singular (no inverse).
---
card_id: 9N4mtDp5
---
Model: Apply scaling then rotation vs rotation then scaling. Same result?
---
Generally different - matrix multiplication not commutative, so transformation order matters.
---
card_id: WORTJHn8
---
Why divide by $\sqrt{d_k}$ in **attention mechanism**?
---
Prevents dot products from becoming too large (which would push softmax into extreme values), stabilizing gradients.
---
