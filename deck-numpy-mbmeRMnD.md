---
card_id: 0AwXA22m
---
What are the **broadcasting rules** in NumPy?
---
1. Dimensions compared from right to left
2. Dimensions compatible if equal or one is 1
3. Dimensions of size 1 stretched to match
4. Missing dimensions added on the left
---
card_id: 4dyh9mmR
---
In **NumPy**, what does `arr[[0, 2, 4]]` do?
---
Returns a new array with elements at indices 0, 2, and 4.
---
card_id: 5XMs8xXE
---
What does `np.eye(n)` create?
---
An n×n identity matrix with ones on the diagonal and zeros elsewhere.
---
card_id: 5czhkB67
---
Why is **broadcasting** important in NumPy?
---
Enables efficient operations on arrays with different shapes without explicit loops or memory duplication.
---
card_id: 5n3kSMmn
---
In **NumPy**, what does `np.arange(start, stop, step)` do?
---
Creates an array with values from start to stop (exclusive) with specified step size.
---
card_id: 5o1ztwck
---
What's wrong with `arr[arr > 5 and arr < 10]` in NumPy?
---
Cannot use `and`/`or` with arrays; use element-wise `&`/`|` with parentheses: `arr[(arr > 5) & (arr < 10)]`.
---
card_id: 74caSzfu
---
How does NumPy **broadcast** a (3,) array with a (3, 4) array?
---
The (3,) array is treated as (3, 1) and stretched across columns to match (3, 4).
---
card_id: 7Fmdf84L
---
What does `np.sum()` compute?
---
The total of all elements in the array or along specified axes.
---
card_id: 7e1Eqduk
---
How do **indexing** and **slicing** differ in NumPy?
---
**Indexing**: Returns a single element / **Slicing**: Returns a view of multiple elements
---
card_id: 8PehIcrK
---
In **NumPy**, what is the shape of `arr.T` if `arr.shape` is `(3, 4)`?
---
`(4, 3)`
---
card_id: 94UTt6Ow
---
What is **slicing** in NumPy?
---
Extracting subarrays using `start:stop:step` notation without copying data.
---
card_id: 9gfiyIsR
---
In **NumPy**, what does `arr[::2]` do?
---
Selects every second element from the array (step of 2, starting at 0).
---
card_id: 9oaY3PYQ
---
In **NumPy**, what is the resulting shape of `arr1 + arr2` where `arr1.shape = (3, 4)` and `arr2.shape = (4,)`?
---
`(3, 4)` - the (4,) array broadcasts across rows.
---
card_id: AL6hsAt2
---
Why use `keepdims=True` in aggregation functions?
---
Preserves dimensions as size 1 for correct broadcasting in subsequent operations.
---
card_id: AyRQ3fRl
---
What does `np.array()` do?
---
Creates a NumPy array from a Python list or tuple.
---
card_id: CATNeqAR
---
Given **NumPy** arrays `a = np.array([[1], [2], [3]])` (shape 3,1) and `b = np.array([10, 20])` (shape 2,), what is `a + b`?
---
```python
[[11, 21],
 [12, 22],
 [13, 23]]
```
Shape (3, 2).
---
card_id: DCuSsUCw
---
You need to normalize each row of a (100, 10) array by its row sum. How?
---
```python
row_sums = arr.sum(axis=1, keepdims=True)
normalized = arr / row_sums
```
---
card_id: DsQXhivF
---
What are **element-wise operations** in NumPy?
---
Arithmetic operations applied simultaneously to all array elements using vectorization.
---
card_id: E7zBqvsa
---
What are **aggregation functions** in NumPy?
---
Operations that reduce arrays to summary statistics along specified axes or entire arrays.
---
card_id: EPKOh7hT
---
How do `np.arange()` and `np.linspace()` differ?
---
**arange**: Specifies step size, stop is exclusive / **linspace**: Specifies number of points, stop is inclusive
---
card_id: FGWQzVQj
---
In **NumPy**, can you add `arr1` with `arr2` where `arr1.shape = (3, 4)` and `arr2.shape = (3,)`?
---
Yes - the (3,) broadcasts to (3, 1) then to (3, 4) matching rows.
---
card_id: FPgZeo51
---
In **NumPy**, how do you access the element at row 1, column 2 in a 2D array `arr`?
---
```python
arr[1, 2]
```
---
card_id: Hdnpa2C4
---
In **NumPy**, what does `arr.sum(axis=0)` do for a 2D array?
---
Sums down columns, returning a 1D array with one sum per column.
---
card_id: IejhuLmF
---
What does `arr.T` do in NumPy?
---
Returns the transpose of the array (rows become columns, columns become rows).
---
card_id: JAxL7sok
---
When should you use **vectorization** in NumPy?
---
- Have element-wise operations on arrays
- Want to avoid explicit Python loops
- Need performance improvement for numerical computations
---
card_id: KC7spGbD
---
What does `np.sqrt()` do?
---
Computes square root of each element in the array.
---
card_id: KTmfrWXP
---
In **NumPy**, what does the `axis` parameter control in aggregation functions?
---
Which dimension to reduce: `axis=0` for columns, `axis=1` for rows in 2D arrays.
---
card_id: KkLDH6H1
---
What advantage does **NumPy** provide over Python lists?
---
Faster computation through vectorization and contiguous memory allocation for homogeneous data.
---
card_id: MJOv13nr
---
What is **indexing** in NumPy arrays?
---
Accessing specific elements using integer positions or coordinates in multi-dimensional arrays.
---
card_id: MWhNe67M
---
In **NumPy**, you have shape (100, 64) batch and shape (64,) bias. What happens with `batch + bias`?
---
Broadcasting expands bias to (1, 64), then to (100, 64), adding bias to each row.
---
card_id: NycYZ0ku
---
What does `np.zeros(shape)` create?
---
An array of specified shape filled with zeros.
---
card_id: OCTor9FO
---
In **NumPy**, what is the resulting shape of `arr1 + arr2` where `arr1.shape = (2, 3, 4)` and `arr2.shape = (4,)`?
---
`(2, 3, 4)` - the (4,) broadcasts across the first two dimensions.
---
card_id: Oo9ic9Jz
---
What is **array reshaping** in NumPy?
---
Transforming array dimensions without changing the underlying data or element order.
---
card_id: QV74RBJA
---
You're adding bias to 1000 samples in a (1000, 512) batch. How does **broadcasting** help?
---
Bias shape (512,) broadcasts to each row without creating 1000 copies in memory.
---
card_id: Qz8wmeHY
---
When should you use **NumPy** over Python lists?
---
- Need numerical computations on large datasets
- Require multi-dimensional data structures
- Want vectorized operations without explicit loops
- Need memory-efficient storage
---
card_id: RhdqA0aE
---
In **NumPy**, what is the resulting shape of `arr1 + arr2` where `arr1.shape = (3, 4)` and `arr2` is a scalar?
---
`(3, 4)` - scalar broadcasts to all elements.
---
card_id: RiKJvYJB
---
What does `np.std()` compute?
---
The standard deviation of array elements or along specified axes.
---
card_id: Sf5pG9JY
---
In **NumPy**, what is the resulting shape of `arr1 + arr2` where `arr1.shape = (2, 1, 4)` and `arr2.shape = (1, 3, 1)`?
---
`(2, 3, 4)` - dimension 1 in arr1 and dimensions 0,2 in arr2 broadcast.
---
card_id: UT46paj8
---
What does **vectorization** enable in NumPy?
---
Applying operations to entire arrays without explicit Python loops for faster execution.
---
card_id: UgMklXjz
---
Given **NumPy** arrays `a = np.ones((2, 3, 1))` and `b = np.ones((1, 1, 4))`, what is the shape of `a * b`?
---
`(2, 3, 4)` - all three dimensions broadcast.
---
card_id: X4oqL9pN
---
In **NumPy**, can you add `arr1` with `arr2` where `arr1.shape = (3, 4)` and `arr2.shape = (5,)`?
---
No - incompatible shapes. The (5,) doesn't match either dimension of (3, 4).
---
card_id: XNyGHxJ2
---
What is **NumPy**?
---
A Python library for efficient numerical operations on multi-dimensional arrays.
---
card_id: Y8Ue3LVO
---
Your code produces NaN values during array operations. What are likely causes?
---
- Division by zero
- Taking sqrt/log of negative numbers
- Overflow from very large numbers
- Invalid broadcasting creating unexpected shapes
---
card_id: ZJuu6k1d
---
How does **matrix multiplication** (`@`) differ from element-wise multiplication (`*`)?
---
**@**: Linear algebra operation requiring compatible inner dimensions / **`*`**: Element-wise with broadcasting
---
card_id: ZueTmYGT
---
In **NumPy**, what does `arr[:, 1]` do for a 2D array?
---
Selects all rows from column 1 (extracts the second column).
---
card_id: a8A7ketm
---
In **NumPy**, what happens with `arr1 + arr2` where `arr1.shape = (3, 4)` and `arr2.shape = (1,)`?
---
Result has shape (3, 4) - the (1,) broadcasts to all elements like a scalar.
---
card_id: aL1Bg9oN
---
What is a **ufunc** in NumPy?
---
A universal function that operates element-wise on arrays with broadcasting support.
---
card_id: aiHkZyXt
---
How do **boolean masking** and **fancy indexing** differ?
---
**Boolean masking**: Uses boolean array as filter / **Fancy indexing**: Uses integer array to specify positions
---
card_id: bHkm9wRR
---
What is **broadcasting** in NumPy?
---
Automatic alignment of arrays with different dimensions during arithmetic operations without copying data.
---
card_id: c5D6ijjf
---
In **NumPy**, what does `arr.sum(axis=1)` do for a 2D array?
---
Sums across rows, returning a 1D array with one sum per row.
---
card_id: eh2pOWpY
---
In **NumPy**, what does `np.linspace(start, stop, num)` do?
---
Creates an array with num evenly spaced values between start and stop (inclusive).
---
card_id: fL1Ex9cF
---
In **NumPy**, what does `arr[arr > 5]` do?
---
Returns a new array containing only elements from `arr` that are greater than 5.
---
card_id: gM6QYfVS
---
What is a **NumPy array**?
---
A multi-dimensional container for homogeneous data with fixed size and type.
---
card_id: h2GLjN0I
---
What does `np.sin()` do?
---
Computes sine of each element in the array.
---
card_id: iVDarG9b
---
You have shape (10, 20) array. What shape results from `arr.sum(axis=0)`?
---
`(20,)` - reduces the first dimension.
---
card_id: ieKVAGGI
---
Given **NumPy** arrays `a = np.array([1, 2, 3])` (shape 3,) and `b = np.array([[10], [20]])` (shape 2,1), what is `a + b`?
---
```python
[[11, 12, 13],
 [21, 22, 23]]
```
Shape (2, 3).
---
card_id: imA0PuU8
---
In **NumPy**, what is the resulting shape of `arr1 + arr2` where `arr1.shape = (3, 4)` and `arr2.shape = (3, 1)`?
---
`(3, 4)` - the (3, 1) array broadcasts across columns.
---
card_id: ioAroUxQ
---
What does `np.ones(shape)` create?
---
An array of specified shape filled with ones.
---
card_id: jWhYhY3d
---
In **NumPy**, what does `arr.reshape(2, -1)` do for a 12-element array?
---
Creates a 2×6 array (2 rows, 6 columns automatically inferred).
---
card_id: kUUtxjGw
---
What operator performs **matrix multiplication** in NumPy?
---
The `@` operator (e.g., `A @ B`).
---
card_id: kwVIrGOB
---
In **NumPy**, what is the resulting shape of `arr1 + arr2` where `arr1.shape = (2, 3, 4)` and `arr2.shape = (3, 4)`?
---
`(2, 3, 4)` - the (3, 4) broadcasts across the first dimension.
---
card_id: la6RbJ6P
---
Array operation is slow despite using NumPy. What might be wrong?
---
- Using Python loops instead of vectorization
- Creating unnecessary copies
- Operations triggering type conversions
- Not using in-place operations where appropriate
---
card_id: ldY3u5lL
---
What is **fancy indexing** in NumPy?
---
Selecting array elements using integer arrays to specify multiple positions simultaneously.
---
card_id: o99cT9GG
---
In **NumPy**, what is the resulting shape of `arr1 * arr2` where `arr1.shape = (3, 1)` and `arr2.shape = (1, 4)`?
---
`(3, 4)` - both dimensions of size 1 stretch to create outer product structure.
---
card_id: okNV61WZ
---
When should you use `np.dot()` versus `@` in NumPy?
---
Both do matrix multiplication; `@` is preferred for clarity and better follows mathematical notation.
---
card_id: oySDoLbD
---
In **NumPy**, what is the resulting shape of `arr1 + arr2` where `arr1.shape = (5, 1, 3)` and `arr2.shape = (1, 4, 3)`?
---
`(5, 4, 3)` - second dimensions 1 in arr1 and first dimension 1 in arr2 broadcast.
---
card_id: pTxXztVJ
---
Why is **broadcasting** important for neural networks in **NumPy**?
---
Enables efficient batch operations and parameter updates without explicit loops or memory duplication.
---
card_id: pkrnNlp2
---
What does `np.exp()` do?
---
Computes exponential (e^x) of each element in the array.
---
card_id: udThFceU
---
You have shape (10, 20) array. What shape results from `arr.mean(axis=1)`?
---
`(10,)` - reduces the second dimension.
---
card_id: vKbBKtwc
---
In **NumPy**, what does the `-1` parameter mean in `reshape()`?
---
Automatically infers that dimension size based on the total number of elements.
---
card_id: w2ChrU8J
---
In **NumPy**, can you add `arr1` with `arr2` where `arr1.shape = (3, 4, 5)` and `arr2.shape = (4, 1)`?
---
Yes - arr2 broadcasts to (1, 4, 1) then to (3, 4, 5).
---
card_id: w4z0c3pG
---
You have a 24-element 1D array. You call `arr.reshape(3, 3)`. What happens?
---
Raises ValueError - total size (9) doesn't match original size (24).
---
card_id: xeK4lclA
---
What is **boolean masking** in NumPy?
---
Selecting array elements using a boolean array as a filter condition.
---
card_id: yFRLBIOy
---
What does `np.mean()` compute?
---
The average value of array elements or along specified axes.
---
card_id: zFYHBaSh
---
When does **broadcasting** apply in NumPy?
---
When operating on arrays with different shapes where dimensions are compatible or equal to 1.
---
