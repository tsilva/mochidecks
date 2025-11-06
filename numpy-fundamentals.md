---
card_id: null
---
What is **NumPy**?
---
A Python library for efficient numerical operations on multi-dimensional arrays.
---
card_id: null
---
What is a **NumPy array**?
---
A multi-dimensional container for homogeneous data with fixed size and type.
---
card_id: null
---
What methods create **NumPy arrays** from scratch?
---
- `np.array()`: from Python lists
- `np.zeros()`: array of zeros
- `np.ones()`: array of ones
- `np.eye()`: identity matrix
- `np.arange()`: range of values
- `np.linspace()`: evenly spaced values
---
card_id: null
---
In **NumPy**, what does `np.arange(start, stop, step)` do?
---
Creates an array with values from start to stop (exclusive) with specified step size.
---
card_id: null
---
In **NumPy**, what does `np.linspace(start, stop, num)` do?
---
Creates an array with num evenly spaced values between start and stop (inclusive).
---
card_id: null
---
What is **array reshaping** in NumPy?
---
Transforming array dimensions without changing the underlying data or element order.
---
card_id: null
---
In **NumPy**, what does the `-1` parameter mean in `reshape()`?
---
Automatically infers that dimension size based on the total number of elements.
---
card_id: null
---
In **NumPy**, what does `arr.reshape(2, -1)` do for a 12-element array?
---
Creates a 2×6 array (2 rows, 6 columns automatically inferred).
---
card_id: null
---
What is **indexing** in NumPy arrays?
---
Accessing specific elements using integer positions or coordinates in multi-dimensional arrays.
---
card_id: null
---
In **NumPy**, how do you access the element at row 1, column 2 in a 2D array `arr`?
---
```python
arr[1, 2]
```
---
card_id: null
---
What is **slicing** in NumPy?
---
Extracting subarrays using `start:stop:step` notation without copying data.
---
card_id: null
---
In **NumPy**, what does `arr[::2]` do?
---
Selects every second element from the array (step of 2, starting at 0).
---
card_id: null
---
In **NumPy**, what does `arr[:, 1]` do for a 2D array?
---
Selects all rows from column 1 (extracts the second column).
---
card_id: null
---
What is **boolean masking** in NumPy?
---
Selecting array elements using a boolean array as a filter condition.
---
card_id: null
---
In **NumPy**, what does `arr[arr > 5]` do?
---
Returns a new array containing only elements from `arr` that are greater than 5.
---
card_id: null
---
What is **fancy indexing** in NumPy?
---
Selecting array elements using integer arrays to specify multiple positions simultaneously.
---
card_id: null
---
In **NumPy**, what does `arr[[0, 2, 4]]` do?
---
Returns a new array with elements at indices 0, 2, and 4.
---
card_id: null
---
What are **element-wise operations** in NumPy?
---
Arithmetic operations applied simultaneously to all array elements using vectorization.
---
card_id: null
---
What are **universal functions (ufuncs)** in NumPy?
---
Pre-optimized functions that operate element-wise on arrays with vectorized C implementations.
---
card_id: null
---
What common **ufuncs** does NumPy provide?
---
- `np.sin()`, `np.cos()`: trigonometric functions
- `np.exp()`: exponential
- `np.abs()`: absolute value
- `np.sqrt()`: square root
---
card_id: null
---
What are **aggregation functions** in NumPy?
---
Operations that reduce arrays to summary statistics along specified axes or entire arrays.
---
card_id: null
---
What common **aggregation functions** does NumPy provide?
---
- `np.sum()`: total of elements
- `np.mean()`: average value
- `np.std()`: standard deviation
- `np.min()`, `np.max()`: minimum and maximum
---
card_id: null
---
In **NumPy**, what does the `axis` parameter control in aggregation functions?
---
Which dimension to reduce: `axis=0` for columns, `axis=1` for rows in 2D arrays.
---
card_id: null
---
In **NumPy**, what does `arr.sum(axis=0)` do for a 2D array?
---
Sums down columns, returning a 1D array with one sum per column.
---
card_id: null
---
In **NumPy**, what does `arr.sum(axis=1)` do for a 2D array?
---
Sums across rows, returning a 1D array with one sum per row.
---
card_id: null
---
What is **broadcasting** in NumPy?
---
Automatic alignment of arrays with different dimensions during arithmetic operations without copying data.
---
card_id: null
---
When does **broadcasting** apply in NumPy?
---
When operating on arrays with different shapes where dimensions are compatible or equal to 1.
---
card_id: null
---
How does NumPy **broadcast** a (3,) array with a (3, 4) array?
---
The (3,) array is treated as (3, 1) and stretched across columns to match (3, 4).
---
card_id: null
---
What are the **broadcasting rules** in NumPy?
---
1. Dimensions compared from right to left
2. Dimensions compatible if equal or one is 1
3. Dimensions of size 1 stretched to match
4. Missing dimensions added on the left
---
card_id: null
---
Why is **broadcasting** important for neural networks in **NumPy**?
---
Enables efficient batch operations and parameter updates without explicit loops or memory duplication.
---
card_id: null
---
What operator performs **matrix multiplication** in NumPy?
---
The `@` operator (e.g., `A @ B`).
---
card_id: null
---
How does **matrix multiplication** (`@`) differ from element-wise multiplication (`*`) in NumPy?
---
**Matrix multiplication (`@`)**: Dot product of rows and columns per linear algebra
**Element-wise (`*`)**: Multiplies corresponding elements directly
---
card_id: null
---
What does `arr.T` do in NumPy?
---
Returns the transpose of the array (rows become columns, columns become rows).
---
card_id: null
---
In **NumPy**, what is the shape of `arr.T` if `arr.shape` is `(3, 4)`?
---
`(4, 3)`
---
card_id: null
---
Why learn **NumPy** for deep learning?
---
PyTorch tensors behave similarly to NumPy arrays, making NumPy fundamental for scientific Python.
---
card_id: null
---
What does `np.eye(n)` create?
---
An n×n identity matrix with ones on the diagonal and zeros elsewhere.
---
card_id: null
---
What is **vectorization** in NumPy?
---
Replacing explicit Python loops with array operations that execute in optimized C code.
---
card_id: null
---
In **NumPy**, model trains slowly using Python loops for array operations. What solution?
---
Use vectorized operations to replace loops with element-wise array operations.
---
card_id: null
---
In **NumPy**, you need to add a bias vector to each row of a batch matrix. What feature enables this?
---
Broadcasting: add the 1D bias vector to the 2D matrix directly.
---
card_id: null
---
In **NumPy**, you have shape (100, 64) batch and shape (64,) bias. What happens with `batch + bias`?
---
Broadcasting expands bias to (1, 64), then to (100, 64), adding bias to each row.
---
card_id: null
---
What's wrong with `arr[arr > 5 and arr < 10]` in NumPy?
---
Cannot use `and`/`or` with arrays; use element-wise `&`/`|` with parentheses: `arr[(arr > 5) & (arr < 10)]`.
---
