# NumPy Cheat Sheet with Examples and Comments

import numpy as np

# -------------------------
# 1. Creating Arrays
# -------------------------
a = np.array([1, 2, 3])               # From list
b = np.zeros((2, 3))                  # 2x3 array of zeros
c = np.ones((3,))                     # Vector of ones
d = np.full((2, 2), 7)                # Filled with 7s
e = np.eye(3)                         # Identity matrix
f = np.arange(0, 10, 2)               # [0 2 4 6 8]
g = np.linspace(0, 1, 5)              # 5 points between 0 and 1

# -------------------------
# 2. Inspecting Arrays
# -------------------------
print(a.shape)         # (3,)
print(b.ndim)          # 2
print(c.dtype)         # float64
print(d.size)          # 4
print(e.itemsize)      # Size of each item in bytes

# -------------------------
# 3. Indexing & Slicing
# -------------------------
print(a[1])            # Second element
print(f[1:4])          # Elements from index 1 to 3
print(b[:, 0])         # All rows, first column
print(b[1:, 1:])       # Submatrix
print(f[-1])           # Last element

# -------------------------
# 4. Reshaping
# -------------------------
reshaped = f.reshape(1, 5)
flat = f.flatten()               # 1D array
transposed = b.T                 # Transpose
expanded = np.expand_dims(a, axis=0) # Add a row dimension

# -------------------------
# 5. Math Operations
# -------------------------
print(a + 5)
print(a * 2)
print(a + a)
print(np.sum(a))
print(np.mean(a))
print(np.std(a))
print(np.max(b, axis=0))
print(np.clip(a, 1, 2))          # Clip values to [1, 2]

# -------------------------
# 6. Boolean Masking
# -------------------------
mask = a > 1
print(a[mask])
print(np.where(a > 1, 100, 0))
mask2 = (a > 1) & (a < 3)
print(a[mask2])

# -------------------------
# 7. String Operations
# -------------------------
str_arr = np.array(['5', '3', '.', '7'])
print(np.char.isdigit(str_arr))
print(np.char.replace(str_arr, '.', '0'))
print(np.char.lower(np.array(['HELLO', 'WORLD'])))

# -------------------------
# 8. Random Numbers
# -------------------------
np.random.seed(42)
print(np.random.rand(2, 3))          # Uniform [0,1)
print(np.random.randn(2, 3))         # Normal distribution
print(np.random.randint(0, 10, 5))   # Random integers

# -------------------------
# 9. Sorting & Searching
# -------------------------
unsorted = np.array([3, 1, 2])
print(np.sort(unsorted))
print(np.argsort(unsorted))
print(np.unique([1, 2, 2, 3]))
print(np.argmax(a))
print(np.argmin(a))

# -------------------------
# 10. Combining & Splitting
# -------------------------
x = np.array([1, 2])
y = np.array([3, 4])
print(np.concatenate([x, y]))
print(np.stack([x, y], axis=0))
print(np.hstack([x, y]))
print(np.vstack([x, y]))
print(np.split(np.array([1, 2, 3, 4, 5, 6]), 3))

# -------------------------
# 11. Cleaning Up
# -------------------------
arr_with_nan = np.array([1, np.nan, 2, np.inf])
print(np.isnan(arr_with_nan))
print(arr_with_nan[~np.isnan(arr_with_nan)])
print(np.nan_to_num(arr_with_nan))

# -------------------------
# 12. Utility Functions
# -------------------------
print(np.all(a > 0))
print(np.any(a < 0))
print(np.cumsum(a))
print(np.diff(a))
