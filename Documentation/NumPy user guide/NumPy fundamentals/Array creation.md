# Array creation

**See also**
[Array creation routines](https://numpy.org/doc/stable/reference/routines.array-creation.html#routines-array-creation)

## Introduction

There are 6 general mechanisms for creating arrays:

1. Conversion from other Python structures (i.e. lists and tuples)
2. Intrinsic NumPy array creation functions (e.g. arange, ones, zeros, etc.)
3. Replicating, joining, or mutating existing arrays 
4. Reading arrays from disk, either from standard or custom formats
5. Creating arrays from raw bytes through the use of strings or buffers
6. Use of special library functions (e.g., random)

You can use these methods to create ndarrays or [Structured arrays](https://numpy.org/doc/stable/user/basics.rec.html#structured-arrays). This document will cover general methods for ndarray creation.

## 1) Converting Python sequences to NumPy Arrays

NumPy arrays can be defined using Python sequences such as lists and tuples. Lists and tuples are defined using `[...]` and `(...)`, respectively. Lists and tuples can difine ndarray creation:

- a list of numbers will create a 1D array,
- a list of lists will create a 2D array,
- futher nested lists will create higher-dimensional arrays. In general, any array object is called an **ndarray** in NumPy.

```python
>>> a1D = np.array([1, 2, 3, 4])
>>> a2D = np.array([[1, 2], [3, 4]])
>>> a3D = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
```

When you use [`numpy.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html#numpy.array "numpy.array") to define a new array, you should consider the  [dtype](https://numpy.org/doc/stable/user/basics.types.html) of the elements in the array, which can be specified explicitly. This feature gives you more control over the underlying data structures and how the elements are handled in C/C++ functions. If you are not carefull with `dtype` assignments, you can get unwanted overflow, as such

```python
>>> a = np.array([127, 128, 129], dtype=np.int8)
>>> a
array([ 127, -128, -127], dtype=int8)
```

An 8-bit signed integer represents integers from -128 to 127. Assigning the `int8` array to integers outside of this range results in overflow. This feature can often be misunderstood. If you perform calculations with mismathing `dyptes`, you can get unwanted results, for example:

```python
>>> a = np.array([2, 3, 4], dtype=np.uint32)
>>> b = np.array([5, 6, 7], dtype=np.uint32)
>>> c_unsigned32 = a - b
>>> print('unsigned c:', c_unsigned32, c_unsigned32.dtype)
unsigned c: [4294967293 4294967293 4294967293] uint32
>>> c_signed32 = a - b.astype(np.int32)
>>> print('signed c:', c_signed32, c_signed32.dtype)
signed c: [-3 -3 -3] int64
```

Notice when you perform operations two arrays of the same `dtype`: `uint32`, the resulting array is the same type. When you perform operations with different `dtype`, NumPy will assign a new type that satisfies all of the array elements involved in the computation, here `uint32` and `int32` can both be represented in as `int64`.

The default NumPy behavior is to create arrays in either 32 or 64-bit signed integers (platform dependent and matches C `long` size) or double precision floating point numbers. If you expect your integer arrays to be a specific type, then you need to specify the dtype while you create the array.

## 2) Intrinsic NumPy array creation functions

NumPy has over 40 built-in functions for creating arrays as laid out in the [Array creation routines](https://numpy.org/doc/stable/reference/routines.array-creation.html#routines-array-creation). These functions can be split into roughly three categories, based on the dimension of the array they create:

1. 1D arrays 
2. 2D arrays 
3. ndarrays 

### 1 - 1D array creation functions 

The 1D array creation functions e.g. [`numpy.linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html#numpy.linspace "numpy.linspace") and [`numpy.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html#numpy.arange "numpy.arange") generally need at least two inputs, `start` and `stop`.

[`numpy.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html#numpy.arange "numpy.arange") creates arrays with regularly incrementing values. Check the documentation for complete information and examples. A few examples are shown:

```python
>>> np.arange(10)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.arange(2, 10, dtype=float)
array([2., 3., 4., 5., 6., 7., 8., 9.])
>>> np.arange(2, 3, 0.1)
array([2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9])
```

Note: best practice for [`numpy.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html#numpy.arange "numpy.arange") is to use integer start, end, and step values. There are some subtleties regarding `dtype`. In the second example, the `dtype` is defined. In the third example, the array is `dtype=float` to accommodate the step size of `0.1`. Due to roundoff error, the `stop` value is somtimes included.

[`numpy.linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html#numpy.linspace "numpy.linspace") will create arrays with a specified number of elements, and spaced equally between the specified beginning and values. For example:

```python
>>> np.linspace(1., 4., 6)
array([1. , 1.6, 2.2, 2.8, 3.4, 4. ])
```

The advantage of this creation function is that you guarantee the number of elements and the starting and end point. The previous `arange(start, stop, step)` will not include the value `stop`.
