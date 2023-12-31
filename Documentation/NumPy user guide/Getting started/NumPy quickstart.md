# NumPy quickstart

## Prerequisites

You'll need to know a bit of Python. For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

To work the examples, you'll need `matplotlib` installed in addition to NumPy.

### Lnearner profile

This is a quick overview of arrays in NumPy. It deonstrates how n-dimensional ($n \geq 2$) arrays are represented and can be manipulated. In particular, if you don't know how to apply common functions to n-dimensional arrays (without using for-loops), or if you want to understand axis and shape properties for n-dimensional arrays, this article might be of help.

### Learning Objectives

After reading, you should be able to:

- Understand the difference between one-, two- and n-dimensional arrays in NumPy;
- Understand how to apply some linear algebra operations to n-dimensional arrays without using for-loops;
- Understand axis and shape properties for n-dimensional arrays.

## The Basics

NumPy's main object is the homogeneous multidimensional array. It is a table of elements (usually numbers), all of the same type, indexed by a tuple of non-negative integers. In NumPy dimensions are called *axes*.

For example, the array for the coordinates of a point in 3D space, `[1, 2, 1]`, has on axis. That axis has 3 elements in it, so we say it has a length of 3. In the example pictured below, the array has 2 axes. The first axis has a length of 2, the second axis has a length of 3.

```bash
[[1., 0., 0.,],
 [0., 1., 2.]]
```

NumPy's array class is called `ndarray`. It is also known by the alias `array`. Note that `numpy.array` is not the same as the Standard Python Library class `array.array`, which only handles one-dimensional arrays and offers less functionality. The more important attributes of an `ndarray` object are:

**ndarray.ndim**
	the number of axes (dimensions) of the array.

**ndarray.shape**
	the dimensions of the array. This is a tuple of intergers indicating the size of the array in each dimension. For a matrix with *n* rows and *m* columns, `shape` will be `(n, m)`. The length of the `shape` tuple is therefore the number of axes, `ndim`.

**ndarray.size**
	the total number of elements of the array. This is equal to the product of the elements of `shape`.

**ndarray.dtype**
	an object describing the type of the elements in the array. One can create or specify dtype's using standard Python types. Additionally NumPy provides types of its own. numpy.int32, numpy.int16, and numpy.float64 are some examples.

**ndarray.itemsize**
	the size in bytes of each element of the array. For example, an array of elements of type `float64` has `itemsize` 8 (=64/8), while one of type `complex32` has `itemsize` 4 (=32/8). It is equivalent to `ndarray.dtype.itemsize`.

**ndarray.data**
	the buffer containing the actual elements of the array. Normally, we won't need to use this attribute because we will access the elements in an array using indexing facilities.

### An example

```python
>>> import numpy as np
>>> a = np.arange(15).reshape(3, 5)
>>> a
array([[ 0, 1, 2, 3, 4],
	   [ 5, 6, 7, 8, 9],
	   [10, 11, 12, 13, 14]])
>>> a.shape
(3, 5)
>>> a.ndim
2
>>> a.dtype.name
'int64'
>>> a.itemsize
8
>>> a.size
15
>>> type(a)
<class 'numpy.ndarray'>
>>> b = np.array([6, 7, 8])
>>> b
array([6, 7, 8])
>>> type(b)
<class 'numpy.ndarray'>
```

### Array Creation

There are several ways to create arrays.

For example, you can create an array from a regular Python list or tuple using the `array` function. The type of the resulting array is deduced from the type of the elements in the sequences.

```python
>>> import numpy as np
>>> a = np.array([2, 3, 4])
>>> a
array([2, 3, 4])
>>> a.dtype
dtype('int64')
>>> b = np.array([1.2, 3.5, 5.1])
>>> b.dtype
dtype('float64')
```

A frequent error consists in calling `array` with multiple arguments, rather than providing a single sequence as an argument.

```python
>>> a = np.array(1, 2, 3, 4) # WRONG
Traceback (most recent call last):
 ...
TypeError: array() takes from 1 to 2 positional argument but 4 were given
>>> a = np.array([1, 2, 3, 4]) # RIGHT
```

`array` transforms sequences of sequences into two-dimensional arrays, sequences of sequences of sequences into three-dimensioanl arrays, and so on.

```python
>>> b = np.array([(1.5, 2, 3), (4, 5, 6)])
>>> b
array([[1.5, 2. , 3. ],
	   [4. , 5. , 6. ]])
```

The type of the array can also be explicitly specified at creation time:
```python
>>> c = np.array([[1, 2], [3, 4]], dtype=complex)
>>> c
array([[1.+0.j, 2.+0.j],
	   [3.+0.j, 4.+0.j]])
```
Often, the elements of an array are origianlly unknown, but its size is known. Hence, NumPy offers several functions to create arrays with initial placeholder content. These minimize the necessity of growing arrays, an expensive operation.

The function `zeros` creates an array full of zeros, the function `ones` creates an array full of ones, and the function `empty` creates an array whose initial content is random and depends on the state of the memory. By default, the dtype of the created array is `float64`, but it can be specified via the key word argument `dtype`.

```python
>>> np.zeros((3, 4))
array([[0., 0., 0., 0.],
	   [0., 0., 0., 0.],
	   [0., 0., 0., 0.],
	   [0., 0., 0., 0.]])
>>> np.ones((2, 3, 4), dtype=np.int16)
array([[[1, 1, 1, 1],
	    [1, 1, 1, 1],
	    [1, 1, 1, 1]],
	    
	   [[1, 1, 1, 1],
	    [1, 1, 1, 1],
	    [1, 1, 1, 1]]], dtype=int16)
>>> np.emtpy((2, 3))
array([[3.73603959e-262, 6.02658058e-154, 6.55490914e-260],  # may vary
       [5.30498948e-313, 3.14673309e-307, 1.00000000e+000]])
```

To create sequences of numbers, NumPy provides the `arange` function which is analogous to the Python built-in `range`, but returns an array.

```python
>>> np.arange(10, 30, 5)
array([10, 15, 20, 25])
>>> np.arange(0, 2, 0.3) # it accepts float arguments
array([0., 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
```

When `arange` is used with floating point arguments, it is generally not possible to predict the number of elements obtained, due to the finite floating point precision. For this reason, it is usually better to use the function `linspace` that receives as an argument the number of elements that we want, instead of the step:

```python
>>> from numpy import pi
>>> np.linspace(0, 2, 9) # 9 numbers from 0 to 2
array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  ])
>>> x = np.linspace(0, 2 * pi, 100) # useful to evaluate function at lots of points
>>> f = np.sin(x)
```

**See also**
[`array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html#numpy.array "numpy.array"), [`zeros`](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html#numpy.zeros "numpy.zeros"), [`zeros_like`](https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html#numpy.zeros_like "numpy.zeros_like"), [`ones`](https://numpy.org/doc/stable/reference/generated/numpy.ones.html#numpy.ones "numpy.ones"), [`ones_like`](https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html#numpy.ones_like "numpy.ones_like"), [`empty`](https://numpy.org/doc/stable/reference/generated/numpy.empty.html#numpy.empty "numpy.empty"), [`empty_like`](https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html#numpy.empty_like "numpy.empty_like"), [`arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html#numpy.arange "numpy.arange"), [`linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html#numpy.linspace "numpy.linspace"), _numpy.random.Generator.rand_, _numpy.random.Generator.randn_, [`fromfunction`](https://numpy.org/doc/stable/reference/generated/numpy.fromfunction.html#numpy.fromfunction "numpy.fromfunction"), [`fromfile`](https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html#numpy.fromfile "numpy.fromfile")

### Printing Arrays 

When you print an array, NumPy displays it in a similar way to nested lists, but with the following layout:

- the last axis is printed from left to right,
- the second-to-last is printed from top to bottom,
- the rest are also printed from top to bottom, with each slice separated from the next by an emtpy line.

One-dimensional arrays are then printed as rows, bidimensionals as matrices and tridimensionals as lists of matrices.

```python
>>> a = np.arange(6) # 1d array
>>> print(a)
[0 1 2 3 4 5]
>>>
>>> b = np.arange(12).reshape(4, 3) # 2d array
>>> print(b)
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
>>>
>>> c = np.arange(24).reshape(2, 3, 4) # 3d array
>>> print(c)
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]
```

See [[#Shape Manipulation]] to get more details on `reshape`.

If an array is too large to be printed, NumPy automatically skips the central part of the array and only prints the corners:

```python
>>> print(np.arange(10000))
[   0    1    2 ... 9997 9998 9999]
>>>
>>> print(np.arange(10000).reshape(100, 100))
[[   0    1    2 ...   97   98   99]
 [ 100  101  102 ...  197  198  199]
 [ 200  201  202 ...  297  298  299]
 ...
 [9700 9701 9702 ... 9797 9798 9799]
 [9800 9801 9802 ... 9897 9898 9899]
 [9900 9901 9902 ... 9997 9998 9999]]
```

To disable this behaviour and force NumPy to print the entire array, you can change the printing options using `set_printoptions`.

```python
>>> np.set_printoptions(threshold=sys.maxsize) # sys module should be imported
```

### Basic Operations

Arithmetic operators on arrays apply *elementwise*. A new array is created and filled with the result.

```python
>>> a = np.array([20, 30, 40, 50])
>>> b = np.arange(4)
>>> b
array([0, 1, 2, 3])
>>> c = a - b
>>> c
array([20, 29, 39, 47])
>>> b**2
array([0, 1, 4, 9])
>>> 10 * np.sin(a)
array([ 9.12945251, -9.88031624,  7.4511316 , -2.62374854])
>>> a < 35
array([ True,  True, False, False])
```

Unlike in many matrix languages, the product operator `*` operates elementwise in NumPy arrays. The matrix product can be performed using the `@` operator (in python >= 3.5) or the `dot` function or method:

```python
>>> A = np.array([[1, 1],
...               [0, 1]])
>>> B = np.array([[2, 0],
...               [3, 4]])
>>> A * B # elementwise product
array([[2, 0],
       [0, 4]])
>>> A @ B # matrix product
array([[5, 4],
       [3, 4]])
>>> A.dot(B) # another matrix product
array([[5, 4],
       [3, 4]])
```

Some operations, such as `+=` and `*=`, act in place to modify an existing array rather than create a new one.

```python
>>> rg = np.random.default_rng(1) # create instance of default random number generator
>>> a = np.ones((2, 3), dtype=int)
>>> b = rg.random((2, 3))
>>> a *= 3
>>> a
array([[3, 3, 3],
	   [3, 3, 3]])
>>> b += a
>>> b
array([[3.51182162, 3.9504637 , 3.14415961],
       [3.94864945, 3.31183145, 3.42332645]])
>>> a + b # b is not automatically converted to integer type
Traceback (most recent call last):
    ...
numpy.core._exceptions._UFuncOutputCastingError: Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
```

When operating with arrays of different types, the type of the resulting array corresponds to the more general or precise one (a behavior known as upcasting).

```python
>>> a = np.ones(3, dtype=np.int32)
>>> b = np.linspace(0, pi, 3)
>>> b.dtype.name
'float64'
>>> c = a + b
>>> c
array([1.        , 2.57079633, 4.14159265])
>>> c.dtype.name
'float64'
>>> d = np.exp(c * 1j)
>>> d
array([ 0.54030231+0.84147098j, -0.84147098+0.54030231j,
       -0.54030231-0.84147098j])
>>> d.dtype.name
'complex128'
```

Many unary operations, such as computing the sum of all the elements in the array, are implemented as methods of the `ndarray` class.

```python
>>> a = rg.random((2, 3))
>>> a
array([[0.82770259, 0.40919914, 0.54959369],
       [0.02755911, 0.75351311, 0.53814331]])
>>> a.sum()
3.1057109529998157
>>> a.min()
0.027559113243068367
>>> a.max()
0.8277025938204418
```

By default, these operations apply to the array as though it were a list of numbers, regardless of its shape. However, by specifying the `axis` parameter you can apply an operation along the specified axis of an array:

```python
>>> b = np.arange(12).reshape(3, 4)
>>> b
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>>
>>> b.sum(axis=0) # sum of each column
array([12, 15, 18, 21])
>>>
>>> b.min(axis=1) # min of each row
array([0, 4, 8])
>>>
>>> b.cumsum(axis=1) # cumulative sum along each row
array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38]], dtype=int32)
```


### Universal Functions

NumPy provides familiar mathematical functions such as sin, cos, and exp. In NumPy, these are called "universal functions" (`ufunc`). Within NumPy, thses functions operate elementwise on an array, producing an array as output.

```python
>>> B = np.arange(3)
>>> B
array([0, 1, 2])
>>> np.exp(B)
array([1.        , 2.71828183, 7.3890561 ])
>>> np.sqrt(B)
array([0.        , 1.        , 1.41421356])
>>> C = np.array([2., -1., 4.])
>>> np.add(B, C)
array([2., 0., 6.])
```

**See aslo**
[`all`](https://numpy.org/doc/stable/reference/generated/numpy.all.html#numpy.all "numpy.all"), [`any`](https://numpy.org/doc/stable/reference/generated/numpy.any.html#numpy.any "numpy.any"), [`apply_along_axis`](https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html#numpy.apply_along_axis "numpy.apply_along_axis"), [`argmax`](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html#numpy.argmax "numpy.argmax"), [`argmin`](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html#numpy.argmin "numpy.argmin"), [`argsort`](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html#numpy.argsort "numpy.argsort"), [`average`](https://numpy.org/doc/stable/reference/generated/numpy.average.html#numpy.average "numpy.average"), [`bincount`](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html#numpy.bincount "numpy.bincount"), [`ceil`](https://numpy.org/doc/stable/reference/generated/numpy.ceil.html#numpy.ceil "numpy.ceil"), [`clip`](https://numpy.org/doc/stable/reference/generated/numpy.clip.html#numpy.clip "numpy.clip"), [`conj`](https://numpy.org/doc/stable/reference/generated/numpy.conj.html#numpy.conj "numpy.conj"), [`corrcoef`](https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html#numpy.corrcoef "numpy.corrcoef"), [`cov`](https://numpy.org/doc/stable/reference/generated/numpy.cov.html#numpy.cov "numpy.cov"), [`cross`](https://numpy.org/doc/stable/reference/generated/numpy.cross.html#numpy.cross "numpy.cross"), [`cumprod`](https://numpy.org/doc/stable/reference/generated/numpy.cumprod.html#numpy.cumprod "numpy.cumprod"), [`cumsum`](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html#numpy.cumsum "numpy.cumsum"), [`diff`](https://numpy.org/doc/stable/reference/generated/numpy.diff.html#numpy.diff "numpy.diff"), [`dot`](https://numpy.org/doc/stable/reference/generated/numpy.dot.html#numpy.dot "numpy.dot"), [`floor`](https://numpy.org/doc/stable/reference/generated/numpy.floor.html#numpy.floor "numpy.floor"), [`inner`](https://numpy.org/doc/stable/reference/generated/numpy.inner.html#numpy.inner "numpy.inner"), [`invert`](https://numpy.org/doc/stable/reference/generated/numpy.invert.html#numpy.invert "numpy.invert"), [`lexsort`](https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html#numpy.lexsort "numpy.lexsort"), [`max`](https://numpy.org/doc/stable/reference/generated/numpy.max.html#numpy.max "numpy.max"), [`maximum`](https://numpy.org/doc/stable/reference/generated/numpy.maximum.html#numpy.maximum "numpy.maximum"), [`mean`](https://numpy.org/doc/stable/reference/generated/numpy.mean.html#numpy.mean "numpy.mean"), [`median`](https://numpy.org/doc/stable/reference/generated/numpy.median.html#numpy.median "numpy.median"), [`min`](https://numpy.org/doc/stable/reference/generated/numpy.min.html#numpy.min "numpy.min"), [`minimum`](https://numpy.org/doc/stable/reference/generated/numpy.minimum.html#numpy.minimum "numpy.minimum"), [`nonzero`](https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html#numpy.nonzero "numpy.nonzero"), [`outer`](https://numpy.org/doc/stable/reference/generated/numpy.outer.html#numpy.outer "numpy.outer"), [`prod`](https://numpy.org/doc/stable/reference/generated/numpy.prod.html#numpy.prod "numpy.prod"), [`re`](https://docs.python.org/3/library/re.html#module-re "(in Python v3.11)"), [`round`](https://numpy.org/doc/stable/reference/generated/numpy.round.html#numpy.round "numpy.round"), [`sort`](https://numpy.org/doc/stable/reference/generated/numpy.sort.html#numpy.sort "numpy.sort"), [`std`](https://numpy.org/doc/stable/reference/generated/numpy.std.html#numpy.std "numpy.std"), [`sum`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html#numpy.sum "numpy.sum"), [`trace`](https://numpy.org/doc/stable/reference/generated/numpy.trace.html#numpy.trace "numpy.trace"), [`transpose`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html#numpy.transpose "numpy.transpose"), [`var`](https://numpy.org/doc/stable/reference/generated/numpy.var.html#numpy.var "numpy.var"), [`vdot`](https://numpy.org/doc/stable/reference/generated/numpy.vdot.html#numpy.vdot "numpy.vdot"), [`vectorize`](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html#numpy.vectorize "numpy.vectorize"), [`where`](https://numpy.org/doc/stable/reference/generated/numpy.where.html#numpy.where "numpy.where")

### Indexing, Slicing and Iterating

**One-dimensional** arrays can be indexed, sliced and iterated over, much like [lists](https://docs.python.org/tutorial/introduction.html#lists) and other Python sequences.

```python
>>> a = np.arange(10)**3
>>> a
array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729], dtype=int32)
>>> a[2]
8
>>> a[2:5]
array([ 8, 27, 64], dtype=int32)
>>> # equivalent to a[0:6:2] = 1000;
>>> # from start to position 6, exclusive, set every 2nd element to 1000
>>> a[:6:2] = 1000
>>> a
array([1000,    1, 1000,   27, 1000,  125,  216,  343,  512,  729],
      dtype=int32)
>>> a[::-1] # reversed a
array([ 729,  512,  343,  216,  125, 1000,   27, 1000,    1, 1000],
      dtype=int32)
>>> for i in a:
...     print(i**(1 / 3.))
...
9.999999999999998
1.0
9.999999999999998
3.0
9.999999999999998
5.0
5.999999999999999
6.999999999999999
7.999999999999999
8.999999999999998
```

**Multidimensional** arrays can have one index per axis.







































## Shape Manipulation