# NumPy: the absolute basics for beginners

Welcome to the absolute beginner's guide to NumPy! If you have comments or suggestions, please don't hesitate to  [reach out](https://numpy.org/community/)!

## Welcome to NumPy!

NumPy (**Numerical Python**) is an open source Python library that's used in almost every field of science and enginerring. It's the universal standard for working with numerical data in Pyhton, and it's at the core of the scientific Python and PyData ecosystems. NumPy users include everyone from beginning coders to experienced researchers doing state-of-the-art scientific and industrial research and development. The NumPy API is used extensively in Pandas, SciPy, Matplotlib, scikit-learn, scikit-image and most other data science and scientific Python pacakges.

The NumPy library contains multidimensional array and matrix data structures (you'll find more information about this in later sections). It provides **ndarray**, a homogeneous n-dimensional array object, with methods to efficiently operate on it. NumPy can be used to perform a wide variety of mathematical operations on arrays. It adds powerful data structures to Python that guarantee efficient calculations with arrays and matrices and it supplies an enormous library of high-level mathematical functions that operate on these arrays and matrices.

Learn more about [[What is NumPy|NumPy here]]!

## Installing NumPy

To install NumPy, we strongly recommend using a scientific Python distribution. If you're looking for the full instructioins for installing NumPy on your operating system, see [[Installing NumPy]].

If you already have Python, you can install NumPy with:

```bash
conda install numpy
```

or

```bash
pip install numpy
```

If you don't have Python yet, you might want to consider using [Anaconda](https://www.anaconda.com/). It's the easiest way to get started. The good thing about getting this distribution is the fact that you don't need to worry too much about separately installing NumPy or any of the major pacakges that you'll be using for your data analyses, like pandas, Scikit-Learn, etc.

## How to import NumPy

To access NumPy and its functions import it in your Python code like this:

```python
import numpy as np
```

We shorten the imported name to `np` for better readability of code using NumPy. This is a widely adopted convetion that makes your code more readable for everyone working on it. We recommend to always use import numpy as `np`.

## Reading the example code

If you aren't already comfortable with reading tutorials that contain a lot of code, you might not know how to interpret a code block that looks like this:

```python
>>> a = np.arange(6)
>>> a2 = a[np.newaxis, :]
>>> a2.shape
(1, 6)
```

If you aren't familiar with this style, it's very easy to understand. If you see `>>>`, you're looking at **input**, or the code that you would enter. Everything that doesn't have `>>>` in front of it is **output**, or the results of running your code. This is the style you see when you run `python` on the command line, but if you're using IPython, you might see a different style. Note that it is not part of the code and will cause an error if typed or pasted into the Python shell. It can be safely typed or pasted into the IPython shell; the `>>>` is ignored.

## What's the difference between a Python list and a NumPy array?

NumPy gives you an enormous range of fast and efficient ways of creating arrays and manipulating numerical data indide them. While a Python list can contain different data types within a single list, all of the elements in a NumPy array should be homogeneous. The mathematical operations that are meant to be performed on arrays would be extremely inefficient if the arrays weren't homogeneous.

### Why use NumPy?

NumPy arrays are faster and more compact than Python lists. An array consumes less memory and is convenient to use. NumPy uses much less memory to store data and it provides a mechanism of specifying the data types. This allows the code to be optimized even further.

## What is an array?

An array is a central data structure of the NumPy library. An array is a grid of values and it contains information about the raw data, how to locate an element, and how to interpret an element. It has a grid of elements that can be indexed in [[NumPy quickstart#Indexing, Slicing and Iterating|various ways]]. The elements are all of the same type, referred to as the array `dtype`.

An array can be indexed by a tuple of nonnegative integers, by booleans, by another array, or by integers. The `rank` of the array is the number of dimensions. The `shape` of the array is a tuple of integers giving the size of the array along each dimension.

One way we can initialize NumPy arrays is form Python lists, using nested lists for two- or higher-dimensional data.

For example:

```python
>>> a = np.array([1, 2, 3, 4, 5, 6])
```

or:

```python
>>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
```

We can access the elements in the array using square brackets. When you're accessing elements, remenber that indexing in NumPy starts at 0. That means that if you want to access the first element in your array, you'll be accessing element "0".

```python
>>> print(a[0])
[1 2 3 4]
```

## More information about arrays

*This section covers* `1D array`, `2D array`, `ndarray`, `vector`, `matrix`

You might occasionally hear an array referred to as a "ndarray", which is shorthand for "N-dimensioanl array". An N-dimensional array is simple an array with any number of dimensions. You might also hear **1-D**, or one-dimensional array, **2-D**, or two-dimensional array, and so on. The NumPy `ndarray` class is used to represent both matrices and vectors. A **vector** is an array with a single dimension (there's no difference between row and column vectors), while a **matrix** refers to an array with two dimensions. For **3-D** or higher dimensional arrays, the term **tensor** is also commonly used.

### What are the attributes of an array?

An array is usually a fixed-size container of items of the same type and size. The number of dimensions and items in an array is defined by its shape. The shape of an array is a tuple of non-negative integers that specify the sizes of each dimension.

In NumPy, dimensions are called **axes**. This means that if you have a 2D array that looks like this:

```python
[[0., 0., 0.],
 [1., 1., 1.]]
```

Your array has 2 axes. The first axis has a length of 2 and the second aixs has a length of 3.

Just like in other Python container objects, the contents of an array can be accessed and modified by indexing or slicing the array. Unlike the typical container objects, different arrays can share the same data, so changes made on one array might be visible in another.

Array **attributes** reflect information intrinsic to the array itself. If you need to get, or even set, properties of an array without creating a new array, you can often access an array through its attributes.

[Read more about array attributes here](https://numpy.org/doc/stable/reference/arrays.ndarray.html#arrays-ndarray) and learn about [array objects here](https://numpy.org/doc/stable/reference/arrays.html#arrays).

## How to create a basic array

*This section covers* `np.array()`, `np.zeros()`, `np.ones()`, `np.empty()`, `np.arange()`, `np.linspace()`, `dtype`

To create a NumPy array, you can use the function `np.array()`.

All you need to do to create a simple array is pass a list to it. If you choose to, you can also specify the type of data in your list. [You can find more information about data types here](https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes).

```python
>>> import numpy as np
>>> a = np.array([1, 2, 3])
```

You can visualize your array this way:

![np_array](../../../image/np_array.png)

*Be aware that these visualizations are meant to simplify ideas and give you a basic understanding of NumPy concepts and mechanics. Arrays and array operations are much more complicated than are caputred here!*

Besides creating an array from a sequence of elements, you can easily create an array filed with `0`'s:

```python
>>> np.zeros(2)
array([0., 0.])
```

Or an array filled with `1`'s:

```python
>>> np.ones(2)
array([1., 1.])
```

Or even an empty array! The function `empty` creates an array whose initial content is random and depends on the state of the memory. The reason to use `empty` over `zeros` (or something similar) is speed - just make sure to fill every element afterwards!

```python
>>> np.empty(2)
array([1., 1.])
```

You can create an array with a range of elements:

```python
>>> np.arange(4)
array([0, 1, 2, 3])
```

And even an array that contains a range of evenly spaced intervals. To do this, you will specify the **first number**, **last number**, and the **step size**.

```python
>>> np.arange(2, 9, 2)
array([2, 4, 6, 8])
```

You can also use `np.linspace()` to create an array with values that are spaced linearly in a specified interval:

```python
>>> np.linspace(0, 10, num=5)
array([ 0. ,  2.5,  5. ,  7.5, 10. ])
```

### Specifying your data type

While the default data type is floating point (`np.float64`), you can explicitly specify which data type you want using the `dtype` keyword.

```python
>>> x = np.ones(2, dtype=np.int64)
>>> x
array([1, 1], dtype=int64)
```

[[NumPy quickstart#Array Creation|Learn more about creating arrays here]]

## Adding, removing, and sorting elements

*This section covers* `np.sort()`, `np.concatenate()`

Sorting an element is simple with `np.sort()`. You can specify the axis. kind, and order when you call the function.

If you start with this array:

```python
>>> arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])
```

You can quickly sort the numbers in ascending order with:

```python
>>> np.sort(arr)
array([1, 2, 3, 4, 5, 6, 7, 8])
```

In addition to sort, which returns a sorted copy of an array, you can use:

- [`argsort`](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html#numpy.argsort "numpy.argsort"), which is an indirect sort along a specified axis,
- [`lexsort`](https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html#numpy.lexsort "numpy.lexsort"), which is an indirect stable sort on multiple keys,
- [`searchsorted`](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html#numpy.searchsorted "numpy.searchsorted"), which will find elements in a sorted array, and
- [`partition`](https://numpy.org/doc/stable/reference/generated/numpy.partition.html#numpy.partition "numpy.partition"), which is a partial sort.

To read more about sorting an array, see: [`sort`](https://numpy.org/doc/stable/reference/generated/numpy.sort.html#numpy.sort "numpy.sort").

If you start with these arrays:

```python
>>> a = np.array([1, 2, 3, 4])
>>> b = np.array([5, 6, 7, 8])
```

You can concatenate them with `np.concatenate()`.

```python
>>> np.concatenate((a, b))
array([1, 2, 3, 4, 5, 6, 7, 8])
```

Or, if you start with these arrays:

```python
>>> x = np.array([[1, 2], [3, 4]])
>>> y = np.array([[5, 6]])
```

You can concatenate them with:

```python
>>> np.concatenate((x, y), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])
```

In order to remove elements from an array, it's simple to use indexing to select the elements that you want to keep.

To read more about concatenate, see: [`concatenate`](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html#numpy.concatenate "numpy.concatenate").

## How do you know the shape and size of an array?

*This section covers* `ndarray.ndim`, `ndarray.size`, `ndarray.shape`

`ndarray.ndim` will tell you the number of axes, or dimensions, of the array.

`ndarray.size` will tell you the total number of elements of the array. This is the *product* of the elements of the array's shape.

`ndarray.shape` will display a tuple of integers that indicate the number of elements stored along each dimension of the array. If, for example, you have a 2-D array with 2 rows and 3 columns, the shape of your array is `(2, 3)`.

For example, if you create this array:

```python
>>> array_example = np.array([[[0, 1, 2, 3],
...                            [4, 5, 6, 7]],
...
...                           [[0, 1, 2, 3],
...                            [4, 5, 6, 7]],
...
...                           [[0, 1, 2, 3],
...                            [4, 5, 6, 7]]])
```

To find the number of dimensions of the array, run:

```python
>>> array_example.ndim
3
```

To find the total number of elements in the array, run:

```python
>>> array_example.size
24
```

And to find the shape of your array, run:

```python
>>> array_example.shape
(3, 2, 4)
```

## Can you reshape an array?

*This section covers* `arr.reshape()`

**Yes!**

Using `arr.reshape()` will give a new shape to an array without changing the data. Just remember that when you use the reshape method, the array you want to produce needs to have the same number of elements as the original arrya. If you start with an array with 12 elements, you'll need to make sure that your new array also has a total of 12 elements.

If you start with this array:

```python
>>> a = np.arange(6)
>>> print(a)
[0 1 2 3 4 5]
```

You can use `reshape()` to reshape your array. For example, you can reshape this array to an array with three rows and two columns:

```python
>>> b = a.reshape(3, 2)
>>> print(b)
[[0 1]
 [2 3]
 [4 5]]
```

With `np.reshape`, you can specify a few optional parameters:

```python
>>> np.reshape(a, newshape=(1, 6), order='C')
array([[0, 1, 2, 3, 4, 5]])
```

`a` is the array to be reshaped.

`newshape` is the new shape you want. You can specify an integer or a tuple of integers. If you specify an integer, the result will be an array of that length. The shape should be compatible with the original shape.

`order: C` means to read/write the elements using C-like index order, `F` means to read/write the elements using Fortran-like index order, `A` means to read/write the elements in Fortra-like index order if a is Fortran contiguous in memory, C-like order otherwise. (This is an optional parameter and doesn't need to be speficied.)

If you want to learn about C and Fortran order, you can [read more about the internal organization of NumPy arrays here](https://numpy.org/doc/stable/dev/internals.html#numpy-internals). Essentially, C and Fortran orders have to do with how indices correspond to the order the array is sorted in memory. In Fortran, when moving through the elements of a two-dimensioanl array as it is sorted in memory, the **first** index is the most rapidly varing index. As the first index moves to the next row as it changes, the matrix is sorted one column at a time. This is why Fortran is thought of as a **Column-major language**. In C on the other hand, the **last** index changes the most rapidly. The matrix is sorted by rows, making it a **Row-major language**. What you do for C or Fortran depends on whether it's more important to preserve the indexing convention or not reorder the data.

[[NumPy quickstart#Shape Manipulation|Learn more about shape manipulation here]].

