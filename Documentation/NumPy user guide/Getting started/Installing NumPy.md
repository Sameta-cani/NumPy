# Installing NumPy

The only prerequisite for installing NumPy is Python itself. If you don't have Python yet and want the simplest way to get started, we recommend you use the [Anaconda Distribution](https://www.anaconda.com/data-science-platform) - it includes Python, NumPy, and many other commonly used packages for scientific computing and data science.

NumPy can be installed with `conda`, with `pip`, with a package manager on macOS and Linux, or [from source]. For more detained instructions, consult our [Python and NumPy installation guide] below.

## CONDA

If you use `conda`, you can install NumPy from the `defaults` or `conda-forge` channels:
```bash
# Best pracitce, use an environment rather than install in the base env
conda create -n my-env
conda activate my-env
# If you want to install from conda-forge
conda config --env --add channels conda-forge
# The actual install command
conda install numpy
```

## PIP

If you use `pip`, you can install NumPy with:

```bash
pip install numpy
```

