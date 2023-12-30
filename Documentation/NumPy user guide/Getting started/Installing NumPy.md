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

Also when using pip, it's good practice to use a virtual environment - see [Reproducible installs] below for why, and [this guide] for details on using virtual environments.

# Python and NumPy installation guide

Installing and managing packages in Python is complicated, there are a number of alternative solutions for most tasks. This guide tries to give the reader a sense of the best (or most popular) solutions, and give clear recommendations. It focuses on users of Python, NumPy, and the PyData (or numerical computing) stack on common operating systems and hardware.

## Recommendations 

We'll start with recommendations based on the user's experience level and operating system of interest. If you're in between "beginning" and "advanced", please go with "beginning" if you want to keep things simple, and with "advanced" if you want to work according to best practices that go a longer way in the future.

### Beginning users 

On all of Windows, macOS, and Linux:

- Install [Anaconda](https://www.anaconda.com/)(it installs all packages you need and all other tools mentioned below).
- For writing and executing code, use notebooks in [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/index.html) for exploratory and interative computing, and [Spyder](https://www.spyder-ide.org/) or [Visual Studio Code](https://code.visualstudio.com/) for writing scripts and packages.
- Use [Anaconda Navigator](https://docs.anaconda.com/free/navigator/) to manage your packages and start JupyterLab, Spyder, or Visual Studio Code.

### Advanced users

#### Conda

- Install [Miniforge](https://github.com/conda-forge/miniforge).
- Keep the `base` conda enviroment minimal, and use one or more [conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to install the package you need for the task or project you're working on.

#### Alternative if you prefer pip/PyPI

For users who know, from personal preference or reading about the main differences between conda and pip below, they prefer a pip/PyPI-based solution, we recommend:

- Install Python from [python.org](python.org), [Homebrew](https://brew.sh/), or your Linux package manager.
- Use [Poetry](https://python-poetry.org/) as the most well-maintained tool that provides a dependency resolver and environment management capabilities in a similar fashion as conda does.

## Python package management 

Managing packages is a challenging problem, and, as a result, there are lots of tools. For web and general purpose Python development there's a whole [host of tools](https://packaging.python.org/en/latest/guides/tool-recommendations/) complementary with pip. For high-performance computing (HPC), [Spack](https://github.com/spack/spack) is worth considering. For most NumPy users though, [conda](https://conda.io/en/latest/) and [pip](https://pip.pypa.io/en/stable/) are the two most popular tools.

### Pip & conda

The two main tools that install Python packages are `pip` and `conda`. Their functionality par