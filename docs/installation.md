
Scientific Machine Learning Benchmark (smlb)
# Installation

## Quick-start 

Clone the repository, locally install `smlb` with dependencies and run unit tests to verify a working installation:

```
git clone git@github.com:CitrineInformatics/smlb.git
make develop
make tests
```

This will download the `smlb` code repository, install a local developer version of the package and test its functionality for correctness.

## Optional steps

### Running notebooks

To run the interactive notebooks provided with `smlb`, such as the tutorial, install [Jupyter](https://jupyter.org/), either via `pip install jupyterlab` or via Anaconda (see below). Then run `jupyter notebook` or `jupyter lab` (new interface) in the directory containing the notebook.

### Virtual environment

Creating a virtual environment helps in developing and testing with different versions of packages. A popular tool is [Anaconda](https://www.anaconda.com/); for this, run below before installing `smlb`:

```
conda create --name <environment name> python=3.8 numpy scipy pandas jupyterlab scikit-learn
conda install -c conda-forge matplotlib
conda activate <environment name>
```

### Dependencies

`smlb` aims to support benchmarking code from different sources, for example, from the general-purpose machine-learning package [scikit-learn](https://scikit-learn.org/), specialized libraries like the [lolo random forest library](https://github.com/CitrineInformatics/lolo), or features from the cheminformatics [Chemistry Development Kit](https://github.com/cdk/cdk).

Consequently, `smlb` has many dependencies. To not have to always install all of them, which can result in large installations and version conflicts, `smlb` lets you choose which dependencies to install by providing different requirements files in directory `requirements`.
