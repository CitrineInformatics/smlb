
Scientific Machine Learning Benchmark (smlb)
# Development

## Code style

`smlb` uses [black](https://github.com/psf/black) with `--line-length 99` as code formatter in conjunction with reasonable [PEP8](https://www.python.org/dev/peps/pep-0008/) adherence.

Both `black` and `PEP8` are supported by all major code editors and integrated development environments.

## Version control of Jupyter notebooks

[Jupyter](https://jupyter.org/) notebooks contain information not suited for version control by [git](https://git-scm.com/), for instance, cell execution counts and binary blobs such as images. The `smlb` repository excludes this information from version control.

### Manual approach

Before adding changes in a notebook, please 
1. open the notebook
2. shutdown its kernel
3. clear all outputs
4. save the notebook

### Automatic approach

Filter via [nbstripout](https://pypi.org/project/nbstripout).

For this, install `nbstripout`, either via [pip](https://pypi.org/project/pip/) as
```
pip install --upgrade nbstripout
```
or via [Anaconda](https://www.anaconda.com/) as
```
conda install -c conda-forge nbstripout
```
then from within the `smlb` repository directory run
```
nbstripout --install
```
`git` will now essentially only control for changes in the input cells of notebooks.
