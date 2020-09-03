![tests](https://travis-ci.com/CitrineInformatics/smlb.svg?token=scgF4QBbt4EG8U38VtzZ&branch=master)
<!-- [![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black) -->

# Scientific Machine Learning Benchmark (smlb)

## Introduction

`smlb` is a toolbox focused on enabling rigorous empirical assessments of data-driven modeling approaches for applications in the natural sciences.
It is particularly useful when developing or fine-tuning data-driven algorithms to ensure statistically sound decisions.
Its focus is on models for experimental and computed properties of molecules and materials.
It emphasizes correctness, flexibility, and domain support.

`smlb` was designed to help answer questions that arise during the development of domain-specific machine-learning models.
Examples of such questions include

* Which of these uncertainty estimate approaches most closely matches the true error distribution? 
* How does removing slow or failing features affect the predictive accuracy of my model? 

`smlb` provides

* ready-to-use synthetic, computational and experimental datasets
* bindings to other software, including domain-specific features and general machine-learning packages
* standard loss functions and error metrics, also for predictive distributions (uncertainties)
* reproducibility  by systematic control of pseudo-random number generation

Other uses include integration tests to ensure that local changes to a modeling pipeline do not have overall adverse effects.

See the [Overview](docs/overview.md) for a more detailed description.

## Getting started

To get started, follow [installation](docs/installation.md) instructions and run the [tutorial](docs/tutorial.ipynb).

## Other

To contribute, see the [Contributing](docs/contributing.md) instructions.

[Related work](docs/related.md)<br>
[Acknowledgments](docs/acknowledgments.md)
