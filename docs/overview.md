
Scientific Machine Learning Benchmark (smlb)
# Overview

`smlb` is a (simple) scientific machine learning benchmarking package.<br>
Its focus is on models for experimental and computed properties of molecules and materials.<br>
Priorities are correctness, flexibility and domain support.

<table>
    <tr><td>Audience:</td><td>(data) scientists</td></tr>
    <tr><td style="vertical-align: top;">Supports:</td><td>
        10 to 100k data points (small datasets)<br>
        features for molecules and materials<br>
        predictive uncertainty estimation<br>
        programmatic and configuration-file control<br>
        reproducibility of results<br>
        benchmark sampling, features and learners<br>
        structured inputs and output<br>
    </td></tr>
    <tr><td style="vertical-align: top;">What it is not:</td><td>
        maximally general or performant<br>
        backwards compatible between versions<br>
    </td></tr>
</table>

## Vocabulary

* `Data` is any source of data for learning ("dataset")
* A `Sampler` selects subset(s) of `Data`
* A `Featurizer` computes features for `Data`
* A `Learner` is a function fitted to (training) `Data` and applied to (validation) `Data`
* A `Noise` model adds noise to `Data`
* A `DataTransformation` is a function accepting `Data`<br>`Sampler`, `Featurizer`, `Noise` and `Learner`  are `DataValuedTransformation`s
* A `Workflow` is a directed graph of `DataTransformation`s with a single sink
* A `Result` is the outcome of a `Workflow`
* An `Outcome` is a textual or graphical summary of `Result`s
* An `Evaluator` creates `Outcomes` from `Result`s
* An `Experiment` completely specifies `Data` (inputs), `Workflow`s and `Evaluator`s (output)

## Design principles

* Workflows are composed of data transformations
* Control is programmatic or via configuration files
* Simplicity:<br>
  Minimal class and function interfaces<br>
  Configuration files specify instantiation (and execution) of Python classes<br>
  Parallelism via filesystem ("poor man's distributed computing")
* Flexible enough for non-standard learning workflows

## Caveat emptor

* `Data` with `Noise` behaves like actual random variables.<br>
  Every time `samples()` or `labels()` with Noise are queried, their values can change.
  Think of noisy samples or labels as random variables, each query being a draw.

