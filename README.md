# Significance Analysis

[![PyPI version](https://img.shields.io/pypi/v/significance-analysis?color=informational)](https://pypi.org/project/significance-analysis/)
[![Python versions](https://img.shields.io/pypi/pyversions/significance-analysis)](https://pypi.org/project/significance-analysis/)
[![License](https://img.shields.io/pypi/l/significance-analysis?color=informational)](LICENSE)
[![Coverage Status](./tests/coverage-badge.svg?dummy=8484744)](./tests/reports/cov_html/index.html)
[![Tests Status](./tests/tests-badge.svg?dummy=8484744)](./tests/reports/junit/report.html)
[![arXiv](https://img.shields.io/badge/arXiv-2408.02533-b31b1b.svg)](https://arxiv.org/abs/2408.02533)

This package is used to analyse datasets of different HPO-algorithms performing on multiple benchmarks, using a Linear Mixed-Effects Model-based approach.

## Note

As indicated with the `v0.x.x` version number, Significance Analysis is early stage code and APIs might change in the future.

## Documentation

For an interactive overview, please have a look at our [example](significance_analysis_example/analysis_example.ipynb).

Every dataset should be a pandas dataframe of the following format:

| algorithm  | benchmark  | metric | optional: budget/prior/... |
| ---------- | ---------- | ------ | -------------------------- |
| Algorithm1 | Benchmark1 | 3.141  | 1.0                        |
| Algorithm1 | Benchmark1 | 6.283  | 2.0                        |
| Algorithm1 | Benchmark2 | 2.718  | 1.0                        |
| ...        | ...        | ...    | ...                        |
| Algorithm2 | Benchmark2 | 0.621  | 2.0                        |

As it is used to train a model, there can not be missing values, but duplicates are allowed.
Our function `dataset_validator` checks for this format.

## Installation

Using R, >=4.0.0
install packages: Matrix, emmeans, lmerTest and lme4

Using pip

```bash
pip install significance-analysis
```

## Usage for significance testing

1. Generate data from HPO-algorithms on benchmarks, saving data according to our format.
1. Build a model with all interesting factors
1. Do post-hoc testing
1. Plot the results as CD-diagram

In code, the usage pattern can look like this:

```python
import pandas as pd
from significance_analysis import dataframe_validator, model, cd_diagram


# 1. Generate/import dataset
data = dataframe_validator(pd.read_parquet("datasets/priorband_data.parquet"))

# 2. Build the model
mod = model("value ~ algorithm + (1|benchmark) + prior", data)

# 3. Conduct the post-hoc analysis
post_hoc_results = mod.post_hoc("algorithm")

# 4. Plot the results
cd_diagram(post_hoc_results)
```

## Usage for hypothesis testing

Use the GLRT implementation or our prepared `sanity checks` to conduct LMEM-based hypothesis testing.

In code:

```python
from significance_analysis import (
    dataframe_validator,
    glrt,
    model,
    seed_dependency_check,
    benchmark_information_check,
    fidelity_check,
)

# 1. Generate/import dataset
data = dataframe_validator(pd.read_parquet("datasets/priorband_data.parquet"))

# 2. Run the preconfigured sanity checks
seed_dependency_check(data)
benchmark_information_check(data)
fidelity_check(data)

# 3. Run a custom hypothesis test, comparing model_1 and model_2
model_1 = model("value ~ algorithm", data)
model_2 = model("value ~ 1", data)
glrt(model_1, model_2)
```

## Usage for metafeature impact analysis

Analyzing the influence, a metafeature has on two algorithms performances.

In code:

```python
from significance_analysis import dataframe_validator, metafeature_analysis

# 1. Generate/import dataset
data = dataframe_validator(pd.read_parquet("datasets/priorband_data.parquet"))

# 2. Run the metafeature analysis
scores = metafeature_analysis(data, ("HB", "PB"), "prior")
```

For more details and features please have a look at our [example](significance_analysis_example/analysis_example.py).

## Contributing

We welcome contributions from everyone, feel free to raise issues or submit pull requests.

### To cite the paper or code

```bibtex
@misc{geburek2024lmemsposthocanalysishpo,
title={LMEMs for post-hoc analysis of HPO Benchmarking},
author={Anton Geburek and Neeratyoy Mallik and Danny Stoll and Xavier Bouthillier and Frank Hutter},
year={2024},
eprint={2408.02533},
archivePrefix={arXiv},
primaryClass={cs.LG}
}
```
