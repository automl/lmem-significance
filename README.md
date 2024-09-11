# Significance Analysis

[![PyPI version](https://img.shields.io/pypi/v/significance-analysis?color=informational)](https://pypi.org/project/significance-analysis/)
[![Python versions](https://img.shields.io/pypi/pyversions/significance-analysis)](https://pypi.org/project/significance-analysis/)
[![License](https://img.shields.io/pypi/l/significance-analysis?color=informational)](LICENSE)

This package is used to analyse datasets of different HPO-algorithms performing on multiple benchmarks, using a Linear Mixed-Effects Model-based approach.

## Note

As indicated with the `v0.x.x` version number, Significance Analysis is early stage code and APIs might change in the future.

## Documentation

Please have a look at our [example](significance_analysis_example/analysis_example.ipynb).
The dataset should be a pandas dataframe of the following format:

| algorithm  | benchmark  | metric | optional: budget/prior/... |
| ---------- | ---------- | ------ | -------------------------- |
| Algorithm1 | Benchmark1 | x.xxx  | 1.0                        |
| Algorithm1 | Benchmark1 | x.xxx  | 2.0                        |
| Algorithm1 | Benchmark2 | x.xxx  | 1.0                        |
| ...        | ...        | ...    | ...                        |
| Algorithm2 | Benchmark2 | x.xxx  | 2.0                        |

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
