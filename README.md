# Significance Analysis

[![PyPI version](https://img.shields.io/pypi/v/significance-analysis?color=informational)](https://pypi.org/project/significance-analysis/)
[![Python versions](https://img.shields.io/pypi/pyversions/significance-analysis)](https://pypi.org/project/significance-analysis/)
[![License](https://img.shields.io/pypi/l/significance-analysis?color=informational)](LICENSE)

This package is used to analyse datasets of different HPO-algorithms performing on multiple benchmarks.

## Note

As indicated with the `v0.x.x` version number, Significance Analysis is early stage code and APIs might change in the future.

## Documentation

Please have a look at our [example](significance_analysis_example/example_analysis.py).
The dataset should have the following format:

| system_id<br>(algorithm name) | input_id<br>(benchmark name) | metric<br>(mean/estimate) | optional: bin_id<br>(budget/traininground) |
| ----------------------------- | ---------------------------- | ------------------------- | ------------------------------------------ |
| Algorithm1                    | Benchmark1                   | x.xxx                     | 1                                          |
| Algorithm1                    | Benchmark1                   | x.xxx                     | 2                                          |
| Algorithm1                    | Benchmark2                   | x.xxx                     | 1                                          |
| ...                           | ...                          | ...                       | ...                                        |
| Algorithm2                    | Benchmark2                   | x..xxx                    | 2                                          |

In this dataset, there are two different algorithms, trained on two benchmarks for two iterations each. The variable-names (system_id, input_id...) can be customized, but have to be consistent throughout the dataset, i.e. not "mean" for one benchmark and "estimate" for another. The `conduct_analysis` function is then called with the dataset and the variable-names as parameters.
Optionally the dataset can be binned according to a fourth variable (bin_id) and the analysis is conducted on each of the bins seperately, as shown in the code example above. To do this, provide the name of the bin_id-variable and if wanted the exact bins and bin labels. Otherwise a bin for each unique value will be created.

## Installation

Using R, >=4.0.0
install packages: Matrix, emmeans, lmerTest and lme4

Using pip

```bash
pip install significance-analysis
```

## Usage

1. Generate data from HPO-algorithms on benchmarks, saving data according to our format.
1. Call function `conduct_analysis` on dataset, while specifying variable-names

In code, the usage pattern can look like this:

```python
import pandas as pd
from signficance_analysis import conduct_analysis

# 1. Generate/import dataset
data = pd.read_csv("./significance_analysis_example/exampleDataset.csv")

# 2. Analyse dataset
conduct_analysis(data, "mean", "acquisition", "benchmark")
```

For more details and features please have a look at our [example](significance_analysis_example/example_analysis.py).
