import unittest
from itertools import product

import numpy as np
import pandas as pd

from significance_analysis import (
    benchmark_information_check,
    fidelity_check,
    seed_dependency_check,
)


class TestSanityChecks(unittest.TestCase):
    def __init__(self, methodName: str = "sanityTest") -> None:
        super().__init__(methodName)

        # Define the parameters for the test data
        algos = ["A-1", "A-2", "A-3"]
        seeds = [str(x) for x in range(50)]
        benchmarks = ["B-0", "B-1", "B-2"]
        budgets = range(1, 16)

        # Create all combinations of the parameters
        combination = list(product(algos, benchmarks, [0], seeds, budgets))

        # Create a DataFrame with random values
        self.random_df = pd.DataFrame(
            combination, columns=["algorithm", "benchmark", "value", "seed", "budget"]
        )
        self.random_df["value"] = self.random_df.apply(
            lambda x: np.random.normal(0, 0.01), axis=1
        )

        # Create a DataFrame with specific values for seed dependency check
        self.seed_df = pd.DataFrame(
            combination, columns=["algorithm", "benchmark", "value", "seed", "budget"]
        )
        self.seed_df["value"] = self.seed_df.apply(
            lambda row: np.random.normal(
                int(row["seed"]) * 0.1
                if row["algorithm"][2] in ["1"]
                and (int(row["seed"]) % 5 == 0 and row["seed"] != "0")
                else 2.5,
                0.1 if row["algorithm"][2] in ["1"] else 0.55,
            ),
            axis=1,
        )

        # Create a DataFrame with specific values for benchmark information check
        self.benchmark_df = pd.DataFrame(
            combination, columns=["algorithm", "benchmark", "value", "seed", "budget"]
        )
        self.benchmark_df["value"] = self.benchmark_df.apply(
            lambda row: np.random.normal(
                0.2 * int(row["algorithm"][2]) * int(row["benchmark"][2]), 0.01
            ),
            axis=1,
        )

        # Create a DataFrame with specific values for fidelity check
        self.budget_df = pd.DataFrame(
            combination, columns=["algorithm", "benchmark", "value", "seed", "budget"]
        )
        self.budget_df["value"] = self.budget_df.apply(
            lambda row: np.random.normal(
                0.1 * np.random.normal(int(row["budget"]) - 5, 6, 1)[0], 0.18
            ),
            axis=1,
        )

    def test_random_seed(self):
        """
        Test that it correctly identifies the random seed as factor for algorithm A-1
        """
        result = seed_dependency_check(self.random_df, verbose=False)
        self.assertEqual(result, [])
        result = seed_dependency_check(self.seed_df, verbose=False)
        self.assertEqual(result, ["A-1"])

    def test_benchmark(self):
        """
        Test that it correctly identifies the uninteresting benchmark
        """
        result = benchmark_information_check(self.random_df, verbose=False)
        self.assertEqual(result, {"B-0": False, "B-1": False, "B-2": False})
        result = benchmark_information_check(self.benchmark_df, verbose=False)
        self.assertEqual(result, {"B-0": False, "B-1": True, "B-2": True})

    def test_budget(self):
        """
        Test that it correctly identifies the budget as factor
        """
        result = fidelity_check(fidelity_var="budget", data=self.random_df, verbose=False)
        self.assertEqual(result, "none")
        result = fidelity_check(fidelity_var="budget", data=self.budget_df, verbose=False)
        self.assertEqual(result, "simple_effect")


if __name__ == "__main__":
    unittest.main()
