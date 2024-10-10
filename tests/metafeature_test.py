import unittest
from itertools import product

import numpy as np
import pandas as pd

from significance_analysis import metafeature_analysis


class TestMetafeatureAnalysis(unittest.TestCase):
    def __init__(self, methodName: str = "sanityTest") -> None:
        super().__init__(methodName)

        # Define the parameters for the test data
        algos = ["A-1", "A-2"]
        seeds = [str(x) for x in range(20)]
        benchmarks = ["B-0", "B-1"]
        budgets = range(1, 11)
        metafeatures = ["M-1", "M-2", "M-3"]

        # Create all combinations of the parameters
        combination = list(product(algos, benchmarks, [0], seeds, budgets, metafeatures))

        # Create a DataFrame with random values
        self.random_df = pd.DataFrame(
            combination,
            columns=["algorithm", "benchmark", "value", "seed", "budget", "metafeature"],
        )
        self.random_df["value"] = self.random_df.apply(
            lambda x: np.random.normal(0, 0.05), axis=1
        )

        # Create a DataFrame with specific values for metafeature analysis
        self.metafeature_df = pd.DataFrame(
            combination,
            columns=["algorithm", "benchmark", "value", "seed", "budget", "metafeature"],
        )
        self.metafeature_df["value"] = self.metafeature_df.apply(
            lambda row: np.random.normal(
                0.5 if int(row["metafeature"][2]) == int(row["algorithm"][2]) else 0.0,
                0.1,
            ),
            axis=1,
        )

    def test_metafeature_analysis(self):
        """
        Test that it correctly identifies differences between metafeatures
        """
        # Perform metafeature analysis on the random DataFrame
        results = metafeature_analysis(
            self.random_df,
            ("A-1", "A-2"),
            metafeature_var="metafeature",
            fidelity_var="budget",
            verbose=False,
        )
        significances = results[1]["Sig"].values.tolist()
        significances = ["" if x == "." else x for x in significances]
        self.assertListEqual(significances, [""] * len(significances))

        # Perform metafeature analysis on the specific DataFrame
        results = metafeature_analysis(
            self.metafeature_df,
            ("A-1", "A-2"),
            metafeature_var="metafeature",
            fidelity_var="budget",
            verbose=False,
        )
        significances = results[1]["Sig"].values.tolist()
        significances = ["" if x == "." else x for x in significances]
        self.assertListEqual(
            significances,
            [
                "",
                "***",
                "",
                "",
                "***",
                "***",
                "",
                "",
                "***",
                "***",
                "***",
                "",
                "",
                "***",
                "***",
            ],
        )
