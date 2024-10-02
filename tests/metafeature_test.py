import unittest

from significance_analysis import metafeature_analysis
import pandas as pd
import numpy as np
from itertools import product

class TestMetafeatureAnalysis(unittest.TestCase):
    def __init__(self, methodName: str = "sanityTest") -> None:
        super().__init__(methodName)
        algos=["A-1","A-2"]
        seeds=[str(x) for x in range(50)]
        benchmarks=["B-0","B-1"]
        budgets=range(1,11)
        metafeatures=["M-1","M-2","M-3"]
        combination = list(product(algos, benchmarks, [0],seeds,budgets,metafeatures))

        self.random_df = pd.DataFrame(combination, columns=["algorithm", "benchmark", "value", "seed","budget","metafeature"])
        self.random_df["value"] = self.random_df.apply(lambda x: np.random.normal(0, 0.05),axis=1)

        self.metafeature_df = pd.DataFrame(combination, columns=["algorithm", "benchmark", "value", "seed","budget","metafeature"])
        self.metafeature_df["value"] = self.metafeature_df.apply(lambda row:np.random.normal(0.5 if int(row["metafeature"][2])==int(row["algorithm"][2]) else 0.0, 0.1), axis=1)

    def test_metafeature_analysis(self):
        """
        Test that it correctly identifies differences between metafeatures
        """
        results=metafeature_analysis(self.random_df,("A-1","A-2"),metafeature_var="metafeature",fidelity_var="budget",verbose=False)
        significances = results[1]["Sig"].values.tolist()
        self.assertListEqual(significances, [""]*len(significances))
        results=metafeature_analysis(self.metafeature_df,("A-1","A-2"),metafeature_var="metafeature",fidelity_var="budget",verbose=False)
        significances = results[1]["Sig"].values.tolist()
        self.assertListEqual(significances, ['', '***', '', '', '***', '***', '', '', '***', '***', '***',  '', '', '***','***'])
        