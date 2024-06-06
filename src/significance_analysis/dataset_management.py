import os
import typing

import numpy as np
import pandas as pd
from scipy.stats import rankdata

pd.set_option("chained_assignment", None)
pd.set_option("display.max_rows", 5000)
pd.set_option("display.max_columns", 5000)
pd.set_option("display.width", 10000)


def load_priorband_data():
    df = pd.read_parquet(
        "../significance_analysis_example/datasets/full_priorband_data.parquet"
    )
    print(df)
    df = df.reset_index()
    df_collection = []
    for seed_nr in range(50):
        partial_df = df[["benchmark", "prior", "algorithm", "used_fidelity"]]
        partial_df["value"] = df[f"seed-{seed_nr}"]
        partial_df["seed"] = seed_nr
        df_collection.append(partial_df)
        print(f"{f'⚙️ Seed {seed_nr+1}/50':<100}", end="\r", flush=True)
    complete_df = pd.concat(df_collection, ignore_index=True)
    print(f"{f'✅ Loading data done':<100}")
    return complete_df


def combine_bench_prior(row):
    return f"{row['benchmark']}_{row['prior']}"


def add_rel_ranks(row, data: pd.DataFrame, benchmark: str, time: str):
    return rankdata(
        data.loc[
            (data[benchmark] == row[benchmark])
            & (data["seed"] == row["seed"])
            & (data[time] == row[time])
        ]["value"].values
    )[
        data.loc[
            (data[benchmark] == row[benchmark])
            & (data["seed"] == row["seed"])
            & (data[time] == row[time])
        ]["value"]
        .values.tolist()
        .index(row["value"])
    ].astype(
        float
    )


def add_regrets(df: pd.DataFrame, benchmark_variable):
    best = {}
    ranges = {}
    # print(df.head(20000))
    print(f"{'⚙️ Preparing regret':<100}", end="\r", flush=True)
    for benchmark in df[benchmark_variable].unique():
        # print(benchmark)
        # print(df.loc[df[benchmark_variable] == benchmark])
        best[benchmark] = min(df.loc[df[benchmark_variable] == benchmark]["value"])
        ranges[benchmark] = (
            max(df.loc[df[benchmark_variable] == benchmark]["value"]) - best[benchmark]
        )

    def calculate_regrets(row):
        return pd.Series(
            [
                abs(best[row[benchmark_variable]] - row["value"]),
                abs(best[row[benchmark_variable]] - row["value"])
                / ranges[row[benchmark_variable]],
            ]
        )

    print(f"⚙️ {'Adding regret':<100}", end="\r", flush=True)
    df[["simple_regret", "normalized_regret"]] = df.apply(calculate_regrets, axis=1)
    print(f"{'✅ Adding regret done':<100}")

    return df


def rename_algos(row, algo_dict: dict):
    return algo_dict[row["algorithm"]]


def rename_benchmarks(row, bench_dict: dict):
    return bench_dict[row["benchmark"]]


def create_incumbent(
    data, f_space, benchmarks, algos, benchmark_variable, algorithm_variable
):
    algo_fidelities = {}
    for algo in algos:
        for bench_prior in data.loc[data[algorithm_variable] == algo][
            benchmark_variable
        ].unique():
            algo_fidelities[f"{algo}_{bench_prior}"] = np.array(f_space)[
                np.array(f_space)
                >= min(
                    data.loc[
                        (data[algorithm_variable] == algo)
                        & (data[benchmark_variable] == bench_prior)
                    ]["used_fidelity"]
                )
            ]
            algo_fidelities[f"{algo}_{bench_prior}"].sort()
            # print(algo,bench_prior,algo_fidelities[f"{algo}_{bench_prior}"][:20])
            # algo_fidelities[f"{algo}_{bench_prior}"].sort()

    # print(data.loc[(data[algorithm_variable]=="hyperband")&(data["benchmark"]=="LC-167190")])

    dataset = pd.DataFrame()
    for prior in data["prior"].unique():
        prior_df = data.loc[data["prior"] == prior]
        # print(prior)
        # print(len(prior_df))
        prior_dataset = pd.DataFrame()
        for seed in prior_df["seed"].unique():
            bench_ds = pd.DataFrame()
            for b_n, bench in enumerate(
                prior_df.loc[prior_df[benchmark_variable].isin(benchmarks)][
                    benchmark_variable
                ].unique()
            ):
                print(
                    f"{f'⚙️ Seed {seed+1}/{len(data.seed.unique())}, Benchmark {b_n+1}/{len(benchmarks)}':<100}",
                    end="\r",
                    flush=True,
                )
                seed_bench_df = prior_df.loc[prior_df["seed"] == seed].loc[
                    prior_df[benchmark_variable] == bench
                ]
                for algo in algos:
                    algo_df = seed_bench_df.loc[seed_bench_df[algorithm_variable] == algo]
                    # if algo=="hyperband":
                    #     print(algo_df[:20])
                    #     print(algo_df.set_index('used_fidelity')[:20])
                    #     print(algo_df.set_index('used_fidelity').reindex(algo_fidelities[f"{algo}_{bench}"])[:20])
                    #     print(algo_df.set_index('used_fidelity').reindex(algo_fidelities[f"{algo}_{bench}"]).fillna(np.NaN)[:20])
                    #     print(algo_df.set_index('used_fidelity').reindex(algo_fidelities[f"{algo}_{bench}"]).fillna(np.NaN).sort_values("used_fidelity")[:20])
                    algo_df = (
                        algo_df.set_index("used_fidelity")
                        .reindex(algo_fidelities[f"{algo}_{bench}"])
                        .fillna(np.NaN)
                        .sort_values("used_fidelity")
                        .reset_index()
                        .ffill()
                    )
                    # print(algo_df)
                    bench_ds = pd.concat([bench_ds, algo_df], ignore_index=True)
                # print(bench_ds["algorithm"].value_counts())
            # print(bench_ds.head(20))
            prior_dataset = pd.concat([prior_dataset, bench_ds], ignore_index=True)
        dataset = pd.concat([dataset, prior_dataset], ignore_index=True)

    return dataset


std_benchmarks = [
    "jahs_cifar10",
    "jahs_colorectal_histology",
    "jahs_fashion_mnist",
    "lcbench-126026",
    "lcbench-167190",
    "lcbench-168330",
    "lcbench-168910",
    "lcbench-189906",
    "cifar100_wideresnet_2048",
    "imagenet_resnet_512",
    "lm1b_transformer_2048",
    "translatewmt_xformer_64",
]

label_dict = {
    "random_search": "RS",
    "hyperband": "HB",
    "pb_mutation_dynamic_geometric-default-at-target": "PB",
    "pb_mutation_dynamic_geometric_bo-default-at-target": "PriorBand+BO",
    "jahs_cifar10": "JAHS-C10",
    "jahs_colorectal_histology": "JAHS-CH",
    "jahs_fashion_mnist": "JAHS-FM",
    "lcbench-126026": "LC-126026",
    "lcbench-167190": "LC-167190",
    "lcbench-168330": "LC-168330",
    "lcbench-168910": "LC-168910",
    "lcbench-189906": "LC-189906",
    "cifar100_wideresnet_2048": "PD1-Cifar100",
    "imagenet_resnet_512": "PD1-ImageNet",
    "lm1b_transformer_2048": "PD1-LM1B",
    "translatewmt_xformer_64": "PD1-WMT",
    "random_search_prior": "RS+Prior",
    "bo": "BO",
    "bo-10": "BO",
    "pibo-no-default": "PiBO",
    "pibo-default-first-10": "PiBO",
    "bohb": "BOHB",
    "priorband_bo": "PriorBand+BO",
}

figures = {}
figures["fig7"] = [
    "random_search_prior",
    "pb_mutation_dynamic_geometric_bo-default-at-target",
    "pibo-default-first-10",
    "bo-10",
    "bohb",
]
figures["fig5"] = [
    "pb_mutation_dynamic_geometric-default-at-target",
    "random_search",
    "hyperband",
]


def get_dataset(
    dataset_name: str,
    algos: typing.Union[list[str], str] = None,
    f_range: list[float] = (0, 24),
    f_steps: int = None,
    priors: list[str] = None,
    benchmarks: list[str] = None,
    rel_ranks: bool = False,
):
    if os.path.exists(
        f"../significance_analysis_example/datasets/{dataset_name.rsplit('_')[0]}/{dataset_name}.parquet"
    ):
        return pd.read_parquet(
            f"../significance_analysis_example/datasets/{dataset_name.rsplit('_')[0]}/{dataset_name}.parquet"
        )
    data = load_priorband_data()
    if not priors:
        priors = data["prior"].unique()
    if not benchmarks:
        benchmarks = std_benchmarks

    algos = figures[algos] if isinstance(algos, str) else algos
    algorithm_variable = "algorithm"
    benchmark_variable = "bench_prior"
    time_variable = "used_fidelity"

    data = data.loc[
        (data[algorithm_variable].isin(algos))
        & (data["benchmark"].isin(benchmarks))
        & (data["prior"].isin(priors))
    ]
    data["used_fidelity"] = data["used_fidelity"].round(8)
    f_space = (
        (
            np.linspace(
                f_range[0],
                f_range[1],
                f_steps if f_steps else f_range[1] - f_range[0] + 1,
            )
            .round(2)
            .tolist()
        )
        if f_steps != 0
        else data[
            data["used_fidelity"].between(f_range[0], f_range[1], inclusive="both")
        ]["used_fidelity"].unique()
    )

    data["benchmark"] = data.apply(rename_benchmarks, bench_dict=label_dict, axis=1)
    data["bench_prior"] = data.apply(combine_bench_prior, axis=1)
    benchmarks_split = data[benchmark_variable].unique()
    data = create_incumbent(
        data, f_space, benchmarks_split, algos, benchmark_variable, algorithm_variable
    )
    data = add_regrets(data, benchmark_variable=benchmark_variable)
    if rel_ranks:
        print(f"{'⚙️ Adding relative ranks':<100}", end="\r", flush=True)
        data["rel_rank"] = data.apply(
            add_rel_ranks,
            data=data,
            benchmark=benchmark_variable,
            time=time_variable,
            axis=1,
        )
    print(f"{'⚙️ Renaming algorithms':<100}", end="\r", flush=True)
    data[algorithm_variable] = data.apply(rename_algos, algo_dict=label_dict, axis=1)
    print(f"{'✅ Dataset loaded':<100}", end="\r", flush=True)
    data.to_parquet(f"datasets/{dataset_name}.parquet")
    return data


def convert_to_autorank(
    data: pd.DataFrame,
    algorithm_variable: str = "algorithm",
    value_variable: str = "value",
    budget_variable: str = "used_fidelity",
    min_f=1,
    max_f=24,
):

    df_autorank = pd.DataFrame()
    for algo in data[algorithm_variable].unique():
        df_autorank[algo] = -data[
            (data[algorithm_variable] == algo)
            & (data[budget_variable] <= max_f)
            & (data[budget_variable] >= min_f)
        ][value_variable].reset_index(drop=True)
    return df_autorank


def add_regret(df: pd.DataFrame, normalize: False):
    best = {}
    ranges = {}
    print("⚙️ Preparing regret", end="\r", flush=True)
    for benchmark in df["bench_prior"].unique():
        best[benchmark] = min(df.loc[df["bench_prior"] == benchmark]["value"])
        ranges[benchmark] = (
            max(df.loc[df["bench_prior"] == benchmark]["value"]) - best[benchmark]
        )

    def calculate_simple_regret(row, normalize: bool = False):
        if normalize:
            return (
                abs(best[row["bench_prior"]] - row["value"]) / ranges[row["bench_prior"]]
            )
        return abs(best[row["bench_prior"]] - row["value"])

    if normalize:
        print("⚙️ Adding regret       ", end="\r", flush=True)
        df["regret"] = df.apply(calculate_simple_regret, axis=1, normalize=True)
        print("✅ Adding regret done                      ")
    else:
        print("⚙️ Adding normalized regret       ", end="\r", flush=True)
        df["norm_regret"] = df.apply(calculate_simple_regret, axis=1, normalize=False)
        print("✅ Adding normalized regret done                      ")
    return df


def create_priorband_benchPrior_relRanks_f24():
    algorithm = "algorithm"
    benchmark = "bench_prior"
    time = "used_fidelity"
    algos = [
        "pb_mutation_dynamic_geometric-default-at-target",
        "random_search",
        "hyperband",
    ]
    fs = [24]
    f_space = np.linspace(1, max(fs), max(fs)).tolist()

    benchmarks = [
        "jahs_cifar10",
        "jahs_colorectal_histology",
        "jahs_fashion_mnist",
        "lcbench-126026",
        "lcbench-167190",
        "lcbench-168330",
        "lcbench-168910",
        "lcbench-189906",
        "cifar100_wideresnet_2048",
        "imagenet_resnet_512",
        "lm1b_transformer_2048",
        "translatewmt_xformer_64",
    ]
    label_dict = {
        "random_search": "RS",
        "hyperband": "HB",
        "pb_mutation_dynamic_geometric-default-at-target": "PB",
        "jahs_cifar10": "JAHS-C10",
        "jahs_colorectal_histology": "JAHS-CH",
        "jahs_fashion_mnist": "JAHS-FM",
        "lcbench-126026": "LC-126026",
        "lcbench-167190": "LC-167190",
        "lcbench-168330": "LC-168330",
        "lcbench-168910": "LC-168910",
        "lcbench-189906": "LC-189906",
        "cifar100_wideresnet_2048": "PD1-Cifar100",
        "imagenet_resnet_512": "PD1-ImageNet",
        "lm1b_transformer_2048": "PD1-LM1B",
        "translatewmt_xformer_64": "PD1-WMT",
        "random_search_prior": "RS+Prior",
        "bo": "BO",
        "pibo-no-default": "PiBO",
        "bohb": "BOHB",
        "priorband_bo": "PriorBand+BO",
    }

    data = load_priorband_data()
    data = data.loc[
        (data[algorithm].isin(algos))
        & (data["benchmark"].isin(benchmarks))
        & (data["prior"].isin(["at25", "bad"]))
    ]
    data["bench_prior"] = data.apply(combine_bench_prior, axis=1)
    data.drop(columns=["benchmark", "prior"], inplace=True)
    benchmarks = data[benchmark].unique()
    max_f = max(fs)
    data = create_incumbent(data, fs, f_space, benchmarks, algos, benchmark, algorithm)[
        max_f
    ]
    print(f"⚙️ F {max_f}: Adding relative ranks             ", end="\r", flush=True)
    data["rel_rank"] = data.apply(
        add_rel_ranks, data=data, benchmark=benchmark, time=time, axis=1
    )
    print(f"⚙️ F {max_f}: Renaming algorithms             ", end="\r", flush=True)
    data[algorithm] = data.apply(rename_algos, algo_dict=label_dict, axis=1)
    print("✅ Dataset loaded                   ", end="\r", flush=True)
    return data


def create_piBo_benchPrior_relRanks_f24():
    algorithm = "algorithm"
    benchmark = "bench_prior"
    time = "used_fidelity"
    algos = ["random_search_prior", "priorband_bo", "pibo-no-default", "bo", "bohb"]
    fs = [24]
    f_space = np.linspace(1, max(fs), max(fs)).tolist()

    benchmarks = [
        "jahs_cifar10",
        "jahs_colorectal_histology",
        "jahs_fashion_mnist",
        "lcbench-126026",
        "lcbench-167190",
        "lcbench-168330",
        "lcbench-168910",
        "lcbench-189906",
        "cifar100_wideresnet_2048",
        "imagenet_resnet_512",
        "lm1b_transformer_2048",
        "translatewmt_xformer_64",
    ]
    label_dict = {
        "random_search": "RS",
        "hyperband": "HB",
        "pb_mutation_dynamic_geometric-default-at-target": "PB",
        "jahs_cifar10": "JAHS-C10",
        "jahs_colorectal_histology": "JAHS-CH",
        "jahs_fashion_mnist": "JAHS-FM",
        "lcbench-126026": "LC-126026",
        "lcbench-167190": "LC-167190",
        "lcbench-168330": "LC-168330",
        "lcbench-168910": "LC-168910",
        "lcbench-189906": "LC-189906",
        "cifar100_wideresnet_2048": "PD1-Cifar100",
        "imagenet_resnet_512": "PD1-ImageNet",
        "lm1b_transformer_2048": "PD1-LM1B",
        "translatewmt_xformer_64": "PD1-WMT",
        "random_search_prior": "RS+Prior",
        "bo": "BO",
        "pibo-no-default": "PiBO",
        "bohb": "BOHB",
        "priorband_bo": "PriorBand+BO",
    }

    data = load_priorband_data()
    data = data.loc[
        (data[algorithm].isin(algos))
        & (data["benchmark"].isin(benchmarks))
        & (data["prior"].isin(["at25", "bad"]))
    ]
    data["bench_prior"] = data.apply(combine_bench_prior, axis=1)
    data.drop(columns=["benchmark", "prior"], inplace=True)
    benchmarks = data[benchmark].unique()
    max_f = max(fs)
    data = create_incumbent(data, fs, f_space, benchmarks, algos, benchmark, algorithm)[
        max_f
    ]
    print(f"⚙️ F {max_f}: Adding relative ranks             ", end="\r", flush=True)
    data["rel_rank"] = data.apply(
        add_rel_ranks, data=data, benchmark=benchmark, time=time, axis=1
    )
    print(f"⚙️ F {max_f}: Renaming algorithms             ", end="\r", flush=True)
    data[algorithm] = data.apply(rename_algos, algo_dict=label_dict, axis=1)
    print("✅ Dataset loaded                   ", end="\r", flush=True)
    data.to_parquet("pibo_benchPrior_relRanks_f24_meta.parquet")
    return data


def add_benchmark_metafeatures(data: pd.DataFrame):
    meta_feature_df = pd.read_csv("benchmark_metafeatures.csv")

    def add_meta_features(row):
        return meta_feature_df.loc[
            meta_feature_df["code_name"] == row["bench_prior"].rsplit("_", 1)[0]
        ][
            ["# Vars", "# cont. Vars", "# cond. Vars", "# cat. Vars", "log int", "int"]
        ].values[
            0
        ]

    data[
        ["n_Vars", "n_cont_Vars", "n_cond_Vars", "n_cat_Vars", "log_int", "int"]
    ] = data.apply(add_meta_features, axis=1).to_list()
    return data
