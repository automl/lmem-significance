import math
import typing

import matplotlib
import numpy as np
import pandas as pd
import requests
from autorank._util import RankResult, get_sorted_rank_groups
from matplotlib import pyplot as plt
from pandas.api.types import is_numeric_dtype
from pymer4.models import Lm, Lmer
from scipy import stats
from pathlib import Path

pd.options.mode.chained_assignment = None

ALGORITHM = "algorithm"
VALUE = "value"
SEED = "seed"
BUDGET = "used_fidelity"
BENCHMARK = "benchmark"

def download_dataset(url: str, path: Path) -> None:
    """Helper function to download a file from a URL and decompress it and store by given name.

    Args:
        url (str): URL of the file to download
        path (Path): Path along with filename to save the downloaded file

    Returns:
        bool: Flag to indicate if the download and decompression was successful
    """
    # Check if the file already exists
    if path.exists():
        return

    # Send a HTTP request to the URL of the file
    response = requests.get(url, allow_redirects=True)

    # Check if the request is successful
    if response.status_code != 200:
        raise ValueError(
            f"Failed to download the surrogate from {url}."
            f" Recieved HTTP status code: {response.status_code}."
            "Please either try again later, use an alternative link or contact the authors through github."
        )

    print(response.content)
    # Save the .tar.gz file
    data=pd.DataFrame(response.content)
    data.to_parquet(path)

def cd_diagram(
    result, reverse: bool = False, width: float = 4, system_id="algorithm", parent_ax=None
):
    """
    Creates a Critical Distance diagram.
    """

    def plot_line(line, color="k", **kwargs):
        ax.plot(
            [pos[0] / width for pos in line],
            [pos[1] / height for pos in line],
            color=color,
            **kwargs,
        )

    def plot_text(x, y, s, rot=0, *args, **kwargs):
        ax.text(x / width, y / height, s, rotation=rot, *args, **kwargs)

    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not all(isinstance(df, pd.DataFrame) for df in result)
    ):
        result_copy = RankResult(**result._asdict())
        result_copy = result_copy._replace(
            rankdf=result.rankdf.sort_values(by="meanrank")
        )
        sorted_ranks, names, groups = get_sorted_rank_groups(result_copy, reverse)
        cd = [result.cd]
    else:
        result = list(result)
        estimates = result[0].set_index(system_id)
        estimates = estimates.sort_values(by="Estimate")
        sorted_ranks = pd.DataFrame()
        sorted_ranks = estimates["Estimate"]
        sorted_ranks.name = "meanrank"
        estimates["ci_upper"] = estimates["2.5_ci"]
        estimates["ci_lower"] = estimates["97.5_ci"]
        names = estimates.index.values.tolist()
        contrasts = result[1]
        for pair in contrasts["Contrast"]:
            sys_1 = pair.split(" - ")[0]
            sys_2 = pair.split(" - ")[1]
            contrasts.loc[contrasts["Contrast"] == pair, f"{system_id}_1"] = (
                sys_1 if sys_1[0] != "(" or sys_1[-1] != ")" else sys_1[1:-1]
            )
            contrasts.loc[contrasts["Contrast"] == pair, f"{system_id}_2"] = (
                sys_2 if sys_2[0] != "(" or sys_2[-1] != ")" else sys_2[1:-1]
            )
        contrasts = contrasts.drop("Contrast", axis=1)
        column = contrasts.pop(f"{system_id}_2")
        contrasts.insert(0, f"{system_id}_2", column)
        column = contrasts.pop(f"{system_id}_1")
        contrasts.insert(0, f"{system_id}_1", column)
        groups = []
        for _, row in contrasts.iterrows():
            algos = (row[f"{system_id}_1"], row[f"{system_id}_2"])
            if row["P-val"] > 0.05:
                group = [names.index(algos[0]), names.index(algos[1])]
                group.sort()
                groups.append((group[0], group[1]))
        new_groups = []
        for group in groups:
            if not any(
                group[0] >= g[0] and group[1] <= g[1] and group != g for g in groups
            ):
                new_groups.append(group)
        groups = new_groups
        # t_stat=max(abs(contrasts.Estimate.min()),abs(contrasts.Estimate.min()))/contrasts.SE.min()
        # p_hsd=1-studentized_range.cdf(t_stat*np.sqrt(2), k=len(estimates), df=contrasts.DF[0])
        hsd = [
            (
                stats.studentized_range.ppf(
                    1 - 0.05, k=len(estimates), df=contrasts.DF.min()
                )
                / np.sqrt(2)
                * contrasts.SE.min()
            ),
            (
                stats.studentized_range.ppf(
                    1 - 0.05, k=len(estimates), df=contrasts.DF.max()
                )
                / np.sqrt(2)
                * contrasts.SE.max()
            ),
        ]
        cd = hsd

    granularity = max(
        2 ** round(math.log2((max(sorted_ranks) - min(sorted_ranks)) / 6)), 0.03125
    )
    if granularity < 0.25:
        granularity = 10 ** round(math.log10((max(sorted_ranks) - min(sorted_ranks)) / 3))

    lowv = round(
        (math.floor(min(sorted_ranks) / granularity)) * granularity,
        len(str(int(1 / granularity))) + 1,
    )
    highv = round(
        (math.ceil(max(sorted_ranks) / granularity)) * granularity,
        len(str(int(1 / granularity))) + 1,
    )
    cline = 0.4
    textspace = 1
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            relative_rank = rank - lowv
        else:
            relative_rank = highv - rank
        return textspace + scalewidth / (highv - lowv) * relative_rank

    linesblank = 0.2 + 0.2 + (len(groups) - 1) * 0.1
    rounder = len(str(int(1 / granularity)))
    # add scale
    if granularity < 0.25:
        numbers = list(
            np.round(
                np.linspace(
                    lowv,
                    highv + granularity,
                    round((highv + granularity - lowv) / granularity),
                    endpoint=False,
                ),
                rounder,
            )
        )
    else:
        numbers = list(
            np.round(
                np.linspace(
                    lowv,
                    highv + granularity,
                    round((highv + granularity - lowv) / granularity),
                    endpoint=False,
                ),
                rounder + 2,
            )
        )  # list(np.arange(lowv, highv + granularity, granularity))
    distanceh = 0.2 if cd else 0
    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((len(sorted_ranks) + 1) / 2) * 0.2 + minnotsignificant

    if not parent_ax:
        fig_cd = plt.figure(figsize=(width, height))
        fig_cd.set_facecolor("white")
        ax = fig_cd.add_axes([0, 0, 1, 1])  # reverse y axis
    else:
        ax = parent_ax
    ax.set_axis_off()

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    plot_line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05
    tinytick = 0.03

    for a in list(np.arange(lowv, highv, granularity)) + [lowv, highv]:
        tick = tinytick
        if a * 2 == int(a * 2):
            tick = smalltick
        if a == int(a):
            tick = bigtick
        plot_line([(rankpos(a), cline - tick / 2), (rankpos(a), cline)], linewidth=0.7)

    for a in numbers:
        a = int(a) if a == int(a) else a
        plot_text(
            rankpos(a),
            cline - tick / 2 - 0.05,
            str(a),
            rot=90
            if (
                (len(numbers) > 7 or (granularity < 0.125 and len(numbers) > 6))
                and len(str(abs(a))) >= 3
            )
            else 0,
            size=12 - 2 * len(str(abs(a))) + 1.2 * min(len(str(abs(a))) for a in numbers),
            ha="center",
            va="bottom",
        )

    # Left half of algorithms and pointers
    for i in range(math.ceil(len(sorted_ranks) / 2)):
        chei = cline + minnotsignificant + i * 0.2
        plot_line(
            [
                (rankpos(sorted_ranks[i]), cline),
                (rankpos(sorted_ranks[i]), chei),
                (textspace - 0.1, chei),
            ],
            linewidth=0.7,
        )
        plot_text(textspace - 0.2, chei, names[i], ha="right", va="center")

    # Right half of algorithms and pointers
    for i in range(math.ceil(len(sorted_ranks) / 2), len(sorted_ranks)):
        chei = cline + minnotsignificant + (len(sorted_ranks) - i - 1) * 0.2
        plot_line(
            [
                (rankpos(sorted_ranks[i]), cline),
                (rankpos(sorted_ranks[i]), chei),
                (textspace + scalewidth + 0.1, chei),
            ],
            linewidth=0.7,
        )
        plot_text(textspace + scalewidth + 0.2, chei, names[i], ha="left", va="center")

    # upper scale
    for cd_n, cdv in enumerate(cd):
        if not reverse:
            begin, end = rankpos(lowv), rankpos(lowv + cdv)
        else:
            begin, end = rankpos(highv), rankpos(highv - cdv)
        plot_line(
            [(begin, distanceh), (end, distanceh)],
            linestyle="dashed" if cd_n == 1 else "solid",
            linewidth=0.7,
        )
        plot_line(
            [(begin, distanceh + bigtick / 2), (begin, distanceh - bigtick / 2)],
            linewidth=0.7,
        )
        plot_line(
            [(end, distanceh + bigtick / 2), (end, distanceh - bigtick / 2)],
            linewidth=0.7,
        )
        if cd_n == len(cd) - 1:
            plot_text((begin + end) / 2, distanceh - 0.05, "CD", ha="center", va="bottom")

    # no-significance lines
    side = 0.015
    no_sig_height = 0.1
    start = cline + 0.2
    for l, r in groups:
        plot_line(
            [
                (rankpos(sorted_ranks[l]) - side, start),
                (rankpos(sorted_ranks[r]) + side, start),
            ],
            linewidth=2.5,
            solid_capstyle="round",
        )
        start += no_sig_height
    if not parent_ax:
        return fig_cd


def dataframe_validator(
    data: pd.DataFrame,
    algorithm_var: str = ALGORITHM,
    benchmark_var: str = BENCHMARK,
    loss_var: str = VALUE,
    fidelity_var: str = BUDGET,
    seed_var: str = SEED,
    verbose: bool = True,
    **extra_vars,
) -> typing.Tuple[pd.DataFrame, list[str]]:
    """
    Validates the columns of a pandas DataFrame and converts them to the appropriate data types if necessary.

    Args:
        data (pd.DataFrame): The input DataFrame to be validated.
        algorithm_var (str, optional): The name of the column containing the algorithm names. Defaults to 'ALGORITHM'.
        benchmark_var (str, optional): The name of the column containing the benchmark names. Defaults to 'BENCHMARK'.
        loss_var (str, optional): The name of the column containing the loss values. Defaults to 'VALUE'.
        fidelity_var (str, optional): The name of the column containing the fidelity values. Defaults to 'BUDGET'.
        seed_var (str, optional): The name of the column containing the seed values. Defaults to 'SEED'.
        verbose (bool, optional): Whether to print conversion messages. Defaults to True.
        **extra_vars: Additional columns to be validated.

    Returns:
        tuple: A tuple containing the validated DataFrame and a list of valid column names.

    Raises:
        None

    Examples:
        >>> df = pd.DataFrame({'ALGORITHM': ['A', 'B', 'C'],
        ...                    'SEED': [1, 2, 3],
        ...                    'BENCHMARK': ['X', 'Y', 'Z'],
        ...                    'VALUE': [0.1, 0.2, 0.3],
        ...                    'BUDGET': [100, 200, 300]})
        >>> dataframe_validator(df)
        (  ALGORITHM  SEED BENCHMARK  VALUE  BUDGET
        0         A     1         X    0.1   100.0
        1         B     2         Y    0.2   200.0
        2         C     3         Z    0.3   300.0,
        ['ALGORITHM', 'SEED', 'BENCHMARK', 'VALUE', 'BUDGET'])
    """
    cols = data.dtypes
    valid_columns = []
    for col in [algorithm_var, seed_var, benchmark_var]:
        if col in data.columns:
            if cols[col] != "object":
                try:
                    data[col] = data[col].astype(object)
                    valid_columns.append(col)
                    if verbose:
                        print(f"Converted column {col} to object.")
                except Exception as e:
                    print(
                        f"Error {e}: Column {col} is not of type object, could not convert all values to object."
                    )
            else:
                valid_columns.append(col)
    for col in [loss_var, fidelity_var]:
        if col in data.columns:
            if not cols[col] in ["float", "int"]:
                try:
                    data[col] = data[col].astype(np.float64)
                    valid_columns.append(col)
                    if verbose:
                        print(f"Converted column {col} to float.")
                except Exception as e:
                    print(
                        f"Error {e}: Column {col} is not numeric, could not convert all values to float."
                    )
            else:
                valid_columns.append(col)
    for _kw, col in extra_vars.items():
        if is_numeric_dtype(data[col]):
            data[col] = data[col].astype(np.float64)
        else:
            data[col] = data[col].astype(object)
        valid_columns.append(col)
    return data, valid_columns


def glrt(mod1, mod2, verbose: bool = False) -> dict[str, typing.Any]:
    """Generalized Likelihood Ratio Test on two Liner Mixed Effect Models from R

    Args:
        mod1 (Lmer): First, simple model, Null-Hypothesis assumes that this model contains not significantly less information as the second model
        mod2 (Lmer): Second model, Alternative Hypothesis assumes that this model contains significant new information
        verbose (bool, optional): Outputting of results. Defaults to False.

    Returns:
        dict[str,typing.Any]: Result dictionary with Chi-Square-Score, Degrees of Freedom and p-value of the test

    Raises:
        None

    Examples:
        >>> mod1 = model("value ~ algorithm + (1|seed)", data)
        >>> mod2 = model("value ~ algorithm + (1|seed) + (1|benchmark)", data)
        >>> glrt(mod1,mod2)
        {'p': 0.0, 'chi_square': 7.0, 'df': 1}
    """
    assert (
        mod1.logLike
        and mod2.logLike
        and mod1.coefs is not None
        and mod2.coefs is not None
    )
    chi_square = 2 * abs(mod1.logLike - mod2.logLike)
    delta_params = abs(len(mod1.coefs) - len(mod2.coefs))
    p = 1 - stats.chi2.cdf(chi_square, df=delta_params)
    if verbose:
        print(
            f"{mod1.formula} ({round(mod1.logLike,4)}) {'==' if p>0.05 or mod1.logLike==mod2.logLike else '>>' if mod1.logLike>mod2.logLike else '<<'} {mod2.formula} ({round(mod2.logLike,4)})"
        )
        print(f"Chi-Square: {chi_square}, P-Value: {p}")
    return {
        "p": p,
        "chi_square": chi_square,
        "df": delta_params,
    }


def model(
    formula: str,
    data: pd.DataFrame,
    system_id: str = "algorithm",
    factor: typing.Union[str, list[str]] = None,
    dummy=True,
    no_warnings=True,
) -> typing.Union[Lm, Lmer]:
    """
    Model object for Linear (Mixed Effects) Model-based significance analysis.

    Args:
        formula (str): The formula specifying the regression model.
        data (pd.DataFrame): The input data for the regression model.
        system_id (str, optional): The column name in the data representing the system ID. Defaults to "algorithm".
        factor (str or list[str], optional): The column name(s) in the data representing the factor(s) to include in the model. Defaults to None.
        dummy (bool, optional): Whether to include a dummy variable in the model to enforce the use of an LMEM. Defaults to True.
        no_warnings (bool, optional): Whether to suppress warnings during model fitting. Defaults to True.

    Returns:
        Union[Lm, Lmer]: The fitted regression model.

    Raises:
        None

    Examples:
        # Example 1: Perform linear regression
        data = pd.DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6]})
        model('y ~ x', data)

        # Example 2: Perform mixed-effects regression
        data = pd.DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6], 'group': ['A', 'B', 'A']})
        model('y ~ x + (1|group)', data)
    """

    if not "|" in formula:
        if dummy:
            data.loc[:,"dummy"] = "0"
            data.at[data.index[0], "dummy"] = "1"
            formula += "+(1|dummy)"
        else:
            mod = Lm(formula, data)
            mod.fit(cluster=data[system_id], verbose=False, summarize=False)
            return mod
    model = Lmer(
        formula=formula,
        data=data,
    )
    factors = {system_id: list(data[system_id].unique())}
    if factor:
        if isinstance(factor, str):
            factors[factor] = list(data[factor].unique())
        else:
            for f in factor:
                factors[f] = list(data[f].unique())
    model.fit(
        factors=factors,
        REML=False,
        summarize=False,
        verbose=False,
        no_warnings=no_warnings,
    )
    return model


def metafeature_analysis(
    dataset: pd.DataFrame,
    algorithms: typing.Tuple[str, str],
    metafeature_var: str,
    algorithm_var: str = ALGORITHM,
    benchmark_var: str = BENCHMARK,
    loss_var: str = VALUE,
    fidelity_var: str = BUDGET,
    path: str = "",
    plot: bool = False,
    verbose: bool = True,
) -> typing.Tuple[matplotlib.figure.Figure, pd.DataFrame]:
    """Metafeature analysis for a given dataset

    Args:
        dataset (pd.DataFrame): Dataset to be analyzed
        algorithms (typing.Tuple[str, str]): Algorithms to be compared
        metafeature_var (str): Metafeature to be analyzed
        algorithm_var (str, optional): Algorithm variable. Defaults to ALGORITHM.
        benchmark_var (str, optional): Benchmark variable. Defaults to BENCHMARK.
        loss_var (str, optional): Loss variable. Defaults to VALUE.
        fidelity_var (str, optional): Fidelity veriable. Defaults to BUDGET.
        path (str, optional): Path to save significance scores. Defaults to "".
        plot (bool, optional): Whether to plot the CD-diagram. Defaults to False.
        verbose (bool, optional): Verbosity of the analysis. Defaults to True.

    Raises:
        ValueError: Error if the named algorithms are not in the dataset

    Returns:
        typing.Tuple[pd.DataFrame, pd.DataFrame]: DataFrame with scores and pairwise signficances
    """
    dataset["benchmark_variant"] = dataset.apply(
        lambda x: f"{x[benchmark_var]} x {x[metafeature_var]}", axis=1
    )
    dataset = dataset.loc[dataset[algorithm_var].isin(algorithms)]
    dataset, cols = dataframe_validator(dataset, fidelity_var=fidelity_var ,metafeature=metafeature_var, verbose=verbose)
    if not all(
        [
            x in cols
            for x in [
                algorithm_var,
                benchmark_var,
                metafeature_var,
                loss_var,
                fidelity_var,
            ]
        ]
    ):
        raise ValueError("Not all necessary columns are included in the dataset")
    benchmark = "benchmark_variant"
    wins_bench = pd.DataFrame()
    full_wins = []
    full_benchmarks = []
    full_fidelities = []
    for f_n, f in enumerate(dataset[fidelity_var].unique()):
        if verbose:
            print(
                f"{f:<{max([str(x) for x in dataset[fidelity_var].unique()],key=len)}} ({f_n+1}/{len(dataset[fidelity_var].unique())})",
                end="\r",
                flush=True,
            )
        wins_budget = []
        for bench in dataset[benchmark].unique():
            if (
                len(
                    dataset.loc[
                        (dataset[fidelity_var] == f) & (dataset[benchmark] == bench)
                    ]
                )
                == 0
            ):
                continue
            full_fidelities.append(f)
            mod = model(
                f"{loss_var}~{algorithm_var}",
                dataset.loc[(dataset[fidelity_var] == f) & (dataset[benchmark] == bench)],
                algorithm_var,
            )
            assert isinstance(mod, Lmer)
            post_hocs = mod.post_hoc(algorithm_var)
            if post_hocs[1].Sig[0] in ["***", "**", "*"]:
                wins_budget.append(
                    -1
                    if post_hocs[1]
                    .Contrast[0]
                    .rsplit(" - ")[0 if post_hocs[1].Estimate[0] < 0 else 1]
                    == algorithms[0]
                    else 1
                )
                full_wins.append(
                    -1
                    if post_hocs[1]
                    .Contrast[0]
                    .rsplit(" - ")[0 if post_hocs[1].Estimate[0] < 0 else 1]
                    == algorithms[0]
                    else 1
                )
            else:
                wins_budget.append(0)
                full_wins.append(0)
            full_benchmarks.append(bench)
    wins_bench[benchmark] = full_benchmarks
    wins_bench["wins"] = full_wins
    wins_bench["fidelity"] = full_fidelities
    wins_bench["wins"] = wins_bench["wins"].astype(float)
    wins_bench[benchmark] = wins_bench[benchmark].astype(str)
    wins_bench[["benchmark", "metafeature"]] = wins_bench.apply(
        lambda x: pd.Series(x[benchmark].rsplit(" x ", 1)), axis=1
    )
    if path:
        path = path if path.endswith(".parquet") else path + ".parquet"
        wins_bench.to_parquet(path)
    if verbose:
        print(wins_bench.head(5))
    wins_model = model("wins~benchmark_variant+fidelity", wins_bench, "benchmark_variant")
    assert isinstance(wins_model, Lmer)
    metafeatures_significances = wins_model.post_hoc("benchmark_variant")
    if plot:
        plt.figure(
            cd_diagram(
                metafeatures_significances,
                system_id="benchmark_variant",
                reverse=False,
                width=5,
            )
        )

    return metafeatures_significances


def seed_dependency_check(
    data: pd.DataFrame,
    algorithm_var: str = ALGORITHM,
    loss_var: str = VALUE,
    seed_var: str = SEED,
    verbose: bool = True,
) -> list[str]:
    """Check for seed dependency in a dataset

    Args:
        data (pd.DataFrame): Dataset to be analyzed
        algorithm_var (str, optional): Algorithm variable. Defaults to ALGORITHM.
        loss_var (str, optional): Loss variable. Defaults to VALUE.
        seed_var (str, optional): Seed variable. Defaults to SEED.
        verbose (bool, optional): Verbosity of check. Defaults to True.

    Returns:
        list[str]: Algorithms that are likely influenced by the seed
    """
    simple_model = model(
        formula=f"{loss_var}~{algorithm_var}",
        data=data,
        dummy=False,
        no_warnings=True,
    )
    seed_model = model(
        formula=f"{loss_var}~(0+{algorithm_var}|{seed_var})",
        data=data,
        dummy=False,
        no_warnings=True,
    )
    test_result = glrt(
        simple_model,
        seed_model,
        verbose,
    )
    assert (
        simple_model.logLike
        and seed_model.logLike
        and seed_model.ranef is not None
        and test_result
    )
    if test_result["p"] < 0.05 and seed_model.logLike > simple_model.logLike:
        ranef_var = seed_model.ranef_var
        influenced = ranef_var.loc[
            (ranef_var["Var"] / 10 >= ranef_var["Var"].min())
            & (ranef_var.index != "Residual")
            & (ranef_var["Var"] * 10 >= ranef_var["Var"].max())
        ]["Name"].to_list()
        influenced = [x.rsplit(algorithm_var, 1)[1] for x in influenced]
        if verbose:
            print(f"Seed is a significant effect, likely influenced algorithms: {influenced}")
        return influenced
    else:
        if verbose:
            print("=> Seed is not a significant effect")
        return []


def benchmark_information_check(
    data: pd.DataFrame,
    algorithm_var: str = ALGORITHM,
    benchmark_var: str = BENCHMARK,
    loss_var: str = VALUE,
    rank_benchmarks: bool = False,
    verbose: bool = True,
) -> typing.Union[dict[str, bool], typing.Tuple[dict[str, bool], pd.DataFrame]]:
    """Benchmark-wise check for variation between algorithms in a dataset.

    Args:
        data (pd.DataFrame): Dataset to be analyzed
        algorithm_var (str, optional): Algorithm variable. Defaults to ALGORITHM.
        benchmark_var (str, optional): Benchmark variable. Defaults to BENCHMARK.
        loss_var (str, optional): Loss variable. Defaults to VALUE.
        rank_benchmarks (bool, optional): Ranking benchmarks (involves calculating the individual random effects - takes time!). Defaults to False.
        verbose (bool, optional): Verbosity of the check. Defaults to True.

    Returns:
        typing.Union[dict[str, bool], typing.Tuple[dict[str, bool], pd.DataFrame]]: Informativeness of benchmarks and random effects if rank_benchmarks is True
    """
    test_results = {}
    benchmark_info = {}
    for benchmark in data[benchmark_var].unique():
        simple_mod = model(
            formula=f"{loss_var}~1",
            data=data.loc[data[benchmark_var] == benchmark],
            # factor_list=[self.exploratory_var],
            dummy=False,
            no_warnings=True,
        )
        benchmark_mod = model(
            formula=f"{loss_var}~{algorithm_var}",
            data=data.loc[data[benchmark_var] == benchmark],
            # factor_list=[self.exploratory_var],
            dummy=False,
            no_warnings=True,
        )
        if verbose:
            print(f"\nBenchmark: {benchmark}")
        test_results[benchmark] = glrt(
            simple_mod,
            benchmark_mod,
            verbose,
        )
        assert benchmark_mod.logLike and simple_mod.logLike
        if (
            test_results[benchmark]["p"] < 0.05
            and benchmark_mod.logLike > simple_mod.logLike
        ):
            if verbose:
                print(
                    f"=> Benchmark {benchmark:<{max([len(x) for x in data[benchmark_var].unique()])}} is informative."
                )
            benchmark_info[benchmark] = True
        else:
            if verbose:
                print(
                    f"=> Benchmark {benchmark:<{max([len(x) for x in data[benchmark_var].unique()])}} is uninformative."
                )
            benchmark_info[benchmark] = False

    if rank_benchmarks:
        all_benchmarks_mod = model(
            formula=f"{loss_var}(0+{benchmark_var}|{algorithm_var})",
            data=data,
            # factor_list=[self.exploratory_var],
            dummy=False,
        )
        if verbose:
            print("")
        ranef_var = all_benchmarks_mod.ranef_var[:-1]

        def rename_var_name(row):
            return row["Name"].rsplit(benchmark_var, 1)[1]

        ranef_var["Name"] = ranef_var.apply(rename_var_name, axis=1)
        if verbose:
            print(ranef_var.reset_index(drop=True))
        uninformative = ranef_var.loc[
            (ranef_var["Var"] * 10 <= ranef_var["Var"].max())
            & (ranef_var.index != "Residual")
            & (ranef_var["Var"] / 10 <= ranef_var["Var"].min())
        ]["Name"].to_list()
        if verbose:
            print(
                f"Benchmarks without algorithm variation: {[x.rsplit(benchmark_var,1)[0] for x in uninformative]}"
            )
        return benchmark_info, ranef_var
    return benchmark_info


def fidelity_check(
    data: pd.DataFrame,
    fidelity_var: str = BUDGET,
    algorithm_var: str = ALGORITHM,
    benchmark_var: str = BENCHMARK,
    loss_var: str = VALUE,
    verbose: bool = True,
) -> None:
    """Check for the significance of a fidelity variable in a dataset

    Args:
        data (pd.DataFrame): Dataset to be analyzed
        fidelity_var (str, optional): Fidelity variable. Defaults to BUDGET.
        algorithm_var (str, optional): Algorithm variable. Defaults to ALGORITHM.
        benchmark_var (str, optional): Benchmark variable. Defaults to BENCHMARK.
        loss_var (str, optional): Loss variable. Defaults to VALUE.
        verbose (bool, optional): Verbosity of the check. Defaults to True.
    """
    significances = {}
    simple_formula = f"{loss_var} ~ {algorithm_var}{f' + (1|{benchmark_var})' if data[benchmark_var].nunique()>1 else ''}"
    simple_mod = model(
        formula=simple_formula,
        data=data,
        # factor_list=[self.exploratory_var],
        dummy=data[benchmark_var].nunique() == 1,
        no_warnings=True,
    )
    fidelity_mod = model(
        formula=f"{simple_formula} + {fidelity_var}",
        data=data,
        # factor_list=[self.exploratory_var],
        dummy=data[benchmark_var].nunique() == 1,
        no_warnings=True,
    )
    test_result = glrt(
        simple_mod,
        fidelity_mod,
        verbose,
    )
    if verbose:
        print("")
    assert fidelity_mod.logLike and simple_mod.logLike
    if test_result["p"] < 0.05 and fidelity_mod.logLike > simple_mod.logLike:
        significances[fidelity_var] = 1
    else:
        significances[fidelity_var] = 0
    fid_group_mod = model(
        formula=f"{simple_formula}+ {algorithm_var}:{fidelity_var}",
        data=data,
        # factor_list=[self.exploratory_var],
        dummy=data[benchmark_var].nunique() == 1,
    )
    test_result = glrt(
        simple_mod,
        fid_group_mod,
        verbose,
    )
    if test_result["p"] < 0.05 and fid_group_mod.logLike > simple_mod.logLike:
        significances[f"{fidelity_var}_group"] = 1
    else:
        significances[f"{fidelity_var}_group"] = 0
    if significances[fidelity_var] == 1 and significances[f"{fidelity_var}_group"] == 1:
        if verbose:
            print("")
        test_result = glrt(
            fidelity_mod,
            fid_group_mod,
            verbose,
        )
        if verbose:
            print("")
        if test_result["p"] < 0.05 and fid_group_mod.logLike > fidelity_mod.logLike:
            if verbose:
                print(
                    f"=> Fidelity {fidelity_var} is both as simple and interaction effect significant, but interaction effect performs better."
                )
            return "interaction_effect"
        else:
            if verbose:
                print(
                    f"=> Fidelity {fidelity_var} is both as simple and interaction effect significant, but as simple effect performs better."
                )
            return "simple_effect"
    elif significances[fidelity_var] == 1:
        if verbose:
            print(f"=> Fidelity {fidelity_var} as simple effect is significant.")
        return "simple_effect"
    elif significances[f"{fidelity_var}_group"] == 1:
        if verbose:
            print(f"=>  Fidelity {fidelity_var} as interaction effect is significant.")
        return "interaction_effect"
    else:
        if verbose:
            print(f"=> Fidelity {fidelity_var} is not a significant effect.")
        return "none"


def convert_to_autorank(
    data: pd.DataFrame,
    algorithm_variable: str = "algorithm",
    value_variable: str = "value"
)->pd.DataFrame:
    """Converts an LMEM-compatible-formatted dataframe to the format required by the autorank package.

    Args:
        data (pd.DataFrame): Dataframe to be converted.
        algorithm_variable (str, optional): Algorithm variable (column names). Defaults to "algorithm".
        value_variable (str, optional): Value variable (column entries). Defaults to "value".

    Returns:
        pd.DataFrame: _description_
    """

    df_autorank = pd.DataFrame()
    for algo in data[algorithm_variable].unique():
        df_autorank[algo] = -data[ (data[algorithm_variable] == algo)][value_variable].reset_index(drop=True)
    return df_autorank
