import math
import typing
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymer4.models import Lm, Lmer
from scipy import stats
from scipy.stats import studentized_range

pd.set_option("chained_assignment", None)
pd.set_option("display.max_rows", 5000)
pd.set_option("display.max_columns", 5000)
pd.set_option("display.width", 10000)


def glrt(
    mod1, mod2, names: list[str] = None, returns: bool = False
) -> dict[str, typing.Any]:
    """Generalized Likelihood Ratio Test on two Liner Mixed Effect Models from R

    Args:
        mod1 (Lmer): First, simple model, Null-Hypothesis assumes that this model contains not significantly less information as the second model
        mod2 (Lmer): Second model, Alternative Hypothesis assumes that this model contains significant new information

    Returns:
        dict[str,typing.Any]: Result dictionary with Chi-Square-Score, Degrees of Freedom and p-value of the test
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
    if names:
        print(
            f"{names[0]} ({round(mod1.logLike,2)}) {'==' if p>0.05 or mod1.logLike==mod2.logLike else '>>' if mod1.logLike>mod2.logLike else '<<'} {names[1]} ({round(mod2.logLike,2)})"
        )
        print(f"Chi-Square: {chi_square}, P-Value: {p}")
    if returns:
        return {
            "p": p,
            "chi_square": chi_square,
            "df": delta_params,
        }


def model(
    formula: str,
    data: pd.DataFrame,
    system_id: str = "algorithm",
    factor: str = None,
    factor_list: list[str] = None,
    dummy=True,
    no_warnings=True,
) -> typing.Union[Lm, Lmer]:
    if not "|" in formula:
        if dummy:
            data["dummy"] = "0"
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
        factors[factor] = list(data[factor].unique())
    if factor_list:
        for factor in factor_list:
            factors[factor] = list(data[factor].unique())
    model.fit(
        factors=factors,
        REML=False,
        summarize=False,
        verbose=False,
        no_warnings=no_warnings,
    )
    return model


def benchmark_clustering(
    dataset: pd.DataFrame,
    algorithm_var: str,
    algorithms: typing.Tuple[str, str],
    benchmark_var: str,
    metafeature_var: str,
    loss_var: str,
    fidelity: str,
    path: str = None,
):
    dataset["benchmark_variant"] = dataset.apply(
        lambda x: f"{x[benchmark_var]} x {x[metafeature_var]}", axis=1
    )
    dataset = dataset.loc[dataset[algorithm_var].isin(algorithms)]
    benchmark = "benchmark_variant"
    wins_bench = pd.DataFrame()
    full_wins = []
    full_benchmarks = []
    full_fidelities = []
    for f_n, f in enumerate(dataset[fidelity].unique()):
        print(
            f"{f:<{max([str(x) for x in dataset[fidelity].unique()],key=len)}} ({f_n+1}/{len(dataset[fidelity].unique())})",
            end="\r",
            flush=True,
        )
        wins_budget = []
        for bench in dataset[benchmark].unique():
            if (
                len(dataset.loc[(dataset[fidelity] == f) & (dataset[benchmark] == bench)])
                == 0
            ):
                continue
            full_fidelities.append(f)
            mod = model(
                f"{loss_var}~{algorithm_var}",
                dataset.loc[(dataset[fidelity] == f) & (dataset[benchmark] == bench)],
                algorithm_var,
            )
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
    return wins_bench


def get_sorted_rank_groups(result, reverse):
    if reverse:
        names = result.rankdf.iloc[::-1].index.to_list()
        if result.cd is not None:
            sorted_ranks = result.rankdf.iloc[::-1].meanrank
            critical_difference = result.cd
        else:
            sorted_ranks = result.rankdf.iloc[::-1].meanrank
            critical_difference = (
                result.rankdf.ci_upper[0] - result.rankdf.ci_lower[0]
            ) / 2
    else:
        names = result.rankdf.index.to_list()
        if result.cd is not None:
            sorted_ranks = result.rankdf.meanrank
            critical_difference = result.cd
        else:
            sorted_ranks = result.rankdf.meanrank
            critical_difference = (
                result.rankdf.ci_upper[0] - result.rankdf.ci_lower[0]
            ) / 2

    groups = []
    cur_max_j = -1
    for i, _ in enumerate(sorted_ranks):
        max_j = None
        for j in range(i + 1, len(sorted_ranks)):
            if abs(sorted_ranks[i] - sorted_ranks[j]) <= critical_difference:
                max_j = j
                # print(i, j)
        if max_j is not None and max_j > cur_max_j:
            cur_max_j = max_j
            groups.append((i, max_j))
    return sorted_ranks, names, groups


class RankResult(
    namedtuple(
        "RankResult",
        (
            "rankdf",
            "pvalue",
            "cd",
            "omnibus",
            "posthoc",
            "all_normal",
            "pvals_shapiro",
            "homoscedastic",
            "pval_homogeneity",
            "homogeneity_test",
            "alpha",
            "alpha_normality",
            "num_samples",
            "posterior_matrix",
            "decision_matrix",
            "rope",
            "rope_mode",
            "effect_size",
            "force_mode",
        ),
    )
):
    __slots__ = ()

    def __str__(self):
        return (
            "RankResult(rankdf=\n%s\n"
            "pvalue=%s\n"
            "cd=%s\n"
            "omnibus=%s\n"
            "posthoc=%s\n"
            "all_normal=%s\n"
            "pvals_shapiro=%s\n"
            "homoscedastic=%s\n"
            "pval_homogeneity=%s\n"
            "homogeneity_test=%s\n"
            "alpha=%s\n"
            "alpha_normality=%s\n"
            "num_samples=%s\n"
            "posterior_matrix=\n%s\n"
            "decision_matrix=\n%s\n"
            "rope=%s\n"
            "rope_mode=%s\n"
            "effect_size=%s\n"
            "force_mode=%s)"
            % (
                self.rankdf,
                self.pvalue,
                self.cd,
                self.omnibus,
                self.posthoc,
                self.all_normal,
                self.pvals_shapiro,
                self.homoscedastic,
                self.pval_homogeneity,
                self.homogeneity_test,
                self.alpha,
                self.alpha_normality,
                self.num_samples,
                self.posterior_matrix,
                self.decision_matrix,
                self.rope,
                self.rope_mode,
                self.effect_size,
                self.force_mode,
            )
        )


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
                studentized_range.ppf(1 - 0.05, k=len(estimates), df=contrasts.DF.min())
                / np.sqrt(2)
                * contrasts.SE.min()
            ),
            (
                studentized_range.ppf(1 - 0.05, k=len(estimates), df=contrasts.DF.max())
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


def ci_plot(result, reverse, width, system_id="algorithm", ax=None, title=None):
    """
    Uses error bars to create a plot of the confidence intervals of the mean value.
    """
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
        sorted_means = sorted_df.meanrank
        ci_lower = sorted_df.ci_lower
        ci_upper = sorted_df.ci_upper
        names = sorted_df.index
        alpha = result.alpha
        if reverse:
            sorted_df = result.rankdf.iloc[::-1]
        else:
            print(result)
        sorted_df = result.rankdf
        height = len(sorted_df)
        # cd = [result.cd]

    else:
        print("LMEM")
        result = list(result)
        estimates = result[0].set_index(system_id)
        estimates = estimates.sort_values(by="Estimate")
        sorted_ranks = pd.DataFrame()
        sorted_ranks = estimates["Estimate"]
        sorted_ranks.name = "meanrank"
        estimates["ci_upper"] = estimates["2.5_ci"]
        estimates["ci_lower"] = estimates["97.5_ci"]
        names = estimates.index.values.tolist()
        names_con = [name if "+" not in name else f"({name})" for name in names]
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
                group = [names_con.index(algos[0]), names_con.index(algos[1])]
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
        # hsd = [
        #     (
        #         studentized_range.ppf(1 - 0.05, k=len(estimates), df=contrasts.DF.min())
        #         / np.sqrt(2)
        #         * contrasts.SE.min()
        #     ),
        #     (
        #         studentized_range.ppf(1 - 0.05, k=len(estimates), df=contrasts.DF.max())
        #         / np.sqrt(2)
        #         * contrasts.SE.max()
        #     ),
        # ]

        sorted_means = sorted_ranks
        ci_lower = estimates.ci_lower
        ci_upper = estimates.ci_upper
        names = names
        alpha = 0.05

        height = len(sorted_ranks)

    if ax is None:
        fig = plt.figure(figsize=(width, height))
        fig.set_facecolor("white")
        ax = plt.gca()
    ax.errorbar(
        sorted_means,
        range(len(sorted_means)),
        xerr=abs((ci_upper[0] - ci_lower[0]) / 4),
        marker="o",
        linestyle="None",
        color="k",
        ecolor="k",
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(names))
    if title:
        ax.set_title(title)
    else:
        ax.set_title("%.1f%% Confidence Intervals of the Mean" % ((1 - alpha) * 100))
    return ax


class model_builder:
    def __init__(
        self,
        df: pd.DataFrame,
        loss_var: str = "value",
        system_var="algorithm",
        benchmark_var="benchmark",
    ):
        self.df = df
        self.loss_formula = f"{loss_var} ~ "
        self.exploratory_var = system_var
        self.benchmark_var = benchmark_var

    def test_seed_dependency(self, verbose: bool = True):
        simpel_model = model(
            formula=f"{self.loss_formula}+{self.exploratory_var}",
            data=self.df,
            factor_list=[self.exploratory_var],
            dummy=False,
            no_warnings=True,
        )
        seed_model = model(
            formula=f"{self.loss_formula}+(0+{self.exploratory_var}|seed)",
            data=self.df,
            factor_list=[self.exploratory_var],
            dummy=False,
            no_warnings=True,
        )
        test_result = glrt(
            simpel_model,
            seed_model,
            names=["Simple model", "Model with Seed-effect"] if verbose else None,
            returns=True,
        )
        if test_result["p"] < 0.05 and seed_model.logLike > simpel_model.logLike:
            ranef_var = seed_model.ranef_var
            influenced = ranef_var.loc[
                (ranef_var["Var"] / 10 >= ranef_var["Var"].min())
                & (ranef_var.index != "Residual")
                & (ranef_var["Var"] * 10 >= ranef_var["Var"].max())
            ]["Name"].to_list()
            influenced = [x.rsplit(self.exploratory_var, 1)[1] for x in influenced]
            print(
                f"Seed is a significant effect, likely influenced algorithms: {influenced}"
            )
            return influenced
        else:
            print("=> Seed is not a significant effect")
            return []

    def test_benchmark_information(
        self, rank_benchmarks: bool = False, verbose: bool = True
    ):
        test_results = {}
        benchmark_info = {}
        for benchmark in self.df[self.benchmark_var].unique():
            simple_mod = model(
                formula=f"{self.loss_formula}1",
                data=self.df.loc[self.df[self.benchmark_var] == benchmark],
                factor_list=[self.exploratory_var],
                dummy=False,
                no_warnings=True,
            )
            benchmark_mod = model(
                formula=f"{self.loss_formula}{self.exploratory_var}",
                data=self.df.loc[self.df[self.benchmark_var] == benchmark],
                factor_list=[self.exploratory_var],
                dummy=False,
                no_warnings=True,
            )
            if verbose:
                print(f"\nBenchmark: {benchmark}")
            test_results[benchmark] = glrt(
                simple_mod,
                benchmark_mod,
                names=["Simple model", "Model with Algorithm-effect"]
                if verbose
                else None,
                returns=True,
            )
            if (
                test_results[benchmark]["p"] < 0.05
                and benchmark_mod.logLike > simple_mod.logLike
            ):
                print(
                    f"=> Benchmark {benchmark:<{max([len(x) for x in self.df[self.benchmark_var].unique()])}} is informative."
                )
                benchmark_info[benchmark] = True
            else:
                print(
                    f"=> Benchmark {benchmark:<{max([len(x) for x in self.df[self.benchmark_var].unique()])}} is uninformative."
                )
                benchmark_info[benchmark] = False

        if rank_benchmarks:
            all_benchmarks_mod = model(
                formula=f"{self.loss_formula}(0+{self.benchmark_var}|{self.exploratory_var})",
                data=self.df,
                factor_list=[self.exploratory_var],
                dummy=False,
            )
            print("")
            ranef_var = all_benchmarks_mod.ranef_var[:-1]

            def rename_var_name(row):
                return row["Name"].rsplit(self.benchmark_var, 1)[1]

            ranef_var["Name"] = ranef_var.apply(rename_var_name, axis=1)
            print(ranef_var.reset_index(drop=True))

            # Plot functionality is not useful for larger variations

            # names, ranks = [ranef_var["Name"].to_list(), ranef_var["Var"].to_list()]
            # x_pos = [0.2] * len(names)
            # plt.figure(figsize=(0.4 + 0.1 * max(len(x) for x in names), 3))
            # plt.scatter(x_pos, ranks, facecolors="none", edgecolors="black")
            # for i, name in enumerate(names):
            #     plt.text(
            #         0.4,
            #         ranks[i],
            #         name,
            #     )
            # plt.ylabel("Variance")
            # plt.title("Variance Ranking")
            # plt.ylim(
            #     min(ranks) - 0.1 * (max(ranks) - min(ranks)),
            #     max(ranks) + 0.1 * (max(ranks) - min(ranks)),
            # )
            # plt.xlim(0, 0.5 + max(len(x) for x in names) / 10)
            # plt.xticks([])
            # plt.show()
            uninformative = ranef_var.loc[
                (ranef_var["Var"] * 10 <= ranef_var["Var"].max())
                & (ranef_var.index != "Residual")
                & (ranef_var["Var"] / 10 <= ranef_var["Var"].min())
            ]["Name"].to_list()
            print(
                f"Benchmarks without algorithm variation: {[x.rsplit(self.benchmark_var,1)[0] for x in uninformative]}"
            )
            return benchmark_info, ranef_var
        return benchmark_info

    def test_fidelity(self, fidelity_var: str, verbose: bool = True):
        significances = {fidelity_var: 0, f"{fidelity_var}_group": 0}
        simple_formula = f"{self.loss_formula} {self.exploratory_var}{f' + (1|{self.benchmark_var})' if self.df[self.benchmark_var].nunique()>1 else ''}"
        simple_mod = model(
            formula=simple_formula,
            data=self.df,
            factor_list=[self.exploratory_var],
            dummy=self.df[self.benchmark_var].nunique() == 1,
            no_warnings=True,
        )
        fidelity_mod = model(
            formula=f"{simple_formula} + {fidelity_var}",
            data=self.df,
            factor_list=[self.exploratory_var],
            dummy=self.df[self.benchmark_var].nunique() == 1,
            no_warnings=True,
        )
        test_result = glrt(
            simple_mod,
            fidelity_mod,
            names=["Simple model", "Model with Fidelity-effect"] if verbose else None,
            returns=True,
        )
        if verbose:
            print("")
        if test_result["p"] < 0.05 and fidelity_mod.logLike > simple_mod.logLike:
            significances[fidelity_var] = 1
        fid_group_mod = model(
            formula=f"{simple_formula} + {self.exploratory_var}:{fidelity_var}",
            data=self.df,
            factor_list=[self.exploratory_var],
            dummy=self.df[self.benchmark_var].nunique() == 1,
        )
        test_result = glrt(
            simple_mod,
            fid_group_mod,
            names=["Simple model", "Model with Fidelity-interaction-effect"]
            if verbose
            else None,
            returns=True,
        )
        if test_result["p"] < 0.05 and fid_group_mod.logLike > simple_mod.logLike:
            significances[f"{fidelity_var}_group"] = 1
        if (
            significances[fidelity_var] == 1
            and significances[f"{fidelity_var}_group"] == 1
        ):
            if verbose:
                print("")
            test_result = glrt(
                fidelity_mod,
                fid_group_mod,
                names=[
                    "Model with Fidelity-effect",
                    "Model with Fidelity-interaction-effect",
                ]
                if verbose
                else None,
                returns=True,
            )
            if verbose:
                print("")
            if test_result["p"] < 0.05 and fid_group_mod.logLike > fidelity_mod.logLike:
                print(
                    f"=> Fidelity {fidelity_var} is both as simple and interaction effect significant, but interaction effect performs better."
                )
            else:
                print(
                    f"=> Fidelity {fidelity_var} is both as simple and interaction effect significant, but as simple effect performs better."
                )
        elif significances[fidelity_var] == 1:
            print(f"=> Fidelity {fidelity_var} as simple effect is significant.")
        elif significances[f"{fidelity_var}_group"] == 1:
            print(f"=>  Fidelity {fidelity_var} as interaction effect is significant.")
        else:
            print(f"=> Fidelity {fidelity_var} is not a significant effect.")
