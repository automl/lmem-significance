"""Imports for analysis method"""
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymer4.models import Lmer
from scipy import stats


def glrt(mod1: Lmer, mod2: Lmer) -> dict[str, typing.Any]:
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
    return {
        "chi_square": chi_square,
        "df": delta_params,
        "p": 1 - stats.chi2.cdf(chi_square, df=delta_params),
    }


def conduct_analysis(
    data: pd.DataFrame,
    metric: str,
    system_id: str,
    input_id: typing.Optional[str] = None,
    bin_id: typing.Optional[str] = None,
    bins: typing.Optional[typing.Union[list[list[str]], list[float], float]] = None,
    bin_labels: typing.Optional[list[str]] = None,
    subset: typing.Optional[
        typing.Union[
            str,
            typing.Tuple[
                str, typing.Union[dict[str, typing.Any], str, list[str], list[list[str]]]
            ],
        ]
    ] = None,
    show_plots: bool = True,
    verbosity: int = 2,
    show_contrasts: bool = True,
) -> typing.Union[
    typing.Union[
        typing.Tuple[
            dict[str, typing.Any],
            typing.Union[typing.Any, typing.Tuple[typing.Any, pd.DataFrame]],
        ],
        dict[str, typing.Any],
    ],
    dict[
        str,
        typing.Union[
            typing.Tuple[
                dict[str, typing.Any],
                typing.Union[typing.Any, typing.Tuple[typing.Any, pd.DataFrame]],
            ],
            dict[str, typing.Any],
        ],
    ],
]:
    """LMER-Based Performance Analysis

    Args:
        data (pd.DataFrame): Dataset
        metric (str): Column name of metric (e.g. Mean) in dataset
        system_id (str): Column name of system (e.g. Algorithm) in dataset
        input_id (str): Column name of input (e.g. Benchmark) in dataset
        bin_id (str, optional): Column name of bin (e.g. Budget). Defaults to None.
        bins (typing.Union[list[list[str]], list[float], float], optional): Specified bins: If None, bins for every unique value are used. If list of float, numeric variable gets binned into intervals according to list. If list of list of str, variable gets binned into bins according to sublists. Defaults to None.
        bin_labels (list[str], optional): Labels for bins. If None, bins are named after content/interval borders. If list of str, bins are named according to list. Defaults to None.
        subset (typing.Union[str,typing.Tuple[str, typing.Union[dict[str, any], str, list[str], list[list[str]]]]], optional): Subset of dataset that should be used for analysis. Only name of variable that defines subset to create subgroup for each entry or: First entry of tuple is name of variable, second entry is either list of entries that should be analysed iteratively (single entries or groups), or "all"/"a" for all entries. Defaults to None.
        show_plots (bool, optional): Show plots. First entry is boxplot comparing systems, second entry is graph showing systems in all bins. Defaults to True.
        verbosity (int, optional): Verbosity of output while analysing. Defaults to 2, other levels are 1 and 0.
        show_contrasts (bool, optional): Develop contrasts between systems
        significance_plot (str, optional): Plot significance of contrasts to this value

    Raises:
        SystemExit: If Subset-Value is not in Dataset.
        SystemExit: If the number of Labels does not fit the number of numeric Bins.
        SystemExit: If the number of Labels does not fit the number of categorical Bins.
        SystemExit: If the value for significance plot is an entry of the system_id.

    Returns:
        typing.Union[typing.Union[typing.Tuple[dict[str,typing.Any],typing.Union[typing.Any,typing.Tuple[typing.Any,pd.DataFrame]]],dict[str,typing.Any]],dict[str,typing.Union[typing.Tuple[dict[str,typing.Any],typing.Union[typing.Any,typing.Tuple[typing.Any,pd.DataFrame]]],dict[str,typing.Any]]]]: Result tuple, first the result-dictionary of the GLRT and second the post_hoc-analysis of the LMEM, consisting of first the estimated means for each system and second the contrasts. If a subsets are used, returns dictionary with full results for each subset. If contrasts are turned off, only returns estimated means as second tuple-entry. If post-hoc-analysis failed, only returns GLRT-Results.
    """

    if subset is not None:
        if isinstance(subset, str):
            subset = (subset, "a")
        if isinstance(subset[1], (str, dict)):
            if isinstance(subset[1], dict):
                new_dict = {}
                for key, value in subset[1].items():
                    if value not in new_dict:
                        new_dict[value] = [key]
                    else:
                        if value in new_dict:
                            new_dict[value].append(key)
                subset_list = list(new_dict.values())
            elif subset[1] in ["all", "a", "All", "A"]:
                subset_list = list(data[subset[0]].unique())
            else:
                subset_list = [subset[1]]
        else:
            subset_list = subset[1]
        return_dict = {}
        for subset_item in subset_list:
            if verbosity > 0:
                print(f"Analysis for {subset_item}")
            if isinstance(subset_item, str):
                subset_item = [subset_item]
            if any(item not in data[subset[0]].unique() for item in subset_item):
                raise SystemExit(
                    f"A Subset-Value of {subset_item} is not in Dataset. Choose from {data[subset[0]].unique()}"
                )
            return_dict["__".join(subset_item)] = conduct_analysis(
                data.loc[data[subset[0]].isin(subset_item)],
                metric,
                system_id,
                input_id,
                bin_id,
                bins=bins,
                bin_labels=bin_labels,
                show_plots=show_plots,
                verbosity=verbosity,
                show_contrasts=show_contrasts,
            )
        return return_dict

    pd.set_option("chained_assignment", None)
    pd.set_option("display.max_rows", 5000)
    pd.set_option("display.max_columns", 5000)
    pd.set_option("display.width", 10000)

    if not bin_id:
        if not input_id:
            input_id = "input_id_dummy"
            data[input_id] = "d"
        if len(data[input_id].unique()) == 1:
            data.loc[data.sample(1).index, input_id] = data[input_id].unique()[0] + "_d"

        if show_plots:
            _, axis = plt.subplots()
            axis.boxplot(
                [group[metric] for _, group in data.groupby(system_id)],
                labels=list(data[system_id].unique()),
            )
            plt.yscale("log")
            plt.xticks(rotation=-45, ha="right")
            plt.show()

        # System-identifier: system_id
        # Input-Identifier: input_id
        # Two models, "different"-Model assumes significant difference between performance of groups, divided by system-identifier
        # Formula has form: "metric ~ system_id + (1 | input_id)"
        different_means_model = Lmer(
            formula=f"{metric}~{system_id}+(1|{input_id})", data=data
        )

        # factors specifies names of system_identifier, i.e. Baseline, or Algorithm1
        different_means_model.fit(
            factors={system_id: list(data[system_id].unique())},
            REML=False,
            summarize=False,
        )

        # "Common"-Model assumes no significant difference, which is why the system-identifier is not included
        common_mean_model = Lmer(formula=f"{metric}~ (1 | {input_id})", data=data)
        common_mean_model.fit(REML=False, summarize=False)

        # Signficant p-value shows, that different-Model fits data sign. better, i.e.
        # There is signficant difference in system-identifier
        result_glrt_dm_cm = glrt(different_means_model, common_mean_model)
        p_value = result_glrt_dm_cm["p"]

        if verbosity > 0:
            print(f"P-value: {p_value}")
            if result_glrt_dm_cm["p"] < 0.05:
                print(
                    f"\nAs the p-value {p_value} is smaller than 0.05, we can reject the Null-Hypothesis that the model "
                    f"that does not consider the {system_id} describes the data as well as the one that does. Therefore "
                    f"there is significant difference within {system_id}.\n"
                )
            else:
                print(
                    f"\nAs the p-value {p_value} is not smaller than 0.05, we cannot reject the Null-Hypothesis that the model "
                    f"that does not consider the {system_id} describes the data as well as the one that does. Therefore "
                    f"there is no significant difference within {system_id}\n."
                )

        # Post hoc divides the "different"-Model into its three systems
        post_hoc_results = different_means_model.post_hoc(marginal_vars=[system_id])
        if post_hoc_results:
            contrasts = post_hoc_results[1]
            for pair in contrasts["Contrast"]:
                contrasts.loc[
                    contrasts["Contrast"] == pair, system_id + "_1"
                ] = pair.split(" - ")[0]
                contrasts.loc[
                    contrasts["Contrast"] == pair, system_id + "_2"
                ] = pair.split(" - ")[1]
            contrasts = contrasts.drop("Contrast", axis=1)
            column = contrasts.pop(system_id + "_2")
            contrasts.insert(0, system_id + "_2", column)
            column = contrasts.pop(system_id + "_1")
            contrasts.insert(0, system_id + "_1", column)

            if verbosity > 1:
                # [0] shows group-means, i.e. performance of the single system-groups
                print(post_hoc_results[0])  # cell (group) means
                # [1] shows the pairwise comparisons, i.e. improvements over each other, with p-value
                if show_contrasts:
                    print(contrasts)  # contrasts (group differences)

            best_system_id = post_hoc_results[0].loc[
                post_hoc_results[0]["Estimate"].idxmin()
            ][system_id]

            contenders = []
            for _, row in contrasts.iterrows():
                if row[system_id + "_1"] == best_system_id and not row["Sig"] in [
                    "*",
                    "**",
                    "***",
                ]:
                    contenders.append(row[system_id + "_2"])
                if row[system_id + "_2"] == best_system_id and not row["Sig"] in [
                    "*",
                    "**",
                    "***",
                ]:
                    contenders.append(row[system_id + "_1"])
            if verbosity > 0:
                if contenders:
                    print(
                        f"The best performing {system_id} is {best_system_id}, but {contenders} are only insignificantly worse.\n"
                    )
                else:
                    print(
                        f"The best performing {system_id} is {best_system_id}, all other perform significantly worse.\n"
                    )

            if show_contrasts:
                return result_glrt_dm_cm, post_hoc_results
            return result_glrt_dm_cm, post_hoc_results[0]
        return result_glrt_dm_cm

    if not any(isinstance(element, str) for element in data[bin_id]):
        if not bins:
            complete_bins = sorted(list(data[bin_id].unique()))
            bin_labels = [str(number) for number in complete_bins]
            complete_bins = complete_bins + [max(complete_bins) + 1]
        else:
            if isinstance(bins, (float, int)):
                bins = np.round(
                    np.linspace(data[bin_id].min(), data[bin_id].max(), bins + 1),
                    decimals=2,
                )
            bins_set = set(bins)
            bins_set.add(data[bin_id].min())
            bins_set.add(data[bin_id].max())
            complete_bins = sorted(list(set(bins_set)))
            if bin_labels is None:
                bin_labels = [
                    f"{complete_bins[i]}_{complete_bins[i+1]}"
                    for i in range(len(complete_bins) - 1)
                ]
            else:
                if any(isinstance(x, (float, int)) for x in bin_labels):
                    bin_labels = [str(number) for number in bin_labels]
                if len(bin_labels) != len(bins) + 1:
                    raise SystemExit(
                        f"Too many or too few labels ({len(bin_labels)} labels and {len(complete_bins)} bins)"
                    )
        data[f"{bin_id}_bins"] = pd.cut(
            data[bin_id],
            bins=complete_bins,
            labels=bin_labels,
            include_lowest=True,
        )

    else:
        if bins:
            mapping = {x: index for index, sublist in enumerate(bins) for x in sublist}
            data[f"{bin_id}_coded"] = data[bin_id].apply(lambda x: mapping[x])
            if bin_labels:
                if len(bin_labels) != len(bins):
                    raise SystemExit(
                        f"Too many or too few labels ({len(bin_labels)} labels and {len(bins)} bins)"
                    )
                bin_labels = [str(number) for number in bin_labels]
            else:
                bin_labels = list(data[bin_id].unique())
            data[f"{bin_id}_bins"] = pd.cut(
                data[f"{bin_id}_coded"],
                bins=len(bins),
                labels=bin_labels,
                include_lowest=True,
            )
        else:
            data[f"{bin_id}_coded"], _ = pd.factorize(data[bin_id])
            bin_labels = list(data[bin_id].unique())
            bin_labels = [str(x) for x in bin_labels]
            data[f"{bin_id}_bins"] = pd.cut(
                data[f"{bin_id}_coded"],
                bins=len(list(data[f"{bin_id}_coded"].unique())),
                labels=bin_labels,
                include_lowest=True,
            )
        data = data.drop(f"{bin_id}_coded", axis=1)

    # New model "expanded": Divides into system AND bin-classes (Term system:bin_id allows for Cartesian Product, i.e. different Mean for each system and bin-class)
    model_expanded = Lmer(
        f"{metric} ~  {system_id} + {bin_id}_bins + {system_id}:{bin_id}_bins + (1 | {input_id})",
        data=data,
    )
    model_expanded.fit(
        factors={
            system_id: list(data[system_id].unique()),
            f"{bin_id}_bins": list(data[f"{bin_id}_bins"].unique()),
        },
        REML=False,
        summarize=False,
    )
    # Second model "nointeraction" lacks system:src-Term to hypothesise no interaction, i.e. no difference when changing bin-class
    model_nointeraction = Lmer(
        f"{metric} ~ {system_id} + {bin_id}_bins + (1 | {input_id})",
        data=data,
    )
    model_nointeraction.fit(
        factors={
            system_id: list(data[system_id].unique()),
            f"{bin_id}_bins": list(data[f"{bin_id}_bins"].unique()),
        },
        REML=False,
        summarize=False,
    )

    # If it's significant, look at if different systems perform better at different bin-classes
    result_glrt_ex_ni = glrt(model_expanded, model_nointeraction)
    p_value = result_glrt_ex_ni["p"]
    if verbosity > 0:
        print(f"P-value: {p_value}")
        if p_value < 0.05:
            print(
                f"\nAs the p-value {p_value} is smaller than 0.05, we can reject the Null-Hypothesis that the model "
                f"that does not consider the {system_id} and the {bin_id} describes the data as well as the one that does. Therefore "
                f"there is significant difference within {system_id} and the {bin_id}.\n"
            )
        else:
            print(
                f"\nAs the p-value {p_value} is not smaller than 0.05, we cannot reject the Null-Hypothesis that the model "
                f"that does not consider the {system_id} and the {bin_id} describes the data as well as the one that does. Therefore "
                f"there is no significant difference within {system_id} and {bin_id}\n."
            )

    post_hoc_results = model_expanded.post_hoc(
        marginal_vars=system_id, grouping_vars=f"{bin_id}_bins"
    )
    if post_hoc_results:
        if verbosity > 1:
            # Means of each combination
            print(post_hoc_results[0])
        if show_contrasts:
            # Comparisons for each combination
            contrasts_collection = pd.DataFrame()
            for group in data[f"{bin_id}_bins"].unique():
                contrasts = post_hoc_results[1].query(f"{bin_id}_bins == '{group}'")

                for pair in contrasts["Contrast"]:
                    contrasts.loc[
                        contrasts["Contrast"] == pair, system_id + "_1"
                    ] = pair.split(" - ")[0]
                    contrasts.loc[
                        contrasts["Contrast"] == pair, system_id + "_2"
                    ] = pair.split(" - ")[1]
                # contrasts = contrasts.drop("Contrast", axis=1)
                column = contrasts.pop(system_id + "_2")
                contrasts.insert(0, system_id + "_2", column)
                column = contrasts.pop(system_id + "_1")
                contrasts.insert(0, system_id + "_1", column)
                contrasts_collection = pd.concat([contrasts_collection, contrasts])
                if verbosity > 1:
                    print(contrasts[contrasts["Sig"] != ""])
                best_system_id = (
                    post_hoc_results[0]
                    .query(f"{bin_id}_bins == '{group}'")
                    .loc[
                        post_hoc_results[0]
                        .query(f"{bin_id}_bins == '{group}'")["Estimate"]
                        .idxmin()
                    ][system_id]
                )
                contenders = []
                for _, row in contrasts.iterrows():
                    if row[system_id + "_1"] == best_system_id and not row["Sig"] in [
                        "*",
                        "**",
                        "***",
                    ]:
                        contenders.append(row[system_id + "_2"])
                    if row[system_id + "_2"] == best_system_id and not row["Sig"] in [
                        "*",
                        "**",
                        "***",
                    ]:
                        contenders.append(row[system_id + "_1"])
                if verbosity > 0:
                    if contenders:
                        print(
                            f"The best performing {system_id} in {bin_id}-class {group} is {best_system_id}, but {contenders} are only insignificantly worse.\n"
                        )
                    else:
                        print(
                            f"The best performing {system_id} in {bin_id}-class {group} is {best_system_id}, all other perform significantly worse.\n"
                        )
            if show_plots:
                _, axis = plt.subplots(figsize=(10, 6))
                for sys_id, group in post_hoc_results[0].groupby(system_id):
                    axis.errorbar(
                        group[f"{bin_id}_bins"],
                        group["Estimate"],
                        yerr=group["SE"],
                        fmt="o-",
                        capsize=3,
                        label=sys_id,
                        lolims=group["2.5_ci"],
                        uplims=group["97.5_ci"],
                    )
                axis.set_xlabel(bin_id)
                axis.set_ylabel("Estimate")
                axis.set_title(f"Estimates by {system_id} and {bin_id}")
                axis.legend()
                plt.show()

        if show_contrasts:
            return result_glrt_ex_ni, post_hoc_results
        return result_glrt_ex_ni, post_hoc_results[0]
    return result_glrt_ex_ni
