import numpy as np
import pandas as pd
import scipy.stats as stats

from experiment_analysis.exceptions import (
    InvalidInputDataframe,
)


def calc_ratio_metric_var(
    df: pd.DataFrame,
    numerator: str,
    denominator: str,
):
    """
    Calculates correct variance of ratio metric using delta method.

    :param df: pd.Dataframe
    :param numerator: str Name of dataframe column containing metric numerator
    :param denominator: str Name of dataframe column containing metric denominator
    :return: Variance of ratio metric
    :rtype: int
    """

    if numerator not in df.columns:
        raise InvalidInputDataframe(
            f"input dataframe is missing numerator column {numerator!r}"
        )

    if denominator not in df.columns:
        raise InvalidInputDataframe(
            f"input dataframe is missing denominator column {denominator!r}"
        )

    n = df.shape[0]
    var_numerator = df[numerator].var()
    mean_numerator = df[numerator].mean()
    var_denominator = df[denominator].var()
    mean_denominator = df[denominator].mean()
    cov_mat = df[[numerator, denominator]].cov()
    covar = cov_mat.iloc[0, 1]
    var_ratio = (
        (1 / (mean_denominator) ** 2) * var_numerator
        + (mean_numerator ** 2 / mean_denominator ** 4) * var_denominator
        - 2 * (mean_numerator / mean_denominator ** 3) * covar
    )

    return var_ratio


def ttest_ratio_metric(
    df: pd.DataFrame,
    numerator: str,
    denominator: str,
    bucket_col: str = "bucket_name",
    treatment_label: str = "treatment",
    control_label: str = "control",
):
    """
    Performs t-test for ratio metric using variance calculated
    by delta method.

    :param df: pd.Dataframe
    :param numerator: str Name of dataframe column containing metric numerator
    :param denominator: str Name of dataframe column containing metric denominator
    :param bucket_col: str Name of dataframe column with treatment and control group labels. Defaults to "bucket_name".
    :param treatment_label: str Name of label for treatment group. Defaults to "treatment".
    :param control_label: str Name of label for control group. Defaults to "control".
    :return: Results of t-test for difference in ratio metric betwen treatment and control
    :rtype: dict
    """

    if numerator not in df.columns:
        raise InvalidInputDataframe(
            f"input dataframe is missing numerator column {numerator!r}"
        )

    if denominator not in df.columns:
        raise InvalidInputDataframe(
            f"input dataframe is missing denominator column {denominator!r}"
        )

    if bucket_col not in df.columns:
        raise InvalidInputDataframe(
            f"input dataframe is missing bucket column {bucket_col!r}"
        )

    if treatment_label not in df[bucket_col].unique():
        raise InvalidInputDataframe(
            f"invalid treatment_label {treatment_label!r}")

    if control_label not in df[bucket_col].unique():
        raise InvalidInputDataframe(f"invalid control_label {control_label!r}")

    n_samples = df.shape[0]
    control_dat = df[df[bucket_col] == control_label]
    treat_dat = df[df[bucket_col] == treatment_label]

    treat_ratio = treat_dat[numerator].sum() / treat_dat[denominator].sum()
    control_ratio = control_dat[numerator].sum(
    ) / control_dat[denominator].sum()

    delta = treat_ratio - control_ratio
    var_treat_ratio_metric = calc_ratio_metric_var(
        treat_dat, numerator, denominator)
    var_control_ratio_metric = calc_ratio_metric_var(
        control_dat, numerator, denominator
    )
    se_delta = np.sqrt(var_treat_ratio_metric + var_control_ratio_metric) / np.sqrt(
        n_samples
    )
    t_stat = delta / se_delta
    p_value = stats.t.sf(np.abs(t_stat), n_samples - 1) * 2

    ttest_results = {
        "treatment_ratio": treat_ratio,
        "control_ratio": control_ratio,
        "delta": delta,
        "se": se_delta,
        "t_stat": t_stat,
        "p_value": p_value,
    }

    return ttest_results
