import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from experiment_analysis.exceptions import (
    InvalidInputDataframe
)


def apply_isotonic(
    df: pd.DataFrame,
    metric: str,
    metric_cv: str,
) -> pd.Series:
    """Apply Isotonic Regression to compute isotnoic metric based on experiment metric and pre-experiment covariate

    :param df: pd.Dataframe
    :param metric: str Name of dataframe column containing experiment metric
    :param metric_cv: str Name of dataframe column containing pre-experiment covariates

    :return: Metric with Isotonic Regression applied to it
    :rtype: pd.Series
    """

    if metric not in df.columns:
        raise InvalidInputDataframe(
            f"input dataframe is missing metric column {metric!r}"
        )

    if metric_cv not in df.columns:
        raise InvalidInputDataframe(
            f"input dataframe is missing cv metric column {metric_cv!r}"
        )

    # train isotonic regression
    iso_reg = (
        IsotonicRegression(increasing='auto')
        .fit(df[metric_cv], df[metric])
    )

    # calculate isotonic-regressed metric
    resid = df[metric] - iso_reg.predict(df[metric_cv]) # get resid
    iso_metric = resid + np.mean(df[metric]) # get isotonic-regressed metric

    return iso_metric
