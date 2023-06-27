from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from experiment_analysis.exceptions import (
    InvalidInputDataframe, InvalidIsotonicMethod
)


class IsotonicMethod(Enum):
    STANDARD = "STANDARD"
    DEMEAN = "DEMEAN"


def apply_isotonic(
    df: pd.DataFrame,
    metric: str,
    metric_cv: str,
    method: Union[str, IsotonicMethod] = IsotonicMethod.STANDARD,
) -> pd.Series:
    """Apply Isotonic Regression to compute isotnoic metric based on experiment metric and pre-experiment covariate

    :param df: pd.Dataframe
    :param metric: str Name of dataframe column containing experiment metric
    :param metric_cv: str Name of dataframe column containing pre-experiment covariates
    :param method: Union[str, IsoraMethod] Variation on IsoraMethod method.

    :return: Metric with Isotonic Regression applied to it
    :rtype: pd.Series
    """
    user_method: IsotonicMethod

    if isinstance(method, str):
        user_method = IsotonicMethod(method.upper())
    else:
        user_method = IsotonicMethod(method)

    if metric not in df.columns:
        raise InvalidInputDataframe(
            f"input dataframe is missing metric column {metric!r}"
        )

    if metric_cv not in df.columns:
        raise InvalidInputDataframe(
            f"input dataframe is missing cv metric column {metric_cv!r}"
        )

    if user_method not in IsotonicMethod:
        raise InvalidIsotonicMethod(
            f"input isotonic regression method is invalid. Must be one of {IsotonicMethod!r}"
        )

    # train isotonic regression
    iso_reg = (
        IsotonicRegression(increasing='auto')
        .fit(df[metric_cv], df[metric])
    )

    # calculate isotonic-regressed metric
    resid = df[metric] - iso_reg.predict(df[metric_cv]) # get resid

    if user_method == IsotonicMethod.STANDARD:
        iso_metric = resid + np.mean(df[metric]) # get isotonic-regressed metric
    elif user_method == IsotonicMethod.DEMEAN:
        iso_metric = resid # get demeaned isotonic metric

    return iso_metric
