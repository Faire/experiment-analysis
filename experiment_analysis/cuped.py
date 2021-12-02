from enum import Enum
from typing import Union

import numpy as np
import pandas as pd

from experiment_analysis.exceptions import (
    InvalidInputDataframe,
)


class CupedMethod(Enum):
    STANDARD = "STANDARD"
    DEMEAN = "DEMEAN"


def apply_cuped(
    df: pd.DataFrame,
    metric: str,
    metric_cv: str,
    method: Union[str, CupedMethod] = CupedMethod.DEMEAN,
) -> pd.Series:
    """Calculate CUPED metric based on experiment metric and pre-experiment covariate.

    :param df: pd.Dataframe
    :param metric: str Name of dataframe column containing experiment metric
    :param metric_cv: str Name of dataframe column containing pre-experiment covariates
    :param method: Union[str, CupedMethod] Variation on CUPED method using demeaned covariate or standard CUPED.

    :return: Metric with CUPED applied to it
    :rtype: pd.Series
    """
    user_method: CupedMethod

    if isinstance(method, str):
        user_method = CupedMethod(method.upper())
    else:
        user_method = CupedMethod(method)

    if metric not in df.columns:
        raise InvalidInputDataframe(
            f"input dataframe is missing metric column {metric!r}"
        )

    if metric_cv not in df.columns:
        raise InvalidInputDataframe(
            f"input dataframe is missing cv metric column {metric_cv!r}"
        )

    # theta is covariance between metric and pre-experiment data divided by variance of pre-experiment data
    theta = (np.cov(df[metric], df[metric_cv]))[1][0] / df[metric_cv].var()
    cv_mean = df[metric_cv].mean()
    if user_method == CupedMethod.DEMEAN:
        cuped_metric = df[metric] - theta * (df[metric_cv] - cv_mean)
    elif user_method == CupedMethod.STANDARD:
        cuped_metric = df[metric] - theta * df[metric_cv]

    return cuped_metric
