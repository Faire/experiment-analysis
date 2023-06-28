from enum import Enum
from typing import Union

import numpy as np
import pandas as pd

from experiment_analysis.exceptions import (
    InvalidInputDataframe,
    InvalidParameter,
)


class CappingMethod(Enum):
    STANDARD = "STANDARD"
    WITH_CUPED = "WITH_CUPED"
    WITH_ISOTONIC = "WITH_ISOTONIC"


def apply_capping(
    df: pd.DataFrame,
    metric: str,
    sd_threshold: int = 5,
    method: Union[str, CappingMethod] = CappingMethod.STANDARD,
    metric_raw: str = None,
) -> pd.Series:
    """Calculates metrics capped at threshold of given SDs from mean (upper bound capping only). Intended for non-negative metrics.

    :param df: pd.Dataframe
    :param metric: str Name of dataframe column containing experiment metric to cap
    :param sd_threshold: sd_threshold Number of standard deviations from mean to set as outlier threshold. Default is 5.
    :param method: Union[str, CappingMethod] If "STANDARD" assumed non-CUPED metric. Use with "WITH_CUPED" if want to cap a CUPED metric.
    :param metric_raw: str Name of dataframe column containing metric before CUPED transformation if applying capping to a CUPED metric. Used in "WITH_CUPED" method.
    :return: Capped metric
    :rtype: pd.Series
    """
    user_method: CappingMethod

    if isinstance(method, str):
        user_method = CappingMethod(method.upper())
    else:
        user_method = CappingMethod(method)

    if metric not in df.columns:
        raise InvalidInputDataframe(
            f"input dataframe is missing metric column {metric!r}"
        )

    if not isinstance(sd_threshold, int):
        raise InvalidParameter(f"sd_threshold must be integer")

    if user_method == CappingMethod.STANDARD:
        metric_mean = df.loc[df[metric] > 0][metric].mean()
        metric_stdev = df.loc[df[metric] > 0][metric].std()
    elif user_method == CappingMethod.WITH_CUPED or user_method == CappingMethod.WITH_ISOTONIC:

        if metric_raw not in df.columns:
            raise InvalidInputDataframe(
                f"input dataframe is missing raw metric column {metric_raw!r}"
            )

        metric_mean = df.loc[df[metric_raw] > 0][metric].mean()
        metric_stdev = df.loc[df[metric_raw] > 0][metric].std()

    upper_bound = metric_mean + sd_threshold * metric_stdev
    capped_metric = pd.Series(
        np.where(df[metric] > upper_bound, upper_bound, df[metric])
    )

    return capped_metric
