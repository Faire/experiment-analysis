import math
import numpy as np
import pandas as pd
import statsmodels.stats.power as pwr


def calculate_samples(
    metric_mean: float,
    metric_var: float,
    mde: float,
    treatment_pct: float = 0.5,
    confidence_level: float = 0.9,
    power: float = 0.8,
):

    """Calculates total samples needed to reach power given metric mean, variance and minimum detectable effect (mde).

    :param metric_mean: float Mean of metric
    :param metric_var: float Variance of metric
    :param mde: float Minimum detectable effect
    :param treatment_pct: float Percentage of total samples in treatment group
    :param confidence_level: float Level of confidence interval to be used. equal to 1 - signficance level
    :param power: float Statistical power of test. Probability of detecting a true effect.
    :rtype:"""

    effect_size = metric_mean * mde / np.sqrt(metric_var)
    treatment_samples = pwr.tt_ind_solve_power(
        effect_size=effect_size,
        alpha=1 - confidence_level,
        power=power,
        ratio=(1 - treatment_pct) / treatment_pct,
    )
    total_samples = treatment_samples / treatment_pct
    return math.ceil(total_samples)


def calculate_mde(
    metric_mean: float,
    metric_var: float,
    total_samples: int,
    treatment_pct: float = 0.5,
    confidence_level: float = 0.9,
    power: float = 0.8,
):

    """Calculates minimum detectable effect at desired powered level given metric mean, variance and total samples.

    :param metric_mean: float Mean of metric
    :param metric_var: float Variance of metric
    :param total_samples: int Total number of samples across both treatment and control groups.
    :param treatment_pct: float Percentage of total samples in treatment group.
    :param confidence_level: float Level of confidence interval to be used. Equal to 1 - signficance level
    :param power: float Statistical power of test. Probability of detecting a true effect.
    :rtype:"""

    treatment_obs = total_samples * treatment_pct
    bucket_ratio = (1 - treatment_pct) / treatment_pct
    # Function calculates Cohen's d (delta/stdev)
    d = pwr.tt_ind_solve_power(
        nobs1=treatment_obs, alpha=1 - confidence_level, power=power, ratio=bucket_ratio
    )

    # Transform Cohen's d to relative lift
    return d * math.sqrt(metric_var) / metric_mean
