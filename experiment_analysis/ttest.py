import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.stats.api as sms


def summarize_ttest(df, control_name, treatment_name, metric, confidence_level=0.9):
    """Helper function to summarize
    results of t-test"""

    alpha = 1 - confidence_level

    treat_dat = df[df['bucket_name'] == treatment_name][metric]
    control_dat = df[df['bucket_name'] == control_name][metric]
    control_mean = control_dat.mean()
    treatment_mean = treat_dat.mean()

    treat_stats = sms.DescrStatsW(treat_dat)
    control_stats = sms.DescrStatsW(control_dat)

    cm = sms.CompareMeans(treat_stats, control_stats)
    ci = cm.tconfint_diff(alpha=alpha)
    delta = (ci[1] - ci[0]) / 2 + ci[0]
    cm_stats = cm.ttest_ind(usevar='unequal')

    se_delta = delta / cm_stats[0]
    se_lift = np.abs(se_delta / control_mean)
    t_crit = stats.t.ppf(1 - alpha / 2.0, df=cm_stats[2])
    pct_lift = delta / control_mean
    ci_lift_lower = pct_lift - t_crit * se_lift
    ci_lift_upper = pct_lift + t_crit * se_lift

    results = {'delta': "{0:.3}".format(delta),
               'control_mean': "{0:.4}".format(control_mean),
               'treatment_mean': "{0:.4}".format(treatment_mean),
               'pct_lift': "{0:.3%}".format(pct_lift),
               'ci_delta_lower': "{0:.3}".format(ci[0]),
               'ci_delta_upper': "{0:.3}".format(ci[1]),
               'ci_lift_lower': "{0:.3%}".format(ci_lift_lower),
               'ci_lift_upper': "{0:.3%}".format(ci_lift_upper),
               'p_val': "{0:.3}".format(cm_stats[1])}

    return (results)
