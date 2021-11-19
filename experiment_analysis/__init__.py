import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
from statsmodels.stats import weightstats
import pandas as pd
import numpy as np
from statsmodels.stats import weightstats
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.graph_objs as go


def estimate_cuped(data, target="brand_orders", ci_scale=1.282,
                   remove_outliers=False, cap=False, print_output=False,
                   demean=False):
    # estimate the theta that minimizes the control variate
    df = data.copy()
    covariate = f"{target}_7D"

    idx_ctrl = df['BUCKET_NAME'] == 'control'
    idx_treat = df['BUCKET_NAME'] == 'test'
    cv = np.sum((df[target] - df[target].mean()) * (df[covariate] - df[covariate].mean()))
    cv /= (df.shape[0] - 1)
    var = np.sum((df[covariate] - df[covariate].mean()) ** 2.0)
    var /= (df.shape[0] - 1)
    theta = cv / var
    if demean:
        df["adjusted"] = df[target] - theta * (df[covariate] - df[covariate].mean())
    else:
        df["adjusted"] = df[target] - theta * df[covariate]

    if remove_outliers:
        df = outlier_removal(df, "adjusted", cap=cap)
        idx_ctrl = df['BUCKET_NAME'] == 'control'
        idx_treat = df['BUCKET_NAME'] == 'test'

    n_treat = np.sum(idx_treat)
    n_ctrl = np.sum(idx_ctrl)
    control_cell_estimate = df["adjusted"].loc[idx_ctrl].mean()
    treatment_cell_estimate = df["adjusted"].loc[idx_treat].mean()
    meandiff = treatment_cell_estimate - control_cell_estimate
    lift = 100 * meandiff / control_cell_estimate

    x1 = df["adjusted"].loc[df.BUCKET_NAME == 'control'].copy()
    x2 = df["adjusted"].loc[df.BUCKET_NAME == 'test'].copy()
    tstat, p, _ = weightstats.ttest_ind(x1, x2, value=meandiff)
    pooled_var = ((x2.shape[0] - 1) * (x2.std() ** 2.0)) + ((x1.shape[0] - 1) * (x1.std() ** 2.0))
    pooled_var /= (x2.shape[0] + x1.shape[0] - 2)
    std_err = np.sqrt(pooled_var / x1.shape[0] + pooled_var / x2.shape[0])
    lower_est = (treatment_cell_estimate - ci_scale * std_err)
    upper_est = (treatment_cell_estimate + ci_scale * std_err)
    output = {
        "metric": target,
        "pct_lift": lift,
        "pvalue": p,
        "samples_control": n_ctrl,
        "samples_treatment": n_treat,
        "std_err": std_err,
        "delta": meandiff,
        "abs_upper": upper_est,
        "abs_lower": lower_est,
        "pct_upper": 100 * (upper_est / control_cell_estimate - 1),
        "pct_lower": 100 * (lower_est / control_cell_estimate - 1)
    }
    return output


def run_regression(data, print_output=False, target="brand_orders", ci_scale=1.28):
    """ Do Variance Reduction """
    df = data.copy()
    idx_ctrl = df['BUCKET_NAME'] == 'control'
    idx_treat = df['BUCKET_NAME'] == 'test'
    n_treat = np.sum(idx_treat)
    n_ctrl = np.sum(idx_ctrl)

    df.bucket_name = df.BUCKET_NAME.astype('category')
    extra_cols = [f'{target}_7d']
    for c in extra_cols:
        df[c] = (df[c] - df[c].mean()) / df[c].std()

    formula = f'{target} ~ BUCKET_NAME'

    if extra_cols:
        formula = formula + ' + ' + ' + '.join(extra_cols)
    model = sm.ols(formula, data=df)
    result = model.fit()

    control_cell_estimate = result.params['Intercept'] + np.sum(
        [df[c].loc[idx_ctrl].mean() * result.params[c] for c in extra_cols])
    treatment_cell_estimate = (result.params['Intercept'] +
                               result.params['BUCKET_NAME[T.test]'] +
                               np.sum([df[c].loc[idx_treat].mean() * result.params[c] for c in extra_cols]))
    p = result.pvalues['BUCKET_NAME[T.test]']
    meandiff = treatment_cell_estimate - control_cell_estimate
    lift = 100 * meandiff / control_cell_estimate

    _df = pd.DataFrame({
        "bucket_name": ["control", "test"],
        f"{target}_7D": [
            df[f"{target}_7D"].loc[idx_treat].mean(),
            df[f"{target}_7D"].loc[idx_treat].mean()]
    })
    _df.bucket_name = _df.bucket_name.astype("category")
    dt = result.get_prediction(_df).summary_frame(alpha=0.1)
    control_cell_estimate = dt['mean'].iloc[0]
    meandiff = dt['mean'].iloc[1] - dt['mean'].iloc[0]
    lower_est = dt['mean_ci_lower'].iloc[1]
    upper_est = dt['mean_ci_upper'].iloc[1]
    lift = 100 * meandiff / control_cell_estimate

    if print_output:
        print(result.summary())
        print(f'Treatment Lift: {lift:>0.3f}% with p-value: {p:>0.4f}')
    std_err = result.bse['Intercept']

    output = {
        "metric": target,
        "pct_lift": lift,
        "pvalue": p,
        "samples_control": n_ctrl,
        "samples_treatment": n_treat,
        "std_err": std_err,
        "delta": meandiff,
        "abs_upper": upper_est,
        "abs_lower": lower_est,
        "pct_upper": 100 * (upper_est / control_cell_estimate - 1),
        "pct_lower": 100 * (lower_est / control_cell_estimate - 1)
    }
    return output


def run_ttest(df, metric='brand_orders', ci_scale=1.28):
    """ Run a T-Test
    """
    x1 = df[metric].loc[df.BUCKET_NAME == 'control'].values
    x2 = df[metric].loc[df.BUCKET_NAME == 'test'].values
    meandiff = np.mean(x2) - np.mean(x1)
    lift = meandiff / np.mean(x1)
    tstat, pvalue, _ = weightstats.ttest_ind(x1, x2, value=meandiff)
    pooled_var = ((x2.shape[0] - 1) * (x2.std() ** 2.0)) + ((x1.shape[0] - 1) * (x1.std() ** 2.0))
    pooled_var /= (x2.shape[0] + x1.shape[0] - 2)
    std_err = np.sqrt(pooled_var / x1.shape[0] + pooled_var / x2.shape[0])

    output = {
        "metric": metric,
        "pct_lift": lift * 100,
        "pvalue": pvalue,
        "samples_control": x1.size,
        "samples_treatment": x2.size,
        "std_err": std_err,
        "delta": meandiff,
        "abs_upper": meandiff + std_err * ci_scale,
        "abs_lower": meandiff - std_err * ci_scale,
        "pct_upper": 100 * ((np.mean(x2) + std_err * ci_scale) / np.mean(x1) - 1),
        "pct_lower": 100 * ((np.mean(x2) - std_err * ci_scale) / np.mean(x1) - 1)
    }
    return output


def outlier_removal(data, target, num_std=5, cap=False):
    """ perform outlier removal """
    df = data.copy()
    if df[target].min() >= 0:
        mval = df[target].loc[df[target] > 0].mean()
    else:
        mval = df[target].mean()
    sval = df[target].loc[df[target] != 0].std()
    upper = mval + num_std * sval
    lower = min(mval - num_std * sval, 0)

    df = df.assign(is_outlier=df[target].apply(lambda x: 1 * ((x > upper) | (x < lower))))
    kdx = df.is_outlier == 0
    if not cap:
        return df.loc[kdx].reset_index(drop=True)

    df[target].loc[df[target] < lower] = lower
    df[target].loc[df[target] > upper] = upper
    return df


def calculate_lift(data, print_output=False, reduction=False, target="brand_orders",
                   remove_outliers=False, cap=False, demean=False):
    """ calculate the lift """
    df = data.copy()
    # if remove_outliers:
    #     df = outlier_removal(df, target, cap=cap)

    if reduction:
        # output = run_regression(df, print_output=print_output, target=target,
        #                         remove_outliers=remove_outliers,
        #                         cap=cap)
        output = estimate_cuped(df, print_output=print_output, target=target,
                                remove_outliers=remove_outliers,
                                cap=cap, demean=demean)
    else:
        output = run_ttest(df, metric=target)

    if print_output:
        df['traffic'] = 1
        metrics = {target: [np.mean, np.std, np.sum], 'traffic': np.sum}
        gcols = ['bucket_name']
        print(df.groupby(gcols).agg(metrics))
        lift = output["pct_lift"]
        p_value = output["pvalue"]
        print(f'Treatment Lift: {lift:>0.3f}% with p-value: {p_value:>0.4f}')
    return output


def get_timeseries(data, metric, remove_outliers=True, variance_reduction=True, cap=False, demean=False):
    """"""

    params = {
        "target": metric,
        "print_output": False,
        "reduction": variance_reduction,
        "remove_outliers": remove_outliers,
        "demean": demean,
        "cap": cap
    }
    idx = data["RETAILER_TOKEN"].notnull()
    df = data.loc[idx].reset_index(drop=True)
    df["bucketed_at"] = pd.to_datetime(df["BUCKETED_AT"])
    # df["order_created_at"] = pd.to_datetime(df["order_created_at"])

    run_dates = []
    start_date = np.min(df["bucketed_at"])
    while start_date <= np.max(df["bucketed_at"]):
        start_date += np.timedelta64(1, 'D')
        run_dates.append(start_date)

    rows = []
    for _date in run_dates:
        idx = df["bucketed_at"] <= _date
        xf = df.loc[idx].reset_index(drop=True)
        # xf[metric].loc[xf["order_created_at"] > _date] = 0
        row = calculate_lift(xf, **params)
        row["bucketed_on"] = _date - np.timedelta64(1, 'D')
        rows.append(row)
    return pd.DataFrame(rows)


def plot_timeseries(df, datecol="bucketed_on", metric="pct_lift",
                    lower="pct_lower", upper="pct_upper", layout=None):
    init_notebook_mode()
    x = df[datecol].apply(lambda x: f"{x.date()}")
    y = df[metric]
    y_upper = df[upper]
    y_lower = df[lower]

    fig = go.Figure([
        go.Scatter(
            name="Lift",
            x=x,
            y=y,
            line=dict(color='#ff0000'),
            mode='lines'
        ),
        go.Scatter(
            name='Upper Bound',
            x=x,
            y=y_upper,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=x,
            y=y_lower,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    if layout is None:
        layout = {
            "yaxis_title": "lift",
            "hovermode": "x",
            "width": 1000,
            "height": 600
        }
    fig.update_layout(**layout)
    fig.show()