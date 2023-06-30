from enum import Enum

import numpy as np
import pandas as pd
import scipy.stats as stats
from experiment_analysis.ttest import summarize_ttest
from experiment_analysis.cuped import apply_cuped
from experiment_analysis.capping import apply_capping, CappingMethod
from experiment_analysis.isotonic import apply_isotonic
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.graph_objs as go


class TimeseriesMethod(Enum):
    STANDARD = "STANDARD"
    CUPED = "CUPED"
    CAPPED = "CAPPED"
    CUPED_CAPPED = "CUPED_CAPPED"
    ISOTONIC = "ISOTONIC"
    ISOTONIC_CAPPED = "ISOTONIC_CAPPED"


def get_timeseries(
    df,
    metric,
    event_ts,
    method=TimeseriesMethod.STANDARD,
    control_name="control",
    treatment_name="treatment",
    confidence_level=0.9,
    covariate_prefix="pre_exp",
    df_pre_exp=None,
    analyze_dates=np.timedelta64(1, 'D')
):
    params = {
        "control_name": control_name,
        "treatment_name": treatment_name,
        "metric": metric,
        "confidence_level": confidence_level
    }
    df.columns = df.columns.str.lower()
    df["bucketed_at"] = pd.to_datetime(df["bucketed_at"])
    df[event_ts] = pd.to_datetime(df[event_ts])
    df["exp_start_date"] = pd.to_datetime(df["exp_start_date"])

    run_dates = []
    # First day of computation is first full day of assignments
    start_date = df["exp_start_date"].unique()[0]

    if type(analyze_dates) == np.timedelta64:
        while start_date <= np.max(df["bucketed_at"]):
            start_date += analyze_dates
            run_dates.append(start_date)
    elif type(analyze_dates) == list:
        run_dates = analyze_dates
    else:
        while start_date <= np.max(df["bucketed_at"]):
            start_date += np.timedelta64(1, 'D')
            run_dates.append(start_date)

    rows = []
    for _date in run_dates:
        # Keep subjects where bucketing event is bfore current date
        # Set events to zero where date of event (e.g., brand order created) is after current date
        # Set events to zero where event happened before bucketing as a safeguard. Should be done in main query.
        idx = df["bucketed_at"] <= _date
        df_date = df.loc[idx].reset_index(drop=True)
        df_date[metric].loc[df_date[event_ts] > _date] = 0
        df_date[metric].loc[df_date[event_ts] < df_date["bucketed_at"]] = 0

        # Must aggregate df to level of unit of randomization for correct t-test
        merge_cols = ["identifier", "bucket_name"]
        df_agg_metric = (df_date[["identifier", "bucket_name", metric]]
                         .groupby(merge_cols)
                         .agg('sum')
                         .reset_index())

        # Update metric if variance reduction method is applied
        if method != TimeseriesMethod.STANDARD:
            params.update({"metric": f"{metric}_{method.value.lower()}"})

        # Calculate CUPED and capped metrics depending on method provided
        if method == TimeseriesMethod.CAPPED:
            df_agg_metric = df_agg_metric.assign(**{f"{metric}_capped": apply_capping(df_agg_metric, metric)})
        else:
            df_agg_metric = df_agg_metric.merge(
                df_pre_exp[[*merge_cols, f"{covariate_prefix}_{metric}"]], on=merge_cols, how="left")
            df_agg_metric[f"pre_exp_{metric}"].fillna(0, inplace=True)
            
            if method == TimeseriesMethod.CUPED:
                df_agg_metric = df_agg_metric.assign(**{
                        f"{metric}_cuped": apply_cuped(df_agg_metric, metric, f"{covariate_prefix}_{metric}")})
            elif method == TimeseriesMethod.ISOTONIC:
                df_agg_metric = df_agg_metric.assign(**{
                        f"{metric}_isotonic": apply_isotonic(df_agg_metric, metric, f"{covariate_prefix}_{metric}")})
            elif method == TimeseriesMethod.CUPED_CAPPED:
                df_agg_metric = df_agg_metric.assign(**{
                        f"{metric}_cuped": apply_cuped(df_agg_metric, metric, f"{covariate_prefix}_{metric}"),
                        f"{metric}_cuped_capped": lambda df_agg_metric: apply_capping(df_agg_metric,
                                                                metric=f"{metric}_cuped",
                                                                method=CappingMethod.WITH_CUPED,
                                                                metric_raw=metric)})
            elif method == TimeseriesMethod.ISOTONIC_CAPPED:
                df_agg_metric = df_agg_metric.assign(**{
                        f"{metric}_isotonic": apply_isotonic(df_agg_metric, metric, f"{covariate_prefix}_{metric}"),
                        f"{metric}_isotonic_capped": lambda df_agg_metric: apply_capping(df_agg_metric,
                                                                metric=f"{metric}_isotonic",
                                                                method=CappingMethod.WITH_ISOTONIC,
                                                                metric_raw=metric)})

        # Calculate ttest results for desired metric
        row = summarize_ttest(df_agg_metric, **params)
        row["ds"] = _date - np.timedelta64(1, 'D')
        row["bucket_count"] = df_agg_metric.shape[0]

        rows.append(row)
    return pd.DataFrame(rows)


def plot_timeseries(df, datecol="ds", metric="pct_lift",
                    lower="ci_lift_lower", upper="ci_lift_upper", layout=None):
    init_notebook_mode()

    x = df[datecol].apply(lambda x: f"{x.date()}")
    y = df[metric]
    y_upper = df[upper]
    y_lower = df[lower]

    fig = go.Figure([
        go.Scatter(
            name="% Lift",
            x=x,
            y=y,
            line=dict(color='maroon'),
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

    fig.update_layout(
        yaxis={'tickformat': ',.3%'}
    )

    fig.show()
