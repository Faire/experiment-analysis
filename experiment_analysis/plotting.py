import pandas as pd
import numpy as np
import scipy.stats as stats
from experiment_analysis.ttest import summarize_ttest
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.graph_objs as go


class TimeseriesMethod(Enum):
    STANDARD = "STANDARD"
    CUPED = "CUPED"
    CAPPED = "CAPPED"
    CUPED_CAPPED = "CUPED_CAPPED"


def get_timeseries(
    df,
    metric,
    event_ts,
    method=TimeseriesMethod.STANDARD,
    control_name="control",
    treatment_name="treatment",
    confidence_level=0.9,
    covariate_prefix="pre_exp",
    df_pre_exp=None
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

        df_agg_metric = (df_date[["identifier", "bucket_name", metric]]
                         .groupby(["identifier", "bucket_name"])
                         .agg('sum')
                         .reset_index())

        # Calculate CUPED and capped metrics depending on method provided
        merge_cols = ["identifier", "bucket_name"]
        if method == TimeseriesMethod.CUPED:
            params.update({"metric": f"{metric}_cuped"})
            df_agg_joined = df_agg_metric.merge(
                df_pre_exp[[*merge_cols, f"{covariate_prefix}_{metric}"]], on=merge_cols, how="left")
            df_agg_joined[f"pre_exp_{metric}"].fillna(0, inplace=True)
            df_agg_joined[f"{metric}_cuped"] = apply_cuped(
                df_agg_joined, metric, f"{covariate_prefix}_{metric}")
        elif method == TimeseriesMethod.CAPPED:
            params.update({"metric": f"{metric}_capped"})
            df_agg_metric[f"{metric}_capped"] = apply_capping(
                df_agg_metric, metric)
        elif method == TimeseriesMethod.CUPED_CAPPED:
            params.update({"metric": f"{metric}_cuped_capped"})
            df_agg_joined = df_agg_metric.merge(
                df_pre_exp[[*merge_cols, f"{covariate_prefix}_{metric}"]], on=merge_cols, how="left")
            df_agg_joined[f"pre_exp_{metric}"].fillna(0, inplace=True)
            df_agg_joined[f"{metric}_cuped"] = apply_cuped(
                df_agg_joined, metric, f"{covariate_prefix}_{metric}")
            df_agg_joined[f"{metric}_cuped_capped"] = apply_capping(df=df_agg_joined,
                                                                    metric=f"{metric}_cuped",
                                                                    method=CappingMethod.WITH_CUPED,
                                                                    metric_raw=metric)

        # Calculate ttest results for desired metric
        if method == TimeseriesMethod.CUPED or method == TimeseriesMethod.CUPED_CAPPED:
            row = summarize_ttest(df_agg_joined, **params)
            row["ds"] = _date - np.timedelta64(1, 'D')
            row["bucket_count"] = df_agg_joined.shape[0]
        else:
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
    fig.show()
