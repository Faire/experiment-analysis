import pandas as pd
import numpy as np
import scipy.stats as stats
from experiment_analysis.ttest import summarize_ttest
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.graph_objs as go


def get_timeseries(df, metric, event_ts, control_name='control', treatment_name='treatment', confidence_level=0.9):
    params = {
        'control_name': control_name,
        'treatment_name': treatment_name,
        'metric': metric,
        'confidence_level': confidence_level
    }
    df.columns = df.columns.str.lower()
    df["bucketed_at"] = pd.to_datetime(df["bucketed_at"])
    df[event_ts] = pd.to_datetime(df[event_ts])
    df["exp_start_date"] = pd.to_datetime(df["exp_start_date"])

    run_dates = []
    # First day of computation is first full day of assignments
    start_date = df['exp_start_date'].unique()[0]  # + np.timedelta64(1, 'D')

    while start_date <= np.max(df["bucketed_at"]):
        start_date += np.timedelta64(1, 'D')
        run_dates.append(start_date)

    rows = []
    for _date in run_dates:
        idx = df["bucketed_at"] <= _date
        df_date = df.loc[idx].reset_index(drop=True)
        df_date[metric].loc[df_date[event_ts] > _date] = 0
        df_date[metric].loc[df_date[event_ts] < df_date['exp_start_date']] = 0
        row = summarize_ttest(df_date, **params)
        row["ds"] = _date - np.timedelta64(1, 'D')
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
