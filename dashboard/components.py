from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from plotly.express import bar

ISTANBUL_TIMEZONE = ZoneInfo("Europe/Istanbul")
NUMERIC_DISPLAY_COLUMNS = [
    "selected_prediction",
    "actual",
    "absolute_error",
    "forecast_ptf",
    "lower_bound_95",
    "upper_bound_95",
    "interval_width_95",
]


def format_metric_value(value: Any, suffix: str = "", decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    try:
        return f"{float(value):,.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def to_istanbul_time(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert(ISTANBUL_TIMEZONE)


def readable_timestamp(value: Any) -> str:
    timestamp = to_istanbul_time(value)
    if timestamp is None:
        return "—"
    return timestamp.strftime("%Y-%m-%d %H:%M")


def prepare_prediction_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe
    table = dataframe.copy()
    table["timestamp"] = table["timestamp"].apply(readable_timestamp)
    for column in NUMERIC_DISPLAY_COLUMNS:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(2)
    columns = [
        "timestamp",
        "selected_model",
        "selected_prediction",
        "actual",
        "absolute_error",
        "lower_bound_95",
        "upper_bound_95",
        "risk_level",
    ]
    return table[[column for column in columns if column in table.columns]]


def prepare_day_ahead_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe
    table = dataframe.copy()
    table["timestamp"] = table["timestamp"].apply(readable_timestamp)
    for column in [
        "forecast_ptf",
        "lower_bound_95",
        "upper_bound_95",
        "interval_width_95",
        "residual_std",
    ]:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(2)
    columns = [
        "horizon_hour",
        "timestamp",
        "forecast_ptf",
        "lower_bound_95",
        "upper_bound_95",
        "interval_width_95",
        "risk_level",
        "selected_model",
    ]
    return table[[column for column in columns if column in table.columns]]


def create_day_ahead_forecast_figure(dataframe: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if dataframe.empty:
        return figure

    chart_frame = dataframe.copy()
    chart_frame["timestamp_local"] = pd.to_datetime(
        chart_frame["timestamp"],
        utc=True,
    ).dt.tz_convert(ISTANBUL_TIMEZONE)
    figure.add_trace(
        go.Scatter(
            x=chart_frame["timestamp_local"],
            y=chart_frame["upper_bound_95"],
            mode="lines",
            line={"width": 0},
            name="Upper 95%",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_frame["timestamp_local"],
            y=chart_frame["lower_bound_95"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255, 127, 14, 0.18)",
            line={"width": 0},
            name="95% confidence interval",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_frame["timestamp_local"],
            y=chart_frame["forecast_ptf"],
            mode="lines+markers",
            line={"color": "#ff7f0e", "width": 2},
            name="Day-ahead forecast",
        )
    )
    figure.update_layout(
        height=420,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        hovermode="x unified",
        xaxis_title="Delivery time (Europe/Istanbul)",
        yaxis_title="PTF (TL/MWh)",
        legend={"orientation": "h", "y": 1.08},
    )
    return figure


def create_forecast_figure(dataframe: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if dataframe.empty:
        return figure

    chart_frame = dataframe.copy()
    chart_frame["timestamp_local"] = pd.to_datetime(
        chart_frame["timestamp"],
        utc=True,
    ).dt.tz_convert(ISTANBUL_TIMEZONE)
    figure.add_trace(
        go.Scatter(
            x=chart_frame["timestamp_local"],
            y=chart_frame["upper_bound_95"],
            mode="lines",
            line={"width": 0},
            name="Upper 95%",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_frame["timestamp_local"],
            y=chart_frame["lower_bound_95"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(65, 105, 225, 0.18)",
            line={"width": 0},
            name="95% confidence interval",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_frame["timestamp_local"],
            y=chart_frame["actual"],
            mode="lines",
            line={"color": "#1f77b4", "width": 2},
            name="Actual PTF",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_frame["timestamp_local"],
            y=chart_frame["selected_prediction"],
            mode="lines",
            line={"color": "#ff7f0e", "width": 2},
            name="Selected forecast",
        )
    )
    figure.update_layout(
        height=520,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        hovermode="x unified",
        xaxis_title="Delivery time (Europe/Istanbul)",
        yaxis_title="PTF (TL/MWh)",
        legend={"orientation": "h", "y": 1.08},
    )
    return figure


def create_risk_distribution_figure(dataframe: pd.DataFrame) -> go.Figure:
    if dataframe.empty or "risk_level" not in dataframe.columns:
        return go.Figure()
    counts = (
        dataframe["risk_level"]
        .value_counts()
        .rename_axis("risk_level")
        .reset_index(name="count")
    )
    return bar(
        counts,
        x="risk_level",
        y="count",
        color="risk_level",
        title="Risk level distribution",
        category_orders={"risk_level": ["LOW", "MEDIUM", "HIGH"]},
    )


def create_risk_error_figure(dataframe: pd.DataFrame) -> go.Figure:
    if dataframe.empty or {"risk_level", "absolute_error"}.difference(
        dataframe.columns
    ):
        return go.Figure()
    grouped = (
        dataframe.groupby("risk_level", as_index=False)["absolute_error"]
        .mean()
        .sort_values("risk_level")
    )
    return bar(
        grouped,
        x="risk_level",
        y="absolute_error",
        color="risk_level",
        title="Average absolute error by risk level",
        category_orders={"risk_level": ["LOW", "MEDIUM", "HIGH"]},
    )


def create_interval_width_figure(dataframe: pd.DataFrame) -> go.Figure:
    if dataframe.empty or {"risk_level", "interval_width_95"}.difference(
        dataframe.columns
    ):
        return go.Figure()
    grouped = (
        dataframe.groupby("risk_level", as_index=False)["interval_width_95"]
        .mean()
        .sort_values("risk_level")
    )
    return bar(
        grouped,
        x="risk_level",
        y="interval_width_95",
        color="risk_level",
        title="Average 95% interval width by risk level",
        category_orders={"risk_level": ["LOW", "MEDIUM", "HIGH"]},
    )
