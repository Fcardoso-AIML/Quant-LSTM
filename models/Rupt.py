import pandas as pd
import numpy as np
import ruptures as rpt
import plotly.graph_objects as go
from pandas.api.types import is_datetime64tz_dtype

DEFAULT_PENALTY = 10

def _clean_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if is_datetime64tz_dtype(dt):
        dt = dt.dt.tz_convert(None)
    return dt

def detect_regimes(csv_path, model_type="rbf", penalty=10):
    df = pd.read_csv(csv_path)

    # Robust datetime parsing
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Use Price if exists, otherwise Close
    price_col = "Price" if "Price" in df.columns else "Close"
    if price_col not in df.columns:
        raise ValueError(f"No valid price column found in {csv_path}. Columns: {df.columns}")

    signal = df[price_col].astype(float).values

    algo = rpt.Pelt(model=model_type).fit(signal)
    change_points = algo.predict(pen=penalty)

    df["Regime"] = 0
    start = 0
    for i, cp in enumerate(change_points):
        df.loc[start:cp - 1, "Regime"] = i
        start = cp

    return df, change_points



def plot_regimes(df: pd.DataFrame, change_points):
    df = df.copy()
    df["Date"] = _clean_datetime(df["Date"])
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    price_col = "Price" if "Price" in df.columns else "Close"
    fig = go.Figure()

    # Plot each segment in green (up) / red (down)
    for i in range(len(change_points)):
        start_idx = 0 if i == 0 else change_points[i - 1]
        end_idx = change_points[i]

        segment = df.iloc[start_idx:end_idx]
        x = segment["Date"]
        y = segment[price_col]

        slope = (y.values[-1] - y.values[0]) / (len(y) + 1e-6)
        color = "green" if slope >= 0 else "red"
        label = "Bullish Regime" if slope >= 0 else "Bearish Regime"

        # Show each label only once in legend
        show_legend = not any(tr.name == label for tr in fig.data)

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=label,
            line=dict(color=color, width=2), showlegend=show_legend
        ))

    # Vertical dashed lines at change-points (skip the last endpoint)
    for cp in change_points[:-1]:
        fig.add_shape(
            type="line",
            x0=df["Date"].iloc[cp], x1=df["Date"].iloc[cp],
            y0=df[price_col].min(), y1=df[price_col].max(),
            line=dict(color="gray", dash="dash")
        )

    fig.update_layout(
        title="Regime Shifts Detected (Ruptures)",
        xaxis_title="Date",
        yaxis_title=price_col,
        template="plotly_dark"
    )
    return fig
