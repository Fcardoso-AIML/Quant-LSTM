import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
from matplotlib.lines import Line2D
import plotly.graph_objects as go


penality=10
def detect_regimes(csv_path, model_type="rbf", penalty=penality):
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Prepare signal (use Price directly or log returns)
    signal = df["Price"].values

    # Detect change points using Ruptures
    algo = rpt.Pelt(model=model_type).fit(signal)
    change_points = algo.predict(pen=penalty)

    # Create a regime column based on break segments
    df["Regime"] = 0
    start = 0
    for i, cp in enumerate(change_points):
        df.loc[start:cp-1, "Regime"] = i
        start = cp

    return df, change_points
def plot_regimes(df, change_points):
    df = df.sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])

    fig = go.Figure()

    # Segment the plot by regime and color
    for i in range(len(change_points)):
        start_idx = 0 if i == 0 else change_points[i - 1]
        end_idx = change_points[i]

        segment = df.iloc[start_idx:end_idx]
        x = segment["Date"]
        y = segment["Price"]

        # Trend direction
        slope = (y.values[-1] - y.values[0]) / (len(y) + 1e-6)
        color = "green" if slope >= 0 else "red"
        label = "Bullish Regime" if slope >= 0 else "Bearish Regime"

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=label,
            line=dict(color=color, width=2),
            showlegend=(i == 0 or label not in [trace.name for trace in fig.data])
        ))

    # Vertical lines for change points
    for cp in change_points[:-1]:
        fig.add_shape(
            type="line",
            x0=df["Date"].iloc[cp],
            x1=df["Date"].iloc[cp],
            y0=df["Price"].min(),
            y1=df["Price"].max(),
            line=dict(color="gray", dash="dash"),
        )

    fig.update_layout(
        title="📉 Regime Shifts Detected (Ruptures)",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark"
    )

    return fig
