import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

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
    fig, ax = plt.subplots(figsize=(14, 6))
    df = df.sort_values("Date").reset_index(drop=True)
    ax.plot(df["Date"], df["Price"], color="black", linewidth=1.5, label="Price")

    # Plot vertical lines at change points
    for cp in change_points[:-1]:  # Skip the last point (end of series)
        ax.axvline(df["Date"].iloc[cp], color="red", linestyle="--", alpha=0.7)

    ax.set_title("📈 Regime Shifts Detected (Ruptures)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    return fig
