import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

def detect_regimes(csv_path, n_components=2):
    df = pd.read_csv(csv_path)
    df["Returns"] = np.log(df["Price"]).diff().dropna()
    returns = df["Returns"].dropna().values.reshape(-1, 1)

    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000)
    model.fit(returns)
    hidden_states = model.predict(returns)

    df = df.iloc[1:].copy()
    df["Regime"] = hidden_states

    return df, model

def plot_regimes(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    for regime in sorted(df["Regime"].unique()):
        subset = df[df["Regime"] == regime]
        ax.plot(subset["Date"], subset["Price"], '.', label=f"Regime {regime}")
    ax.legend()
    ax.set_title("HMM Regime Detection")
    return fig
