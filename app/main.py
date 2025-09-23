import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # /app
ROOT_DIR = os.path.dirname(BASE_DIR)                   # project root
sys.path.append(ROOT_DIR)

# Import custom regime detection functions from models/
from models.Rupt import detect_regimes, plot_regimes

# --- Set page style and layout ---
st.set_page_config(
    page_title="AI Forecast & Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Background gradient CSS ---
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(90deg, #111827, #1f2937);
            color: white;
        }
        .stMetric label, .stMetric span {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview & Dashboard",
    "Forecasting Model",
    "AI Sentiment Analysis",
    "Regime Change Analysis"
])

# === PAGE: Overview & Dashboard ===
if page == "Overview & Dashboard":
    st.markdown("# Quant LSTM Intelligence Suite")
    st.markdown("### Welcome to your all-in-one predictive finance dashboard.")
    st.markdown("""
    This tool is designed to help you explore, forecast, and analyze financial time series using modern AI and statistical tools.
    Use it to understand market behavior, identify regime shifts, and visualize future price ranges with confidence bounds.

    **Here's what's inside:**
    """)

    st.markdown("#### Forecasting Model")
    st.markdown("""
    - Powered by LSTM + Quantile Regression
    - Predicts multiple price quantiles (10th, 50th, 90th)
    - Ideal for high/medium/low scenario planning
    - Explore forecasts interactively with a dynamic time horizon slider
    """)

    st.markdown("#### AI Sentiment Analysis")
    st.markdown("""
    - Uses pre-trained RoBERTa for daily sentiment scoring
    - Visualizes public market mood over time
    - Helps gauge crowd psychology for smarter decision-making
    """)

    st.markdown("#### Regime Change Analysis")
    st.markdown("""
    - Uses Ruptures for change-point detection
    - Identifies bullish and bearish phases in market behavior
    - Colored trendlines (green = bull, red = bear) highlight structural shifts
    """)

    st.markdown("----")
    st.info("Use the sidebar to begin exploring each model in detail.")

# === PAGE: Forecasting Model ===
elif page == "Forecasting Model":
    st.markdown("## Forecasting Model")
    st.markdown("Welcome to your AI-powered forecast interface.")

    forecast_dir = os.path.join(ROOT_DIR, "forecast_outputs")

    with st.sidebar:
        st.subheader("Forecast Settings")
        asset = st.selectbox("Select asset", ["BTC-USD", "ETH-USD", "EURUSD=X"])

    file_path = os.path.join(forecast_dir, f"{asset}.json")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            df = pd.DataFrame(data)

        if df.columns[0] != "Day":
            df.columns = ["Day", "q10", "q50", "q90"]

        max_horizon = int(df["Day"].max())

        with st.sidebar:
            selected_days = st.slider("Forecast horizon (days)", 1, max_horizon, 15)

        df = df[df["Day"] <= selected_days]

        latest = df[df["Day"] == selected_days]
        st.markdown("### Forecast Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("10th Percentile", f"${latest['q10'].values[0]:.2f}")
        col2.metric("Median Forecast", f"${latest['q50'].values[0]:.2f}")
        col3.metric("90th Percentile", f"${latest['q90'].values[0]:.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Day"], y=df["q10"], mode='lines', name='10th Percentile'))
        fig.add_trace(go.Scatter(x=df["Day"], y=df["q50"], mode='lines', name='Median Forecast (q50)'))
        fig.add_trace(go.Scatter(x=df["Day"], y=df["q90"], mode='lines', name='90th Percentile'))

        fig.update_layout(
            title=f"Forecast for {asset}",
            xaxis_title="Forecast Horizon (Days)",
            yaxis_title="Price (USD)",
            yaxis=dict(tickformat=".4f", tickmode="auto"),
            xaxis=dict(tickmode='linear'),
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Forecast Table")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name=f"{asset}_forecast.csv",
            mime='text/csv'
        )

    except FileNotFoundError:
        st.error(f"No forecast file found for {asset}.")
    except json.JSONDecodeError:
        st.error(f"Invalid JSON format in {file_path}")
    except Exception as e:
        st.error(f"Could not load file: {e}")

# === PAGE: AI Sentiment Analysis ===
elif page == "AI Sentiment Analysis":
    st.markdown("## AI Sentiment Analysis")
    st.write("Visualize daily sentiment scores over time (collected via RoBERTa)")

    CSV_PATH = os.path.join(ROOT_DIR, "NLP sentiment", "daily_sentiment.csv")

    try:
        df = pd.read_csv(CSV_PATH)

        # Ensure date is treated as integer day index
        df["date"] = df["date"].astype(int)
        df = df.sort_values("date")

        st.line_chart(df.set_index("date")["score"])

        st.markdown("### Daily Sentiment Scores")
        st.dataframe(df)

    except FileNotFoundError:
        st.error("Sentiment file not found. Run the sentiment collector first.")
    except Exception as e:
        st.error(f"Error loading sentiment file: {e}")


# === PAGE: Regime Change Analysis ===
elif page == "Regime Change Analysis":
    st.markdown("## Regime Change Analysis")
    st.write("Visualize market regime shifts using change-point detection (Ruptures).")

    data_folder = os.path.join(ROOT_DIR, "yf data")
    files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

    selected_file = st.selectbox("Select asset for regime analysis", files)
    penalty = st.slider("Penalty (change-point sensitivity)", 1, 50, 10)

    if selected_file:
        try:
            df, change_points = detect_regimes(os.path.join(data_folder, selected_file), penalty=penalty)
            fig = plot_regimes(df, change_points)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error running regime detection: {e}")
