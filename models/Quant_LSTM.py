# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import optuna
from tqdm import tqdm
import yfinance as yf
from datetime import date, timedelta
import os

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

lookback = 60
horizon = 30
batch_size = 16
learning_rate = 0.001
num_epochs = 50
hidden_size = 128
num_stacked_layers = 3
dropout = 0.1
quantiles = [0.1, 0.5, 0.9]

# %%
def prepare_lstm_data(df, target_column="Price", lookback=60, horizon=30, scaler_type="robust"):
    df = df.copy()
    df.set_index("Date", inplace=True)
    df = df[[target_column]].copy()
    df[target_column] = df[target_column].shift(-horizon)

    for i in range(1, lookback + 1):
        df[f'{target_column}(t-{i})'] = df[target_column].shift(i)

    df.dropna(inplace=True)
    data_np = df.to_numpy()

    scaler = RobustScaler() if scaler_type == "robust" else MinMaxScaler(feature_range=(-1, 1))
    split_index = int(len(data_np) * 0.8)
    scaler.fit(data_np[:split_index])
    data_np = scaler.transform(data_np)

    X = np.flip(data_np[:, 1:], axis=1).copy()
    y = data_np[:, 0]

    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]

    X_train = X_train.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    return (torch.tensor(X_train).float(), torch.tensor(y_train).float(),
            torch.tensor(X_test).float(), torch.tensor(y_test).float(),
            scaler)

class TSData(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]
    
def quantile_loss(preds, target, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i].unsqueeze(1)
        loss = torch.max((q - 1) * errors, q * errors)
        losses.append(torch.mean(loss))
    return torch.stack(losses).sum()

# %%
df_dict = {}
tickers = ["BTC-USD", "ETH-USD", "EURUSD=X"]

for ticker in tickers:
    df = yf.Ticker(ticker).history(period="1y", interval="1d")
    df.reset_index(inplace=True)
    df = df.iloc[::-1]
    df["Price"] = df["Close"].astype(float)
    
    df_dict[ticker] = df
    X_train, y_train, X_test, y_test, scaler = prepare_lstm_data(df)

df_dict

# %%


# %%
class QuantileLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_quantiles):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, num_quantiles)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.layer_norm(out[:, -1, :])
        return self.fc(out)

def inverse_transform(y_scaled, scaler, lookback):
    dummy = np.zeros((len(y_scaled), lookback + 1))
    dummy[:, 0] = y_scaled
    return scaler.inverse_transform(dummy)[:, 0]
def evaluate_quantile_predictions(model, X_train, y_train, X_test, y_test, scaler, lookback, ticker):
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train.to(device)).cpu().numpy()
        test_preds = model(X_test.to(device)).cpu().numpy()

    train_q50 = inverse_transform(train_preds[:, 1], scaler, lookback)
    train_actual = inverse_transform(y_train.cpu().numpy().flatten(), scaler, lookback)
    
    test_q10 = inverse_transform(test_preds[:, 0], scaler, lookback)
    test_q50 = inverse_transform(test_preds[:, 1], scaler, lookback)
    test_q90 = inverse_transform(test_preds[:, 2], scaler, lookback)
    test_actual = inverse_transform(y_test.cpu().numpy().flatten(), scaler, lookback)

    plt.figure(figsize=(15, 6))
    plt.plot(train_actual, label="Training Actual", color="black")
    plt.plot(train_q50, label="Training Predictions", color="blue")
    plt.title(f"Training Data: Predictions vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 6))
    plt.plot(test_actual, label="Test Actual", color="black")
    plt.plot(test_q50, label="Test Predictions", color="blue")
    plt.fill_between(range(len(test_actual)), test_q10, test_q90, color="gray", alpha=0.3, 
                    label="80% Prediction Interval")
    plt.title(f"Test Data: {ticker} — Predictions vs Actual ({horizon}-day ahead)")

    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    mae = np.mean(np.abs(test_actual - test_q50))
    rmse = np.sqrt(np.mean((test_actual - test_q50)**2))
    mape = np.mean(np.abs((test_actual - test_q50) / test_actual)) * 100
    
    print(f"\nTest Set Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    return {
        "train_actual": train_actual,
        "train_pred": train_q50,
        "test_actual": test_actual,
        "test_q10": test_q10,
        "test_q50": test_q50,
        "test_q90": test_q90
    }

def forecast_next_days(model, X_last_window, scaler, lookback, horizon, quantiles):
    model.eval()
    preds_q10, preds_q50, preds_q90 = [], [], []

    current_input = X_last_window.clone().to(device)

    for _ in range(horizon):
        with torch.no_grad():
            out = model(current_input)
            q10, q50, q90 = out[0].cpu().numpy()
            preds_q10.append(q10)
            preds_q50.append(q50)
            preds_q90.append(q90)

        next_step = q50 + np.random.normal(0, abs(q90 - q10) / 4)
        next_step_tensor = torch.tensor([[[next_step]]], dtype=torch.float32).to(device)
        current_input = torch.cat([current_input[:, 1:, :], next_step_tensor], dim=1)

    def inv(preds_scaled):
        dummy = np.zeros((len(preds_scaled), lookback + 1))
        dummy[:, 0] = preds_scaled
        return scaler.inverse_transform(dummy)[:, 0]

    return {
        "q10": inv(preds_q10),
        "q50": inv(preds_q50),
        "q90": inv(preds_q90)
    }

def plot_forecast(forecast_dict, ticker, last_actual=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.set(style="whitegrid", rc={"grid.linewidth": 0.5, "grid.alpha": 0.5})
    sns.set_context("notebook", font_scale=1.2)
    
    q10 = forecast_dict["q10"]
    q50 = forecast_dict["q50"]
    q90 = forecast_dict["q90"]
    
    df = pd.DataFrame({
        'Day': range(len(q50)),
        'Lower Bound (q10)': q10,
        'Median Forecast (q50)': q50,
        'Upper Bound (q90)': q90
    })
    
    plt.figure(figsize=(15, 8))
    
    ax = plt.gca()
    ax.fill_between(df['Day'], df['Lower Bound (q10)'], df['Upper Bound (q90)'], 
                    color='lightsteelblue', alpha=0.5, label='80% Prediction Interval')
    
    sns.lineplot(data=df, x='Day', y='Median Forecast (q50)', 
                 color='royalblue', linewidth=3, label='Median Forecast (q50)')
    sns.lineplot(data=df, x='Day', y='Lower Bound (q10)', 
                 color='crimson', linewidth=1.5, linestyle='--', label='Lower Bound (q10)')
    sns.lineplot(data=df, x='Day', y='Upper Bound (q90)', 
                 color='forestgreen', linewidth=1.5, linestyle='--', label='Upper Bound (q90)')
    
    plt.scatter(df['Day'], df['Median Forecast (q50)'], color='royalblue', s=40, zorder=5)
    
    pct_change = ((q50[-1] - q50[0]) / q50[0] * 100)
    change_direction = "↑" if pct_change > 0 else "↓"
    change_color = "green" if pct_change > 0 else "red"
    
    plt.annotate(f"${q50[0]:,.2f}", (0, q50[0]), xytext=(-10, -20), 
                textcoords='offset points', fontweight='bold')
    plt.annotate(f"${q50[-1]:,.2f} ({change_direction}{abs(pct_change):.1f}%)", 
                (len(q50)-1, q50[-1]), xytext=(10, 10), 
                textcoords='offset points', fontweight='bold', color=change_color)
    
    plt.title(f"{horizon}-Day {ticker} Price Forecast", fontsize=16, fontweight='bold', pad=20)

    plt.xlabel("Days Ahead", fontsize=12)
    plt.ylabel("Price ($)", fontsize=12)
    plt.yscale('log')

    plt.legend(loc='upper left', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.show()


# %%
def train_model(model, train_loader, optimizer, num_epochs, device, ticker=""):
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        
        loop = tqdm(train_loader, desc=f"[{ticker}] Epoch {epoch+1}/{num_epochs}")

        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)
            
            preds = model(xb)
            loss = quantile_loss(preds, yb, quantiles)
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            loop.set_postfix(loss=loss.item())
        
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        print(f"[Epoch {epoch+1}/{num_epochs}] Training Loss: {avg_loss:.6f}")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")
    
    plt.figure(figsize=(12, 5))
    plt.plot(loss_history)
    plt.title(f"Training Loss History — {ticker}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(f"\nTraining completed! Final loss: {loss_history[-1]:.6f}")
    return model

def objective(trial, df):
    lookback = trial.suggest_int("lookback", 30, 90, step=10)
    hidden_size = trial.suggest_int("hidden_size", 64, 256, step=32)
    num_stacked_layers = trial.suggest_int("num_layers", 1, 5)
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    
    print(f"\nTrial #{trial.number}:")
    print(f"  lookback: {lookback}, hidden_size: {hidden_size}, num_layers: {num_stacked_layers}")
    print(f"  dropout: {dropout}, learning_rate: {learning_rate}, batch_size: {batch_size}")
    
    X_train, y_train, X_test, y_test, scaler = prepare_lstm_data(
        df, lookback=lookback, horizon=horizon, scaler_type="robust"
    )
    
    val_size = int(len(X_train) * 0.2)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]
    
    train_dataset = TSData(X_train, y_train)
    val_dataset = TSData(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = QuantileLSTM(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_stacked_layers,
        dropout=dropout,
        num_quantiles=len(quantiles)
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(15):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = quantile_loss(preds, yb, quantiles)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += quantile_loss(preds, yb, quantiles).item()
        
        val_loss /= len(val_loader)
        
        print(f"    Epoch {epoch+1}/15 - Validation Loss: {val_loss:.6f}" + 
              (f" (best)" if val_loss < best_val_loss else ""))
        
        trial.report(val_loss, epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break
            
        if trial.should_prune():
            print(f"    Trial pruned at epoch {epoch+1}")
            raise optuna.exceptions.TrialPruned()
    
    print(f"  Final validation loss: {best_val_loss:.6f}")
    return best_val_loss

def run_hyperparameter_optimization(df, ticker, n_trials=50):
    print("\n" + "="*50)
    print("STARTING HYPERPARAMETER OPTIMIZATION")
    print("="*50)
    print(f"Number of trials: {n_trials}")
    print("Parameters being optimized:")
    print("  - lookback window (30-90 days)")
    print("  - hidden size (64-256 neurons)")
    print("  - number of LSTM layers (1-5)")
    print("  - dropout rate (0.0-0.5)")
    print("  - learning rate (1e-4 to 1e-2)")
    print("  - batch size (8, 16, 32, 64)")
    print("="*50 + "\n")
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, df), n_trials=n_trials)
    
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best validation loss: {study.best_trial.value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  - {key}: {value}")
    print("="*50)
    
    try:
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.show()
        
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.show()
        
        fig3 = optuna.visualization.plot_contour(study)
        fig3.show()
    except Exception as e:
        print(f"Unable to display Optuna visualization: {str(e)}")
    
    return study.best_params

# %%
def main():
    tickers = ["BTC-USD", "ETH-USD", "EURUSD=X"]

    for ticker in tickers:
        print(f"\n{'='*60}\n🏁 Running pipeline for {ticker}\n{'='*60}")

        # Download and preprocess data
        df = yf.Ticker(ticker).history(period="1y", interval="1d")
        df.reset_index(inplace=True)
        df = df[::-1]
        df["Price"] = df["Close"].astype(float)

        # Run Optuna hyperparameter search
        best_params = run_hyperparameter_optimization(df, ticker, n_trials=20)

        lookback = best_params["lookback"]
        hidden_size = best_params["hidden_size"]
        num_stacked_layers = best_params["num_layers"]
        dropout = best_params["dropout"]
        learning_rate = best_params["learning_rate"]
        batch_size = best_params["batch_size"]

        # Prepare data using best hyperparameters
        X_train, y_train, X_test, y_test, scaler = prepare_lstm_data(
            df, lookback=lookback, horizon=horizon, scaler_type="robust"
        )

        train_dataset = TSData(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = QuantileLSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_stacked_layers,
            dropout=dropout,
            num_quantiles=len(quantiles)
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        model = train_model(model, train_loader, optimizer, num_epochs=num_epochs, device=device, ticker=ticker)

        results = evaluate_quantile_predictions(model, X_train, y_train, X_test, y_test, scaler, lookback, ticker)
        X_last_window = X_test[-1].unsqueeze(0)
        forecast = forecast_next_days(model, X_last_window, scaler, lookback, horizon, quantiles)

        # Save forecast as JSON with Day column
        os.makedirs("forecast_outputs", exist_ok=True)
        forecast_df = pd.DataFrame(forecast)
        forecast_df["Day"] = list(range(1, horizon + 1))
        forecast_df = forecast_df[["Day", "q10", "q50", "q90"]]  # reordering
        forecast_df.to_json(f"forecast_outputs/{ticker.replace('/', '_')}.json", orient="records", indent=4)
        print(f"✅ Saved forecast to forecast_outputs/{ticker.replace('/', '_')}.json")

        # Save model
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), f"models/model_{ticker.replace('/', '_')}.pt")
        print(f"✅ Saved model to models/model_{ticker.replace('/', '_')}.pt")

        # Plot forecast
        plot_forecast(forecast, ticker)

        print(f"✅ Completed: {ticker}")

if __name__ == "__main__":
    main()


# %%



