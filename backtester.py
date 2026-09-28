import yfinance as yf
import pandas as pd
import pandas_ta as ta
import pandas_datareader.data as web
import numpy as np
import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Base Market Data Ingestion (Updated to ARKK)
# ---------------------------------------------------------
print("Downloading market data...")
asset = yf.Ticker("ARKK") # Swapped from SPY to ARKK
start_date = "2014-01-01"
end_date = datetime.date.today().strftime("%Y-%m-%d")
df = asset.history(start=start_date, end=end_date) 

# Strip timezone for merging
df.index = df.index.tz_localize(None)

# ---------------------------------------------------------
# 2. Macroeconomic Data Ingestion (FRED)
# ---------------------------------------------------------
print("Downloading macroeconomic data from FRED...")
macro_tickers = ['DGS10', 'T10Y2Y', 'CPIAUCSL', 'FEDFUNDS']
macro_df = web.DataReader(macro_tickers, 'fred', start_date, end_date)

# Forward-fill the monthly data to create daily values
macro_df = macro_df.ffill()

# Join the macro data onto our daily stock dataframe
print("Merging datasets and engineering technical features...")
df = df.join(macro_df, how='left')

# ---------------------------------------------------------
# 3. Technical Feature Engineering
# ---------------------------------------------------------
df.ta.rsi(length=14, append=True)
df.ta.macd(fast=12, slow=26, signal=9, append=True)
df['Daily_Return'] = df['Close'].pct_change()
df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()

# Drop rows with NaN values created by rolling windows and missing macro data
df = df.dropna()

# ---------------------------------------------------------
# 4. Define the Target (UPDATED: 5-Day Horizon)
# ---------------------------------------------------------
# We ask the model: "Will the price be higher 5 trading days from now?"
# This smooths out daily volatility and matches the slower macro data.
df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)

# Crucial step: Drop the final 5 rows where the future price isn't known yet 
# to prevent the model from "cheating" during training.
df = df.dropna()

# ---------------------------------------------------------
# 5. Machine Learning Setup
# ---------------------------------------------------------
features = ['RSI_14', 'MACD_12_26_9', 'Volatility_10', 'DGS10', 'T10Y2Y', 'CPIAUCSL', 'FEDFUNDS']
X = df[features]
y = df['Target']

split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# ---------------------------------------------------------
# 6. Hyperparameter Tuning (Grid Search)
# ---------------------------------------------------------
print("Setting up Grid Search with Time Series Cross-Validation...")
xgb = XGBClassifier(random_state=42)

param_grid = {
    'max_depth': [2, 3, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200]
}

tscv = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=tscv,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# ---------------------------------------------------------
# 7. Execution & Feature Extraction
# ---------------------------------------------------------
print("Running Grid Search... (This may take a minute)")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\n--- TUNING RESULTS ---")
print(f"Best Parameters Found: {grid_search.best_params_}")

print("\n--- FEATURE IMPORTANCE ---")
importance = best_model.feature_importances_
for i, col in enumerate(features):
    print(f"{col}: {importance[i]:.4f}")
print("--------------------------\n")

# ---------------------------------------------------------
# 8. The Vectorized Backtest Simulator
# ---------------------------------------------------------
test_df = df.iloc[split_index:].copy()
test_df['Prediction'] = best_model.predict(X_test)

# Because our target is smoothed over 5 days, the model's daily predictions 
# will naturally flip much less frequently, drastically reducing our fee burden.
test_df['Position'] = test_df['Prediction'].shift(1)
test_df = test_df.dropna()

# Apply transaction fees
trading_fee_pct = 0.001
test_df['Position_Change'] = test_df['Position'].diff().abs().fillna(0)
test_df['Fees'] = test_df['Position_Change'] * trading_fee_pct

# Strategy return applies the daily market return to our held position
test_df['Strategy_Return'] = (test_df['Position'] * test_df['Daily_Return']) - test_df['Fees']

# Calculate Sharpe Ratio
annual_risk_free_rate = 0.02
daily_rf = annual_risk_free_rate / 252
excess_returns = test_df['Strategy_Return'] - daily_rf
annualized_sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
print(f"Optimized Sharpe Ratio: {annualized_sharpe:.3f}")

test_df['Cumulative_Market'] = (1 + test_df['Daily_Return']).cumprod()
test_df['Cumulative_Strategy'] = (1 + test_df['Strategy_Return']).cumprod()

# ---------------------------------------------------------
# 9. Plotting the Results
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(test_df.index, test_df['Cumulative_Market'], label='Buy & Hold ARKK', color='blue')
plt.plot(test_df.index, test_df['Cumulative_Strategy'], label=f'Macro Tuned XGBoost (Sharpe: {annualized_sharpe:.2f})', color='green')
plt.title('Macro Tuned ML Strategy vs. Buy & Hold (5-Day Horizon)')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()