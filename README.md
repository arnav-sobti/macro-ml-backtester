# Macro-Aware Algorithmic Trading: An XGBoost Approach to Regime Detection

## Executive Summary
This project is an end-to-end vectorized backtesting engine built in Python. The objective was to determine if combining technical momentum indicators with macroeconomic data could generate profitable regime-detection signals in highly volatile equities, explicitly accounting for a 0.1% transaction fee drag.

## Architecture & Data Pipeline
*   **Market Data:** Ingested via `yfinance`.
*   **Macroeconomic Data:** Fed Funds Rate, CPI, 10-Year Treasury, and Yield Curve Spread ingested via the St. Louis Fed API (`pandas-datareader`). Monthly data was forward-filled to prevent look-ahead bias.
*   **Validation:** Implemented `TimeSeriesSplit` cross-validation to prevent the model from overfitting to financial noise.
*   **Hyperparameter Tuning:** Automated via `GridSearchCV`.

## Phase 1: The Efficient Market Discovery (S&P 500)
Initial tests were run on the S&P 500 with a 1-day predictive horizon. The cross-validated XGBoost model recognized that the index is too efficient to trade daily against a 0.1% transaction fee. The mathematically optimal solution discovered by the algorithm was to default to a Buy & Hold strategy, effectively proving the Efficient Market Hypothesis.

## Phase 2: Resolving the Horizon Mismatch (ARKK)
The pipeline was pivoted to a highly volatile tech ETF (ARKK). Initially, the model moved entirely to cash during market rallies. Feature extraction revealed that the model heavily weighted Inflation (19%) and the Yield Curve (18.5%). 

**The Insight:** Feeding slow, monthly macroeconomic data into a daily classification target created a horizon mismatch. To resolve this, the predictive target was expanded to a 5-day horizon.

## Final Results
By predicting a 5-day horizon, the model successfully utilized macroeconomic data to detect risk-off regimes. 
*   It successfully moved to cash to avoid massive drawdowns in early 2025 and 2026.
*   It achieved a much smoother equity curve than the baseline asset.
*   **Final Optimized Sharpe Ratio:** 0.714
