# Cryptocurrency Price Movement Prediction (BTC-USD)

## Overview

This project aims to build a simple Machine Learning model to predict the daily price movement (up or down) of Bitcoin (BTC-USD) based on its public historical price data. The primary data source is Yahoo Finance, accessed via the `yfinance` library.

The project follows a standard ML workflow:
1.  **Data Loading and Preprocessing:** Fetching historical BTC-USD data.
2.  **Exploratory Data Analysis (EDA):** Initial visualization and checks.
3.  **Feature Engineering:** Creating predictive features from raw price/volume data.
4.  **Model Training:** Using a baseline Logistic Regression model.
5.  **Model Evaluation:** Assessing performance using various metrics.
6.  **Interpretability:** Understanding feature influence on the model's predictions.

This project was completed as a response to the "Cryptocurrency/Stock Price Movement Prediction Challenge." The focus areas were feature selection/engineering, model choice (baseline allowed), evaluation metrics, and interpretability.

## Data

*   **Source:** Yahoo Finance (`yfinance` library)
*   **Ticker:** BTC-USD
*   **Period:** Data from 2018-01-01 up to the date of the last run (e.g., 2025-05-25 in the provided notebook outputs, which might reflect future-dated data in that specific run; the model itself uses a chronological train/test split on this data).
*   **Fields Used:** Date, Open, High, Low, Close, Volume.

## Methodology

The project is implemented in a Jupyter Notebook (`Crypt0nest_Challenge.ipynb`).

### 1. Data Loading & Preprocessing
*   Historical OHLCV (Open, High, Low, Close, Volume) data for BTC-USD was downloaded using `yfinance`.
*   Potential MultiIndex columns from `yfinance` were flattened for easier use.
*   The data was checked for missing values; none were found in the relevant period.

### 2. Feature Engineering
The core of the predictive power relies on features engineered from historical data:
*   **Target Variable (`Price_Up`):** A binary variable indicating if the next day's closing price is higher than the current day's closing price (1 for up, 0 for down/same).
*   **Lagged Features:**
    *   `Close_Lag_{1, 3, 7, 14}`: Closing prices from 1, 3, 7, and 14 days prior.
    *   `Volume_Lag_{1, 3, 7, 14}`: Trading volumes from 1, 3, 7, and 14 days prior.
*   **Moving Averages (MA):**
    *   `MA_Close_{7, 14, 30}`: Simple moving averages of the closing price over 7, 14, and 30-day windows.
    *   `MA_Volume_{7, 14, 30}`: Simple moving averages of volume over 7, 14, and 30-day windows.
*   **Daily Return:** Percentage change in the closing price from the previous day.
*   **Volatility:**
    *   `Volatility_{7, 30}`: Rolling standard deviation of the closing price over 7 and 30-day windows.
*   Rows with NaN values resulting from these shifting/rolling operations were dropped before model training.

### 3. Model Training
*   **Features (X):** The engineered features listed above.
*   **Target (y):** The `Price_Up` variable.
*   **Train-Test Split:** Data was split chronologically (80% for training, 20% for testing) to prevent data leakage and simulate real-world prediction.
*   **Feature Scaling:** `StandardScaler` was applied to the features before training the model.
*   **Model Choice:** `LogisticRegression` from `scikit-learn` was chosen as an interpretable baseline model.

### 4. Model Evaluation
The model's performance was assessed on the test set using several metrics:
*   **Accuracy:** ~49.91%
*   **ROC AUC Score:** ~0.5320
*   **Confusion Matrix:**
    *   True Negatives (Predicted Down, Actual Down): 241
    *   False Positives (Predicted Up, Actual Down): 15
    *   False Negatives (Predicted Down, Actual Up): 253
    *   True Positives (Predicted Up, Actual Up): 26
*   **Classification Report:**
    *   The model showed a high recall (0.94) for predicting "Price Down" but a very low recall (0.09) for "Price Up".
    *   Precision for "Price Up" was 0.63 when it did predict an upward movement, but this was infrequent.
*   **Overall Performance:** The model performed close to random chance, which is common for simple models in financial markets without extensive feature engineering or more complex modeling techniques. The test set was fairly balanced (approx. 52% 'Up' days).

### 5. Interpretability
For the Logistic Regression model, feature coefficients were examined:
*   **Positive Predictors (increase log-odds of Price_Up=1):** `MA_Close_14` (0.28), `MA_Volume_14` (0.23), `Volatility_7` (0.10).
*   **Negative Predictors (decrease log-odds of Price_Up=1):** `Close_Lag_7` (-0.26), `Volume_Lag_3` (-0.18), `Close_Lag_1` (-0.14), `Daily_Return` (-0.12).
*   This suggests the model picked up on some (weak) trend-following signals from medium-term MAs and some mean-reversion signals from recent price/return changes.

## Assumptions and Limitations

### Assumptions
*   Historical price and volume data contains some signal predictive of future short-term price direction.
*   The engineered features (lags, MAs, returns, volatility) are capable of capturing parts of this signal.

### Limitations
*   **Simple Model:** Logistic Regression is linear and may not capture complex non-linear market dynamics.
*   **Limited Features:** Only OHLCV data was used. External factors like news sentiment, macroeconomic data, or on-chain metrics (for crypto) were not included.
*   **Stationarity:** Financial time series are often non-stationary. While scaling was used, formal stationarity tests and transformations (e.g., differencing) were not extensively applied beyond calculating returns.
*   **Market Efficiency:** Financial markets are generally considered (at least semi-strong) efficient, making consistent prediction very challenging.
*   **No Hyperparameter Tuning:** The baseline model used default or common-sense parameters.
*   **Fixed Train/Test Split:** Performance might vary with different split points or using more robust time-series cross-validation.
*   **Binary Outcome:** The model predicts only "up" or "down," not the magnitude of the price change.
*   **Data Scope:** The specific dataset used in the notebook run included some future-dated entries from `yfinance`. While the train/test split was chronological on this data, using purely historical, validated data is standard for backtesting.

## What You'd Do Next With More Time

*   **Advanced Feature Engineering:**
    *   Incorporate more technical indicators: RSI, MACD, Bollinger Bands, Fibonacci retracements, etc.
    *   Create interaction terms between features.
    *   Add time-based features (e.g., day of the week, month, volatility regimes).
*   **Alternative Data Sources:**
    *   Integrate sentiment data from news articles or social media (e.g., Twitter, Reddit).
    *   Include relevant macroeconomic indicators (e.g., interest rates, inflation).
    *   For cryptocurrencies, explore on-chain data (e.g., transaction volumes, active addresses).
*   **More Sophisticated Models:**
    *   Tree-based ensembles: Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost).
    *   Time-series specific models if predicting actual values (ARIMA, Prophet), or recurrent neural networks (LSTMs, GRUs) for sequence modeling even for classification.
*   **Robust Evaluation & Validation:**
    *   Implement time-series cross-validation (e.g., walk-forward validation) for more reliable performance estimates.
*   **Hyperparameter Optimization:**
    *   Use techniques like GridSearchCV or RandomizedSearchCV, or Bayesian optimization.
*   **Address Class Imbalance (if present):** If the target becomes imbalanced with different definitions, use techniques like SMOTE or class weighting.
*   **Dimensionality Reduction:** If a very large number of features are generated (e.g., PCA).
*   **Consider Transaction Costs:** If developing a trading strategy, factor in trading fees and slippage.

## How to Run
1.  Ensure you have Python installed.
2.  Install the necessary libraries:
    ```bash
    pip install pandas numpy yfinance matplotlib seaborn scikit-learn
    ```
3.  Open and run the `Crypt0nest_Challenge.ipynb` Jupyter Notebook in an environment like Jupyter Lab, Jupyter Notebook, or Google Colab.
