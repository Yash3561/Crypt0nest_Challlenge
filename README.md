# Cryptocurrency Price Movement Prediction (BTC-USD)

## Project Overview

This project presents a Machine Learning model designed to predict the daily price movement (up or down) of Bitcoin (BTC-USD) using historical public price data. The primary data source is Yahoo Finance, accessed via the `yfinance` library.

The project follows a standard Machine Learning workflow:
1.  **Data Loading and Preprocessing:** Fetching and cleaning historical BTC-USD data.
2.  **Exploratory Data Analysis (EDA):** Visualizing data trends, distributions, and handling outliers.
3.  **Feature Engineering:** Creating a set of potentially predictive features and applying feature selection.
4.  **Model Training:** Employing two models: a baseline Logistic Regression and a Random Forest classifier, with considerations for time-series data and class balance.
5.  **Model Evaluation:** Assessing the models' predictive performance on unseen data using various metrics and visualizations.
6.  **Interpretability:** Analyzing feature influence to understand the models' decision-making processes.

This project was developed in response to the "AI & Tech Innovation Intern Challenge" by Tekly Studio x CryptoNest.io, with a focus on feature selection/engineering, model choice, evaluation metrics, and interpretability.

## Data

*   **Source:** Yahoo Finance (via `yfinance` library)
*   **Ticker Symbol:** BTC-USD
*   **Time Period:** Historical daily data from January 1, 2018, up to **May 26, 2025** (as per the last available data point from `yfinance` during execution, though "2025-05-27" was targeted).
*   **Data Fields Used:** Date, Open, High, Low, Close, Volume.

## Methodology

The analysis is implemented within the Jupyter Notebook (`Crypt0nest_Challenge.ipynb`).

### 1. Data Loading & Preprocessing
*   Historical OHLCV data for BTC-USD was downloaded. `Log_Volume` was created by applying a log transformation (`np.log1p`) to the 'Volume' to handle outliers.
*   The data was validated to ensure the range was as expected, noting the minor discrepancy in the final available date.

### 2. Exploratory Data Analysis (EDA)
*   **Closing Price Trend:** Visualized using line plots (linear and log scale) with annotations for significant market events, including a recent peak around May 22, 2025.
*   **Target Variable Distribution:** The "Price Up" vs. "Price Down/Same" distribution was plotted, showing a fairly balanced dataset.
*   **Volatility Analysis:** 30-day rolling volatility of the closing price was plotted.
*   **Stationarity Insight:** The closing price was plotted with its 30-day rolling mean.

### 3. Feature Engineering
A variety of features were engineered:
*   **Target Variable (`Price_Up`):** Binary (1 if next day's close > current day's close, 0 otherwise).
*   **Lagged Features:** `Close_Lag_{1, 3, 7, 14}` and `Log_Volume_Lag_{3, 7}` (Lags for Log_Volume 1 and 14 were dropped due to low correlation).
*   **Moving Averages (MA):** `MA_Close_{7, 14, 30}` (MAs for Log_Volume were dropped due to low correlation).
*   **Daily Return:** Percentage change in closing price.
*   **MACD:** `MACD_Hist` (MACD and Signal_Line were dropped due to low correlation).
*   **NaN Handling:** Lagged features were forward-filled. Rows with any remaining NaNs were dropped (~30 rows), resulting in 2673 data points for modeling.
*   **Feature Selection:** Features with an absolute correlation to the target variable of less than `0.01` were dropped. This resulted in 12 features being selected for the models.
    *   **Final Features Used:** `['Log_Volume', 'Close_Lag_1', 'Close_Lag_3', 'Log_Volume_Lag_3', 'Close_Lag_7', 'Log_Volume_Lag_7', 'Close_Lag_14', 'MA_Close_7', 'MA_Close_14', 'MA_Close_30', 'Daily_Return', 'MACD_Hist']`.

### 4. Model Training
*   **Train-Test Split:** Data was split chronologically (80% training: 2018-01-30 to 2023-12-07; 20% testing: 2023-12-08 to 2025-05-25).
*   **Feature Scaling:** `StandardScaler` was applied to the features.
*   **Models Trained:**
    1.  `LogisticRegression` (with `class_weight='balanced'`).
    2.  `RandomForestClassifier` (with `n_estimators=100`, `class_weight='balanced_subsample'`).
*   **Cross-Validation:** `TimeSeriesSplit` (5 folds) was used on the training data for both models:
    *   LogReg CV: Avg. Accuracy ~0.4904, Avg. ROC AUC ~0.5088.
    *   RF CV: Avg. Accuracy ~0.4815, Avg. ROC AUC ~0.4826.

### 5. Model Evaluation
Models were assessed on the held-out test set. Confusion matrix heatmaps and ROC curves were plotted for both.

#### Results Summary (Test Set)

| Metric         | Logistic Regression | Random Forest |
|----------------|---------------------|---------------|
| Accuracy       | 0.5009              | 0.4991        |
| ROC AUC        | 0.5518              | 0.5352        |
| Precision (Up) | 0.67                | 0.58          |
| Recall (Up)    | 0.09                | 0.14          |
| F1-score (Up)  | 0.15                | 0.23          |

*   **Logistic Regression (Test Set):**
    *   Confusion Matrix: TN=244, FP=12, FN=255, TP=24.
*   **Random Forest (Test Set):**
    *   Confusion Matrix: TN=227, FP=29, FN=239, TP=40.
*   **Overall Performance:** Both models performed close to random chance on accuracy. Logistic Regression showed a slightly better ROC AUC on the test set. The majority class baseline accuracy on the test set was ~0.5215. Both models struggled with recall for the "Price Up" class.

### 6. Interpretability

*   **Logistic Regression Coefficients:**
    *   Top positive influencers for "Price Up": `MA_Close_30` (0.23), `Close_Lag_1` (0.19), `Log_Volume` (0.17).
    *   Top negative influencers for "Price Up" (suggesting Price Down): `Close_Lag_7` (-0.49), `Log_Volume_Lag_3` (-0.17), `Daily_Return` (-0.12).
    This suggests a mix of medium-term trend (MA_Close_30) and short-term mean-reversion (Close_Lag_7, Daily_Return) signals.
*   **Random Forest Feature Importances (Gini):**
    *   Top influential features: `Daily_Return` (0.103), `Log_Volume_Lag_3` (0.095), `Log_Volume_Lag_7` (0.091), `MACD_Hist` (0.090), `Log_Volume` (0.090).
    The RF model highlighted recent returns and lagged log-volumes as most important.

## Assumptions and Limitations

### Assumptions
*   Historical price/volume data contains some signal predictive of future short-term price direction.
*   The engineered features are capable of capturing parts of this signal.

### Limitations
*   **Model Simplicity:** While Random Forest can capture non-linearities, both models are relatively simple given market complexity.
*   **Feature Scope:** Only OHLCV data was used. External factors (news, sentiment, macroeconomic data, on-chain metrics) were not included.
*   **Market Efficiency:** Financial markets are highly efficient, making consistent prediction very challenging.
*   **No Extensive Hyperparameter Tuning:** Baseline parameters were primarily used.
*   **Fixed Train/Test Split:** While TimeSeriesSplit CV was used on training data, a full walk-forward validation on the entire dataset would offer a more robust backtest.
*   **Data Availability:** The targeted end date of 2025-05-27 was not fully available from `yfinance` at runtime; the model used data up to 2025-05-26.

## What You'd Do Next With More Time

*   **Advanced Feature Engineering:** More complex technical indicators, interaction terms, market regime features.
*   **Alternative Data Sources:** Integrate sentiment data, macroeconomic indicators, on-chain data.
*   **More Sophisticated Models:** Experiment with Gradient Boosting (XGBoost, LightGBM), LSTMs, or GRUs.
*   **Robust Validation & Backtesting:** Implement full walk-forward validation.
*   **Hyperparameter Optimization:** Systematic tuning for chosen models.
*   **CryptoNest.io Integration:** Explore integrating probability scores from the model into CryptoNest.io's platform as part of trading signals or risk indicators, potentially involving API development or dashboard integration.
*   **Address Low Recall:** Investigate techniques to improve the model's ability to identify "Price Up" days, such as different class weighting strategies, oversampling/undersampling (with care for time series), or adjusting decision thresholds.

## How to Run
1.  Ensure Python (3.x) is installed.
2.  Install the necessary libraries:
    ```bash
    pip install pandas numpy yfinance matplotlib seaborn scikit-learn jupyter
    ```
3.  Launch Jupyter Notebook or Jupyter Lab.
4.  Open the `Crypt0nest_Challenge.ipynb` file and run the cells sequentially. An active internet connection is required for `yfinance` to download data.
