# Cryptocurrency Price Movement Prediction (BTC-USD)

## Project Overview

This project presents a simple Machine Learning model designed to predict the daily price movement (up or down) of Bitcoin (BTC-USD) using historical public price data. The primary data source is Yahoo Finance, accessed via the `yfinance` library.

The project follows a standard Machine Learning workflow:
1.  **Data Loading and Preprocessing:** Fetching and cleaning historical BTC-USD data.
2.  **Exploratory Data Analysis (EDA):** Visualizing data trends, distributions, and checking for anomalies.
3.  **Feature Engineering:** Creating a set of potentially predictive features from raw price and volume data.
4.  **Model Training:** Employing a baseline Logistic Regression model, with considerations for time-series data and class balance.
5.  **Model Evaluation:** Assessing the model's predictive performance on unseen data using various metrics.
6.  **Interpretability:** Analyzing feature importance to understand the model's decision-making process.

This project was developed in response to the "Cryptocurrency/Stock Price Movement Prediction Challenge," with a focus on feature selection/engineering, appropriate model choice (baseline accepted), robust evaluation metrics, and model interpretability.

## Data

*   **Source:** Yahoo Finance (via `yfinance` library)
*   **Ticker Symbol:** BTC-USD
*   **Time Period:** Historical daily data from January 1, 2018, up to the day before the last execution (e.g., May 27, 2025, as per notebook outputs, ensuring only complete historical data is used).
*   **Data Fields Used:** Date, Open, High, Low, Close, Volume.

## Methodology

The analysis is implemented within a Jupyter Notebook (`Crypt0nest_Challenge (1).ipynb`).

### 1. Data Loading & Preprocessing
*   Historical OHLCV (Open, High, Low, Close, Volume) data for BTC-USD was downloaded using `yfinance`.
*   `auto_adjust=True` was used for simplicity, providing adjusted prices.
*   Column names were flattened if `yfinance` returned a MultiIndex.
*   The dataset was confirmed to have no missing values in the core OHLCV fields for the selected period.

### 2. Exploratory Data Analysis (EDA)
*   **Closing Price Trend:** Visualized using a line plot, also on a log scale to observe long-term growth patterns. Key market events (like BTC Peak and FTX Collapse) were annotated.
*   **Target Variable Distribution:** The distribution of the "Price Up" (next day's price increase) vs. "Price Down/Same" was plotted, showing a relatively balanced dataset.
*   **Volatility Analysis:** A 30-day rolling standard deviation of the closing price was plotted to visualize periods of high and low volatility.
*   **Outlier Check:** A basic outlier check on 'Volume' (values beyond 3 standard deviations from the mean) was performed.
*   **Stationarity Insight:** The closing price was plotted alongside its 30-day rolling mean to visually assess trends and potential non-stationarity.

### 3. Feature Engineering
A variety of features were engineered to capture different aspects of price and volume dynamics:
*   **Target Variable (`Price_Up`):** A binary variable (1 if the next day's close > current day's close, 0 otherwise).
*   **Lagged Features:**
    *   `Close_Lag_{1, 3, 7, 14}`: Closing prices from 1, 3, 7, and 14 days prior.
    *   `Volume_Lag_{1, 3, 7, 14}`: Trading volumes from 1, 3, 7, and 14 days prior.
*   **Moving Averages (MA):**
    *   `MA_Close_{7, 14, 30}`: Simple moving averages of closing price.
    *   `MA_Volume_{7, 14, 30}`: Simple moving averages of volume.
*   **Daily Return:** Percentage change in closing price from the previous day.
*   **Volatility:**
    *   `Volatility_{7, 30}`: Rolling standard deviation of closing price.
*   **Relative Strength Index (RSI):**
    *   `RSI_14`: 14-day RSI to measure price momentum and overbought/oversold conditions.
*   **Moving Average Convergence Divergence (MACD):**
    *   `MACD`, `Signal_Line`, `MACD_Hist`: To identify trend direction and momentum.
*   **Time-Based Feature:**
    *   `Day_of_Week`: Integer representing the day of the week (Monday=0, Sunday=6).
*   **NaN Handling:** Lagged features were initially forward-filled. Subsequently, all rows containing any NaN values (primarily from the initial periods of rolling window calculations and the last row's target) were dropped. This resulted in approximately 30 rows being removed.

### 4. Model Training
*   **Features (X):** All engineered features, excluding raw OHLCV data for the current day, helper columns for MACD (EMA_12, EMA_26), and the `Future_Close` column.
*   **Target (y):** The binary `Price_Up` variable.
*   **Train-Test Split:** The data was split chronologically, with the first 80% used for training and the subsequent 20% for testing, to simulate real-world prediction and avoid look-ahead bias.
*   **Feature Scaling:** `StandardScaler` from `scikit-learn` was applied to standardize the feature set.
*   **Model Choice:** A `LogisticRegression` model was selected as an interpretable baseline. `class_weight='balanced'` was used to handle any potential class imbalance in the target variable.
*   **Cross-Validation:** `TimeSeriesSplit` (with 5 splits) was performed on the training data to assess the stability of the model's performance. The average CV Accuracy was approximately 0.4978 (+/- 0.0167) and ROC AUC was approximately 0.4977 (+/- 0.0295).

### 5. Model Evaluation
The trained model's performance was evaluated on the held-out test set:
*   **Accuracy:** 0.4991
*   **ROC AUC Score:** 0.5398
*   **Confusion Matrix (Test Set):**
    *   True Negatives (Predicted Down, Actual Down): 243
    *   False Positives (Predicted Up, Actual Down): 13
    *   False Negatives (Predicted Down, Actual Up): 255
    *   True Positives (Predicted Up, Actual Up): 24
*   **Classification Report (Test Set):**
    *   **Price Down (0):** Precision 0.49, Recall 0.95, F1-score 0.65
    *   **Price Up (1):** Precision 0.65, Recall 0.09, F1-score 0.15
*   **Baseline Comparison:** The test set had a majority class proportion of ~0.5215 (for "Price Up"). The model's accuracy (0.4991) was slightly below this naive baseline.
*   **Overall Performance:** The evaluation metrics (accuracy near 50%, ROC AUC near 0.54) indicate that the baseline Logistic Regression model, even with the engineered features, has very limited predictive power and performs close to random chance. This is a common outcome for simple models in complex financial markets.

### 6. Interpretability
Feature coefficients from the trained Logistic Regression model were examined to understand their influence (note: coefficients reflect changes in log-odds per one standard deviation change in the scaled feature):
*   **Strongest Positive Influence (suggesting Price Up):**
    *   `MA_Close_30` (Coefficient: ~0.25)
    *   `MA_Volume_14` (Coefficient: ~0.17)
    *   `MA_Volume_30` (Coefficient: ~0.15)
    *   `Volatility_7` (Coefficient: ~0.11)
    *   `RSI_14` (Coefficient: ~0.11)
*   **Strongest Negative Influence (suggesting Price Down):**
    *   `Close_Lag_7` (Coefficient: ~-0.52)
    *   `Volume_Lag_3` (Coefficient: ~-0.18)
    *   `Daily_Return` (Coefficient: ~-0.14)
    *   `Close_Lag_1` (Coefficient: ~-0.04, though weaker in this run)
*   **Insight:** The model appears to assign some positive weight to medium-to-longer-term moving averages of price and volume, and recent volatility/RSI, potentially indicating trend or momentum signals. Conversely, strong negative weights for features like `Close_Lag_7` and `Daily_Return` suggest mean-reversion tendencies (i.e., if the price was high 7 days ago or had a strong positive return today, the model leans towards predicting a down day tomorrow).

## Assumptions and Limitations

### Assumptions
*   Historical price/volume data contains discernible patterns that can predict future short-term directional movements.
*   The engineered features (lags, MAs, technical indicators) can capture these patterns.

### Limitations
*   **Model Simplicity:** Logistic Regression is a linear model and may not capture the complex, non-linear dynamics inherent in financial markets.
*   **Feature Scope:** The model relies solely on OHLCV data. It does not incorporate external factors such as market sentiment (news, social media), macroeconomic data (interest rates, inflation), or on-chain metrics for cryptocurrencies (transaction volumes, active addresses, network health).
*   **Market Efficiency:** Financial markets are often considered (at least semi-strong form) efficient, making it exceedingly difficult to achieve consistent predictive accuracy beyond random chance.
*   **Non-Stationarity:** While features like returns and RSI inherently address some aspects of non-stationarity, formal tests and advanced transformations were not deeply explored.
*   **No Hyperparameter Tuning:** The Logistic Regression model used default or common-sense parameters (e.g., `class_weight='balanced'`).
*   **Fixed Train/Test Split:** Performance was evaluated on a single chronological split. While `TimeSeriesSplit` was used on training data for stability checks, a full walk-forward validation or multiple rolling-window backtests would provide more robust out-of-sample performance estimates.
*   **Binary Outcome:** The model predicts only the direction (up or down), not the magnitude of the price change or probabilities for risk assessment.
*   **Data Nuances:** The `yfinance` data source, depending on the execution date, might include forward-looking or placeholder data if `end_date` is set to "today". The code attempts to mitigate this by setting `actual_end_date` to yesterday.

## What You'd Do Next With More Time

*   **Advanced Feature Engineering:**
    *   Explore a wider array of technical indicators (e.g., Bollinger Bands, Stochastic Oscillator, Fibonacci levels, Ichimoku Cloud).
    *   Create interaction terms between features or ratios (e.g., price relative to moving average).
    *   Incorporate features representing different market regimes or volatility states.
*   **Alternative Data Sources:**
    *   Integrate sentiment analysis scores from financial news or social media platforms.
    *   Include relevant macroeconomic indicators.
    *   For cryptocurrencies, leverage on-chain data.
*   **More Sophisticated Models:**
    *   Experiment with tree-based ensemble methods: Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost), which can capture non-linearities.
    *   Explore time-series specific models like ARIMA (if predicting values rather than direction) or advanced sequence models like LSTMs or GRUs (Recurrent Neural Networks), especially if more granular data is used.
*   **Robust Validation & Backtesting:**
    *   Implement rigorous time-series cross-validation techniques like walk-forward validation or k-fold with chronological blocking.
    *   Conduct thorough backtesting, considering look-ahead bias and data snooping.
*   **Hyperparameter Optimization:**
    *   Use systematic methods like GridSearchCV, RandomizedSearchCV, or Bayesian optimization, ensuring time-series integrity during validation.
*   **Dimensionality Reduction:** If a very large feature set is created, techniques like PCA or feature selection based on importance could be applied.
*   **Portfolio & Risk Management Context:** If used for trading, incorporate transaction costs, slippage, and position sizing strategies.

## How to Run
1.  Ensure Python (3.x) is installed.
2.  Install the necessary libraries. It's recommended to use a virtual environment:
    ```bash
    pip install pandas numpy yfinance matplotlib seaborn scikit-learn jupyter
    ```
3.  Launch Jupyter Notebook or Jupyter Lab:
    ```bash
    jupyter notebook
    # OR
    jupyter lab
    ```
4.  Open the `Crypt0nest_Challenge (1).ipynb` file and run the cells sequentially. An active internet connection is required for `yfinance` to download data.
