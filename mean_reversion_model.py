import os
import pandas as pd
import yfinance as yf
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.metrics import r2_score


# Step 1: Define crypto tickers
crypto_tickers = ['WETH-USD', 'WBTC-USD']
print(f"Number of crypto assets: {len(crypto_tickers)}")

# Step 2: Setup Google Drive directory for persistent storage
save_dir = 'crypto_data'
os.makedirs(save_dir, exist_ok=True)

# Step 3: Download historical data for all S&P 500 stocks
start_date = "2019-01-01"
end_date = "2023-12-31"
batch_size = 50  # Process in batches to reduce memory usage and API calls
def download_and_save_data(tickers, save_dir):
    for ticker in tickers:
        print(f"Downloading data for {ticker}")
        try:
            # Download data for the crypto
            data = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                interval="1d"
            )
            if not data.empty:
                # Save to crypto_data directory
                data.to_csv(f"{save_dir}/{ticker}.csv")
                print(f"Saved {ticker} data to {save_dir}")
            else:
                print(f"No data found for {ticker}")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
        # Add a small delay between downloads
        time.sleep(2)

# Execute the download and save process
download_and_save_data(crypto_tickers, save_dir)

print("All crypto downloads complete. Data saved to crypto_data directory!")


# Path to the folder containing crypto CSV files
data_folder = 'crypto_data'

# Initialize an empty list to store individual DataFrames
all_dataframes = []

# Iterate over all CSV files in the folder
for file_name in os.listdir(data_folder):
    if file_name.endswith('.csv'):  # Ensure the file is a CSV
        file_path = os.path.join(data_folder, file_name)

        # Read the CSV into a DataFrame
        stock_data = pd.read_csv(file_path)

        # Add a column for the ticker symbol (from the file name)
        ticker = file_name.split('.')[0]  # Extract the ticker from the file name
        stock_data['Ticker'] = ticker

        # Append to the list of DataFrames
        all_dataframes.append(stock_data)

# Concatenate all DataFrames into a single DataFrame
combined_data = pd.concat(all_dataframes, ignore_index=True)

# Show the first few rows of the combined DataFrame
print(combined_data.head())

# Path to save the combined dataset
combined_file_path = 'crypto_combined_data.csv'

# Save the combined DataFrame to a CSV file
combined_data.to_csv(combined_file_path, index=False)

print(f"Combined data saved to {combined_file_path}")

# Load the CSV file
file_path = 'crypto_combined_data.csv'
combined_data = pd.read_csv(file_path)

# Display the first few rows to verify
print(combined_data.head())

def calculate_channel_bound_metric(data, window=20):
    """
    Calculates a channel-bound metric based on the coefficient of variation of ATR.

    Args:
        data (pd.DataFrame): DataFrame containing stock data with 'High', 'Low', 'Close' columns.
        window (int): Rolling window size for ATR and CV calculation.

    Returns:
        pd.Series: Channel-bound metric for each data point.
    """
    # Calculate ATR
    data['TR'] = np.maximum(
        np.maximum(data['High'] - data['Low'], np.abs(data['High'] - data['Close'].shift(1))),
        np.abs(data['Low'] - data['Close'].shift(1))
    )
    data['ATR'] = data['TR'].rolling(window=window, min_periods=window).mean()

    # Calculate Coefficient of Variation of ATR
    data['ATR_CV'] = data['ATR'].rolling(window=window, min_periods=window).std() / data['ATR']

    # Channel-bound metric (lower is better)
    data['Channel_Bound_Metric'] = data['ATR_CV']

    return data['Channel_Bound_Metric']

# Assuming 'combined_data' is your DataFrame from earlier steps

# Calculate the channel-bound metric for all stocks
combined_data['Channel_Bound_Metric'] = combined_data.groupby('Ticker').apply(calculate_channel_bound_metric).reset_index(level=0, drop=True)

# Set a threshold (e.g., 0.2)
threshold = 0.07

# Filter stocks
filtered_tickers = combined_data.groupby('Ticker')['Channel_Bound_Metric'].mean()[combined_data.groupby('Ticker')['Channel_Bound_Metric'].mean() < threshold].index.tolist()

print(f"Filtered tickers: {filtered_tickers}")

# Count the number of filtered tickers
num_filtered_tickers = len(filtered_tickers)

# Print the count
print(f"Number of filtered tickers: {num_filtered_tickers}")

def calculate_features_and_labels(data):
    # Ensure proper column names and data types
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by='Date', inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Calculate daily returns
    data['Close'] = data['Close'].fillna(method='ffill')
    data['Return'] = data['Close'].pct_change()
    data['Next_Day_Return'] = data['Return'].shift(-1)

    # Scale the returns using tanh
    data['Signal'] = np.tanh(data['Next_Day_Return'] * 10)

    # Rolling features (using past data only)
    data['Rolling_Mean'] = data['Close'].rolling(window=20, min_periods=20).mean().shift(1) # Shifted by 1 to avoid using current day
    data['Rolling_Std'] = data['Close'].rolling(window=20, min_periods=20).std().shift(1)
    data['Z-Score'] = (data['Close'].shift(1) - data['Rolling_Mean']) / data['Rolling_Std']  # Shifted Close

    # Volume Spike (using past data for Avg_Volume)
    data['Avg_Volume'] = data['Volume'].rolling(window=20, min_periods=20).mean().shift(1)
    data['Volume_Spike'] = data['Volume'].shift(1) / data['Avg_Volume']

    # EMA and MACD
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean().shift(1)
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean().shift(1)
    data['MACD'] = data['EMA_12'] - data['EMA_26']

    # DMA
    data['DMA'] = (data['Close'].shift(1) - data['Rolling_Mean']) / data['Rolling_Mean']

    # RSI (using past data)
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean().shift(1)
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean().shift(1)
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Drop rows with NaN values (from rolling calculations)
    data.dropna(inplace=True)
    return data

    # Combine all CSVs into one DataFrame
def combine_and_process_all_data(folder_path):
    combined_data = pd.DataFrame()
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            stock_data = pd.read_csv(file_path)
            stock_data['Ticker'] = file_name.split('.')[0]  # Extract ticker from file name

            # Apply feature engineering
            stock_data = calculate_features_and_labels(stock_data)
            combined_data = pd.concat([combined_data, stock_data], ignore_index=True)
    return combined_data

# Define folder path and process all data
folder_path = 'crypto_data'
processed_data = combine_and_process_all_data(folder_path)

# Save the processed data to a CSV
processed_data.to_csv('crypto_processed_data.csv', index=False)
print("Processed data saved to crypto_processed_data.csv")

# Show a preview of the data
print(processed_data.head())

print(processed_data.shape)

print(processed_data.isnull().sum())

# Check for outliers in key features
features_to_check = ['Next_Day_Return', 'Volume', 'Z-Score', 'ATR', 'MACD']

for feature in features_to_check:
    q1 = processed_data[feature].quantile(0.25)
    q3 = processed_data[feature].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    print(f"{feature}: Outliers below {lower_bound:.2f} or above {upper_bound:.2f}")

# Example: Cap extreme values in 'Return'
processed_data['Next_Day_Return'] = processed_data['Next_Day_Return'].clip(lower_bound, upper_bound)
processed_data.shape

# Plot the distribution of 'Return'
sns.histplot(processed_data['Next_Day_Return'], kde=True)
plt.title('Distribution of Return')
plt.show()

print(processed_data.columns)
print(processed_data.shape)

print(processed_data.head())
print(f"Final data shape: {processed_data.shape}")

# Drop unnecessary columns
columns_to_drop = [
      'Date', 'Open', 'High', 'Low', 'Adj Close', 'Ticker', 'Next_Day_Return', 'Return'
]

processed_data = processed_data.drop(columns=columns_to_drop, errors='ignore')

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Assuming your features are in a pandas DataFrame called 'df'
features_to_standardize = ['Close', 'Volume', 'Rolling_Mean', 'Rolling_Std',
                           'Avg_Volume', 'Volume_Spike', 'EMA_12', 'EMA_26',
                           'MACD', 'DMA']

features_to_scale = ['Z-Score', 'RSI']  # These are already somewhat normalized

# Create scaler objects
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

# Apply standardization
processed_data[features_to_standardize] = standard_scaler.fit_transform(processed_data[features_to_standardize])

# Apply scaling
processed_data[features_to_scale] = minmax_scaler.fit_transform(processed_data[features_to_scale])

print(processed_data.head())
print(f"Final data shape: {processed_data.shape}")
processed_data.columns

# Verify final dataset
print(processed_data.head())
print(f"Final data shape: {processed_data.shape}")

from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
X = processed_data.drop(columns=['Signal'])  # Features
y = processed_data['Signal']                # Target

# Step 1: Split into Train+Validation and Test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # Use 20% of the data for the test set
)

# Step 2: Further split Train+Validation into Training and Validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42  # Use 20% of Train+Validation for validation
)

# Print the shapes of the datasets
print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Validation set shape: X_val={X_val.shape}, y_val={y_val.shape}")
print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")

print(X_train.columns)


# Convert datasets to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)
# Define XGBoost parameters
params = {
    "objective": "reg:squarederror",  # Regression objective
    "eval_metric": "rmse",           # Evaluation metric
    "learning_rate": 0.1,
    "max_depth": 6,
    "seed": 42                       # For reproducibility
}
# Specify evaluation sets
eval_set = [(dtrain, "train"), (dval, "eval")]

# Train the model
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,            # Maximum number of boosting rounds
    evals=eval_set,                 # Evaluation sets
    early_stopping_rounds=10,       # Stop if no improvement for 10 rounds
    verbose_eval=True               # Print progress during training
)
# Predict on the test set
y_pred = model.predict(dtest)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Test Set Mean Squared Error: {mse:.4f}")

model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric="rmse"
)

# Perform 5-fold cross-validation
scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)

# Convert negative MSE to positive
mse_scores = -scores
print(f"Cross-Validation Mean MSE: {mse_scores.mean():.4f}")

# Train the model on the full training set
model.fit(X_train, y_train)

# Get feature names from original DataFrame (before conversion to NumPy array)
# Assuming 'processed_data' is your original DataFrame before splitting
feature_names = processed_data.drop(columns=['Signal']).columns

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance using Seaborn
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))  # Show top 15 features
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

# Save the model in binary format
model.save_model('/content/drive/MyDrive/xgboost_model_2024.model')

print("Model saved successfully!")

# # Load the saved model
# loaded_model = xgb.Booster()
# loaded_model.load_model('/content/drive/MyDrive/xgboost_model_2024.model')

# print("Model loaded successfully!")


# Add random noise to the features
X_train_noisy = X_train + np.random.normal(0, 0.01, X_train.shape)

# Retrain the model with noisy data
model.fit(X_train_noisy, y_train)

# Evaluate on validation set
y_pred_noisy = model.predict(X_val)
mse_noisy = mean_squared_error(y_val, y_pred_noisy)
print(f"Validation MSE with Noise: {mse_noisy:.4f}")

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,  # 3-fold cross-validation
    verbose=1
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Display best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation MSE:", -grid_search.best_score_)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")