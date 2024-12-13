import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time

# Step 1: Define the folder and date range
start_date = "2024-01-01"
end_date = "2024-06-01"
date_range_name = f"{start_date.replace('-', '')}_to_{end_date.replace('-', '')}"
data_folder = f'6m_data_{date_range_name}'

# Step 2: Fetch S&P 500 stock tickers
def get_sp500_tickers():
    # Define crypto tickers for WETH and WBTC over USDT
    crypto_tickers = ['WETH', 'WBTC']
    print(f"Number of crypto assets: {len(crypto_tickers)}")
    return crypto_tickers

# Step 3: Download historical data
def download_and_save_data(tickers, save_dir, start_date, end_date):
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    batch_size = 50  # Process in batches to avoid hitting API limits
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i + batch_size]
        print(f"Downloading batch {i // batch_size + 1}: {batch_tickers}")
        try:
            # Download data for the batch
            data = yf.download(
                tickers=batch_tickers,
                start=start_date,
                end=end_date,
                interval="1d",
                group_by="ticker"
            )
            # Save each stock's data to Google Drive
            for ticker in batch_tickers:
                try:
                    ticker_data = data[ticker]
                    if not ticker_data.empty:
                        ticker_data.to_csv(f"{save_dir}/{ticker}.csv")
                        print(f"Saved {ticker} data to {save_dir}")
                except Exception as e:
                    print(f"Error saving data for {ticker}: {e}")
        except Exception as e:
            print(f"Error downloading batch {i // batch_size + 1}: {e}")
        # Add a delay to prevent hitting API rate limits
        time.sleep(10)

def calculate_features(data):
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

def combine_and_process_data(folder_path):
    combined_data = pd.DataFrame()
    tickers_dates = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            stock_data = pd.read_csv(file_path)

            # Extract ticker and save dates
            ticker = file_name.split('.')[0]
            stock_data['Ticker'] = ticker
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            tickers_dates.extend(stock_data[['Ticker', 'Date']].values.tolist())

            # Apply feature engineering
            stock_data = calculate_features(stock_data)
            combined_data = pd.concat([combined_data, stock_data], ignore_index=True)

    # Save tickers and dates
    tickers_dates_df = pd.DataFrame(tickers_dates, columns=['Ticker', 'Date'])
    tickers_dates_df = tickers_dates_df[tickers_dates_df['Date'].isin(combined_data['Date'])]
    tickers_dates_path = f"/content/drive/MyDrive/sp500_tickers_dates_{date_range_name}.csv"
    tickers_dates_df.to_csv(tickers_dates_path, index=False)
    print(f"Tickers and dates saved to: {tickers_dates_path}")

    return combined_data

def scale_and_standardize_data(data):
    # Retain the Date and Ticker columns temporarily
    date_ticker = data[['Date', 'Ticker']]

    # Retain only the necessary columns for the final dataset
    required_columns = [
        'Date', 'Ticker',           # Retain for tracking
        'Signal',                   # Target label
        'Z-Score',                  # Feature
        'RSI',                      # Feature
        'Volume_Spike',             # Feature
        'EMA_26',                   # Feature
        'DMA',                      # Feature
        'MACD',                     # Feature
        'EMA_12',                   # Feature
        'Rolling_Mean',             # Feature
        'Rolling_Std',
        "Close",
        'Volume',
        'Avg_Volume'
    ]

    # Keep only the required columns
    data = data[required_columns]

    # StandardScaler for select features
    standardize_features = ['Close', 'Volume', 'Rolling_Mean', 'Rolling_Std',
                           'Avg_Volume', 'Volume_Spike', 'EMA_12', 'EMA_26',
                           'MACD', 'DMA']
    scaler_standard = StandardScaler()
    data[standardize_features] = scaler_standard.fit_transform(data[standardize_features])

    # MinMaxScaler for others
    scale_features = ['Z-Score', 'RSI']
    scaler_minmax = MinMaxScaler()
    data[scale_features] = scaler_minmax.fit_transform(data[scale_features])

    # Restore the Date and Ticker columns
    data[['Date', 'Ticker']] = date_ticker


    return data

# Execute the pipeline
sp500_tickers = get_sp500_tickers()
download_and_save_data(sp500_tickers, data_folder, start_date, end_date)

processed_data = combine_and_process_data(data_folder)
processed_data = scale_and_standardize_data(processed_data)

# Save the processed data
processed_data_path = f"6m_processed_data_{date_range_name}.csv"
processed_data.to_csv(processed_data_path, index=False)
print(f"Processed data saved to: {processed_data_path}")

# Show preview
print(processed_data.head())
print(f"Final processed data shape: {processed_data.shape}")

def split_data_for_backtesting_and_ml(processed_data_path):
    """
    Splits the processed data into two files:
    1. tickers_dates.csv: Contains 'Date' and 'Ticker' columns for backtesting.
    2. ml_features.csv: Contains only the features for ML modeling.
    """

    # Read the processed data
    processed_data = pd.read_csv(processed_data_path)

    # Extract tickers and dates
    tickers_dates = processed_data[['Date', 'Ticker']]

    # Extract ML features
    ml_features = processed_data.drop(columns=['Date', 'Ticker'])

    # Save tickers and dates to a separate file
    date_range_name = processed_data_path.split('_')[-1].split('.')[0]  # Extract date range from file name
    tickers_dates_path = f"6m_tickers_dates_{date_range_name}.csv"
    tickers_dates.to_csv(tickers_dates_path, index=False)
    print(f"Tickers and dates saved to: {tickers_dates_path}")

    # Save ML features to a separate file
    ml_features_path = f"6m_ml_features_{date_range_name}.csv"
    ml_features.to_csv(ml_features_path, index=False)
    print(f"ML features saved to: {ml_features_path}")

# Assuming 'processed_data_path' is the path to your processed data file
processed_data_path = "6m_processed_data.csv"
split_data_for_backtesting_and_ml(processed_data_path)