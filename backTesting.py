from IPython import get_ipython
from IPython.display import display
from google.colab import drive
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error
import os
from datetime import timedelta
import matplotlib.dates as mdates


# Step 1 Load Data
processed_data_path = '/content/drive/MyDrive/sp500_ml_features_20240601.csv'  # Updated path
tickers_dates_path = '/content/drive/MyDrive/sp500_tickers_dates_20240601.csv'  # Updated path

processed_data = pd.read_csv(processed_data_path)
tickers_dates = pd.read_csv(tickers_dates_path)


# Reset indices if they are not default (0, 1, 2, ...)
processed_data = processed_data.reset_index()
tickers_dates = tickers_dates.reset_index()

# Rename the index column in tickers_dates for clarity
tickers_dates = tickers_dates.rename(columns={'index': 'original_index'})

# Merge the two dataframes on index
merged_data = pd.merge(processed_data, tickers_dates, left_index=True, right_index=True, how='inner')


# Define the required columns for the ML features dataframe
required_columns = [
'Close', 'Volume', 'Rolling_Mean', 'Rolling_Std', 'Z-Score', 'Avg_Volume', 'Volume_Spike', 'EMA_12', 'EMA_26', 'MACD', 'DMA', 'RSI'
]

# Create the ML features dataframe
ml_features_df = merged_data[required_columns].copy()

# Create the dataframe with the remaining columns
remaining_columns = [col for col in merged_data.columns if col not in required_columns]
other_data_df = merged_data[remaining_columns].copy()

# Step 2: Load the Trained Model
model = xgb.Booster()
model.load_model('/content/drive/MyDrive/xgboost_model_2024.model')
print("Model loaded successfully!")


# Make predictions using the loaded model
predictions = model.predict(xgb.DMatrix(ml_features_df))

# Create a dataframe with the predictions
predictions_df = pd.DataFrame(predictions, columns=['Predicted_Signals'])

predictions_df.head()

# Merge the two dataframes on index
backtesting_results_df = pd.merge(other_data_df, predictions_df, left_index=True, right_index=True)

backtesting_results_df.head()


# Calculate MSE
mse = mean_squared_error(backtesting_results_df['Predicted_Signals'], backtesting_results_df['Signal'])

# Print the MSE value
print(f"Mean Squared Error (MSE): {mse}")

# Clip Confidence_Score to the valid range (-1, 1)
backtesting_results_df['Signal'] = np.clip(backtesting_results_df['Signal'], -0.99999, 0.99999)

# Clip Predicted_Confidence_Score to the valid range (-1, 1)
backtesting_results_df['Predicted_Signals'] = np.clip(backtesting_results_df['Predicted_Signals'], -0.99999, 0.99999)

# Now recalculate the returns:
backtesting_results_df['Next_Day_Returns'] = np.arctanh(backtesting_results_df['Signal']) / 10
backtesting_results_df['Predicted_Next_Day_Returns'] = np.arctanh(backtesting_results_df['Predicted_Signals']) / 10


# --- Path to your data folder ---
data_folder = '/content/drive/MyDrive/sp500_data_20240101_to_20240601'

# --- Create a list to store close prices ---
close_prices = []

# --- Iterate through each row in backtesting_results_df ---
for index, row in backtesting_results_df.iterrows():
    ticker = row['Ticker']
    date = row['Date']

    # --- Construct the file path for the current ticker ---
    file_path = os.path.join(data_folder, f"{ticker}.csv")

    # --- Check if the file exists ---
    if os.path.exists(file_path):
        # --- Read the CSV file for the current ticker ---
        ticker_data = pd.read_csv(file_path)

        # --- Find the close price for the specific date ---
        close_price = ticker_data.loc[ticker_data['Date'] == date, 'Close'].values

        # --- Append the close price to the list ---
        if len(close_price) > 0:
            close_prices.append(close_price[0])
        else:
            close_prices.append(np.nan)  # Append NaN if not found
    else:
        close_prices.append(np.nan)  # Append NaN if file not found

# --- Add the close prices to backtesting_results_df ---
backtesting_results_df['Close_Price'] = close_prices

# Specify the file path where you want to save the DataFrame
file_path = '/content/drive/MyDrive/backtesting_results.csv'  # Replace with your desired path

# Save the DataFrame to a CSV file
backtesting_results_df.to_csv(file_path, index=False)

import pandas as pd

# Specify the file path where you saved the DataFrame
file_path = '/content/drive/MyDrive/backtesting_results.csv'  # Replace with your actual path

# Load the DataFrame from the CSV file
backtesting_results_df = pd.read_csv(file_path)


# --- Data Preparation ---
# Filter data for the desired time frame (2/1/2024-5/30/2024)
start_date = '2024-02-01'
end_date = '2024-05-30'
filtered_data = backtesting_results_df[(backtesting_results_df['Date'] >= start_date) & (backtesting_results_df['Date'] <= end_date)]

# Get all unique dates in the filtered data and sort them
all_dates = pd.to_datetime(filtered_data['Date'].unique()).sort_values()

# --- Portfolio Simulation ---
portfolio_value = 0  # Initial portfolio value (starts at 0, all in cash)
cash_held = 10000  # Initial cash held
daily_values = [portfolio_value + cash_held]  # Total account value (portfolio + cash)
holdings = {}  # Initialize holdings dictionary
trading_log = []  # Initialize trading log

# Function to calculate Sharpe Ratio
def calculate_sharpe_ratio(daily_returns):
    return np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

for i, date in enumerate(all_dates):
    # Get data for the current trading day
    daily_data = filtered_data[filtered_data['Date'] == date.strftime('%Y-%m-%d')]

    # Get the previous trading day date
    if i > 0:  # Check if it's not the first day
        previous_date = all_dates[i - 1]
    else:
        # If it's the first day, there's no previous data to use for stock selection,
        # so we'll skip stock selection on the first day
        print("Skipping stock selection for the first day due to lack of previous day's data.")
        continue

    previous_daily_data = filtered_data[filtered_data['Date'] == previous_date.strftime('%Y-%m-%d')]

    # --- Portfolio Optimization ---
    # Select top N stocks based on predicted confidence score
    n_top_stocks = 10
    filtered_daily_data = daily_data[daily_data['Predicted_Signals'] >= 0.6]

    if filtered_daily_data.empty:
        print(f"Warning: No stocks meet the signal threshold on {date.strftime('%Y-%m-%d')}. Skipping stock selection for this day.")
        continue  # Skip to the next day
    sorted_stocks = filtered_daily_data.sort_values(by=['Predicted_Signals'], ascending=False)
    top_stocks = sorted_stocks['Ticker'].head(n_top_stocks).tolist()

    # Assume equal weights for selected stocks
    weights = np.array([1 / n_top_stocks] * n_top_stocks)

    # Print selected top stocks
    print(f"Date: {date.strftime('%Y-%m-%d')} Selected Top Stocks: {top_stocks}")

    # --- Update Portfolio Value, Cash Held, and Holdings ---
    # Liquidate existing holdings (sell all stocks using current day's closing price)
    for ticker, shares in holdings.copy().items():  # Iterate over a copy to avoid RuntimeError
        try:
            # Use current day's closing price for selling
            close_price = daily_data.loc[daily_data['Ticker'] == ticker, 'Close_Price'].values[0]
            cash_held += shares * close_price  # Add the value of sold stocks to cash held
            portfolio_value -= shares * close_price  # Subtract the value from portfolio value
            trading_log.append([date.strftime('%Y-%m-%d'), ticker, 'SELL', shares, close_price])  # Log the sell trade
        except IndexError:
            print(f"Warning: Close price not found for {ticker} on {date.strftime('%Y-%m-%d')} for liquidation. Assuming no change.")

    # Reset holdings for the current day after selling
    holdings = {}

    # Buy new stocks based on the top stocks selected (using previous day's closing price)
    for ticker in top_stocks:
        try:
            # Use previous day's closing price for buying
            close_price = previous_daily_data.loc[previous_daily_data['Ticker'] == ticker, 'Close_Price'].values[0]
            allocation = cash_held * weights[top_stocks.index(ticker)]  # Allocate capital from cash held
            shares = int(allocation /close_price)

            # Check if enough cash is available
            if cash_held >= shares * close_price:
                holdings[ticker] = shares
                cash_held -= shares * close_price  # Deduct the cost of bought stocks from cash held
                portfolio_value += shares * close_price  # Add the value to portfolio value

                # Log the trade
                trading_log.append([date.strftime('%Y-%m-%d'), ticker, 'BUY', shares, close_price])
            else:
                print(f"Warning: Insufficient cash to buy {ticker} on {date.strftime('%Y-%m-%d')}. Skipping this ticker.")

        except IndexError:
            print(f"Warning: Close price not found for {ticker} on {previous_date.strftime('%Y-%m-%d')} for buying. Skipping this ticker.")

            # Check if enough cash is available
            if cash_held >= shares * close_price:
                holdings[ticker] = shares
                cash_held -= shares * close_price  # Deduct the cost of bought stocks from cash held
                portfolio_value += shares * close_price  # Add the value to portfolio value

                # Log the trade
                trading_log.append([date.strftime('%Y-%m-%d'), ticker, 'BUY', shares, close_price])
            else:
                print(f"Warning: Insufficient cash to buy {ticker} on {date.strftime('%Y-%m-%d')}. Skipping this ticker.")

        except IndexError:
            print(f"Warning: Close price not found for {ticker} on {previous_date.strftime('%Y-%m-%d')} for buying. Skipping this ticker.")

    # --- Simulate daily returns and update portfolio value ---
    current_daily_return = 0
    for ticker, shares in holdings.items():
        try:
            close_price_today = daily_data.loc[daily_data['Ticker'] == ticker, 'Close_Price'].values[0]
            close_price_yesterday = previous_daily_data.loc[previous_daily_data['Ticker'] == ticker, 'Close_Price'].values[0]

            # Calculate return for this stock
            ticker_return = (close_price_today - close_price_yesterday) / close_price_yesterday

            # Update portfolio value based on stock return
            portfolio_value += shares * close_price_yesterday * ticker_return

            # Update current daily return (for Sharpe Ratio calculation)
            current_daily_return += ticker_return * (shares * close_price_yesterday / (portfolio_value + cash_held))

        except IndexError:
            print(f"Warning: Close price not found for {ticker} on {date.strftime('%Y-%m-%d')} or {previous_date.strftime('%Y-%m-%d')} for return calculation. Assuming no change.")

    daily_values.append(portfolio_value + cash_held)  # Append total account value to daily values list

# --- Performance Evaluation ---
# Calculate daily returns of the portfolio
daily_returns = np.diff(daily_values) / daily_values[:-1]

# Calculate Sharpe Ratio (assuming a risk-free rate of 0)
sharpe_ratio = calculate_sharpe_ratio(daily_returns)

# Calculate Maximum Drawdown
peak = daily_values[0]
drawdown = 0
max_drawdown = 0
for value in daily_values:
    if value > peak:
        peak = value
        drawdown = 0
    else:
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)

# Print performance metrics
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")

# Total profit
total_profit = daily_values[-1] - 10000  # Corrected calculation

# Percentage gain
percentage_gain = (total_profit / 10000) * 100  # Corrected calculation

# Print the results
print(f"Total Profit: ${total_profit:.2f}")
print(f"Percentage Gain: {percentage_gain:.2f}%")

# --- Trading Log ---
print("\nTrading Log:")
for trade in trading_log:
    print(trade)


daily_returns = np.diff(daily_values) / daily_values[:-1]  # Calculate daily returns from portfolio values
initial_portfolio_value = 10000

portfolio_values = [initial_portfolio_value]
for daily_return in daily_returns:
    portfolio_values.append(portfolio_values[-1] * (1 + daily_return))

# Create a DataFrame with correct alignment
# The index should be all_dates, not portfolio_values[1:]
# Use the entire portfolio_values list (including the initial value)
# The length of 'portfolio_values' and the index should be the same.
# We will create a date range that matches the number of portfolio values.
date_range = pd.date_range(start=all_dates.min(), periods=len(portfolio_values), freq='D')

portfolio_df = pd.DataFrame({'Portfolio_Value': portfolio_values}, index=date_range)


# 2. Plot the portfolio value over time
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(portfolio_df.index, portfolio_df['Portfolio_Value'])

# Format x-axis to show weekly intervals
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45, ha='right')

plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.tight_layout()
plt.show()

# 3. Calculate and print total profit and percentage gain
initial_investment = 10000  # Your initial investment
final_portfolio_value = portfolio_df['Portfolio_Value'].iloc[-1] # Get the last value

total_profit = final_portfolio_value - initial_investment
percentage_gain = (total_profit / initial_investment) * 100

print(f"Total Profit: ${total_profit:.2f}")
print(f"Percentage Gain: {percentage_gain:.2f}%")