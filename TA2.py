import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas_ta as pdt
from datetime import datetime


# import the functions from functions.py
import Functions

# import dataframe
df_raw = pd.read_csv('/Users/tgraf/Google Drive/Uni SG/Master/Smart Data Analytics/00 Group Project/df_raw.csv')

# Set the Subset here
index = df_raw[df_raw['Time'] == '2019-01-01 00:00:00'].index.values[0]

# Take the Subset of the Data
x = index
y = len(df_raw)
df_subset = df_raw[x:y]
print('Number of rows: {}, Number of columns: {}'.format(*df_subset.shape))


# convert the Dates from object to datetime
df_subset = df_subset.drop(df_subset.columns.difference(['Time', 'Open', 'Close', 'High', 'Low', 'Volume']), axis = 1)
df_subset['DateTime'] = pd.to_datetime(df_subset['Time'])
df_subset['Date'] = df_subset['DateTime'].dt.date
df_subset['Time'] = df_subset['DateTime'].dt.time
# set index
df_subset = df_subset.set_index('DateTime')

# plot the series
df_subset.plot(x='Date', y='Close', grid=True, figsize=(10, 10))
plt.title('BTC PRICE')
plt.show()

"""
## MINUTELY RETRUNS & HISTORGRAM
minutely_pct_c = df_subset['Close'].pct_change()
minutely_pct_c.fillna(0, inplace=True)
print('\n', minutely_pct_c)
# Minutely log returns
minutely_log_returns = np.log(df_subset['Close'].pct_change() + 1)
print('\n', minutely_log_returns)
"""

## MINUTELY RETRUNS & HISTORGRAM
minutely = df_subset
minutely.drop(['Time', 'Date'], axis=1, inplace=True)
minutely_pct_c = minutely.pct_change()
minutely_pct_c['Close'].hist(bins = 20)
plt.title('MINUTELY RETRUNS')
plt.show()

## HOURLY RETRUNS & HISTORGRAM
# Resample df_subset['Close'] to hours, take last observation as value
hourly = df_subset.resample('60T').apply(lambda x: x)
hourly_pct_c = hourly.pct_change()
hourly_pct_c['Close'].hist(bins=20)
plt.title('HOURLY RETRUNS')
plt.show()

## DAILY RETRUNS & HISTORGRAM
daily = df_subset.resample('D').apply(lambda x: x[-1])
daily_pct_c = daily.pct_change()
daily_pct_c['Close'].hist(bins=50)
plt.title('DAILY RETRUNS')
plt.show()

## MONTHLY RETRUNS & HISTORGRAM
monthly = df_subset.resample('M').apply(lambda x: x[-1])
monthly_pct_c = monthly.pct_change()
monthly_pct_c['Close'].hist(bins=50)
plt.title('Monthly RETRUNS')
plt.show()


# Pull up summary statistics
print('\n', 'Minutely: ', minutely_pct_c['Close'].describe())
print('\n', 'Hourly: ', hourly_pct_c['Close'].describe())
print('\n', 'Daily: ', daily_pct_c['Close'].describe())
print('\n', 'Monthly: ', monthly_pct_c['Close'].describe())


# CUMULATIVE DAILY RETURNS
cum_daily_return = (1 + daily_pct_c).cumprod()
print('\n', cum_daily_return)
cum_daily_return['Close'].plot(figsize=(12, 8))
plt.title('CUMULATIVE DAILY RETURNS')
plt.show()


# Calculate the moving average
df_subset['MA_40'] = df_subset['Close'].rolling(window = 40).mean()
df_subset['MA_252'] = df_subset['Close'].rolling(window = 252).mean()
df_subset[['Close', 'MA_40', 'MA_252']].plot(figsize = (10,10))
plt.title('MOVING AVERAGES')
plt.show()


# Calculate the volatility and define the minumum of periods to consider
min_periods = 10
vol = daily_pct_c['Close'].rolling(min_periods).std() * np.sqrt(min_periods)
vol.plot(figsize=(10, 8))
plt.title('DAILY VOLATILITY for min_periods', min_periods)
plt.show()

"""
### OLS -------------------------------------
# Import the `api` model of `statsmodels` under alias `sm`
import statsmodels.api as sm
from pandas import tseries

# Calculate the returns
returns = np.log(df_subset['Close'] / df_subset['Close'].shift(1))

# Add a constant
X = sm.add_constant(df_subset['Close'])

# Construct the model
model = sm.OLS(returns, X).fit()
print(model.summary()) # perfect multicollinarity
"""

### TRADING STRATEGY ----------------------------------

# choose which time sample to take (minutely, hourly, daily)
df = daily

# Initialize the short and long windows
short_window = 40
long_window = 100

# Initialize the `signals` DataFrame with the `signal` column
signals = pd.DataFrame(index=df.index)
signals['signal'] = 0.0

# Create short simple moving average over the short window
signals['short_mavg'] = df['Close'].rolling(window=short_window,
                                                   min_periods=1,
                                                   center=False).mean()

# Create long simple moving average over the long window
signals['long_mavg'] = df['Close'].rolling(window=long_window,
                                                  min_periods=1,
                                                  center=False).mean()

# Create signals, 1.0 = Signal, 0.0 = No signal
signals['signal'][short_window:] = np.where(
    signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:],
    1.0, 0.0)

# Generate trading orders
signals['positions'] = signals['signal'].diff()
signals.loc[signals['positions'] == 1]
signals.loc[signals['positions'] == 0]
signals.loc[signals['positions'] == -1]
# 0 = do nothing, 1 = Buy, 2 = Sell



# Initialize the plot figure
fig = plt.figure(num=None,
                 figsize=(10, 10),
                 dpi=80,
                 facecolor='w',
                 edgecolor='k')
# Add a subplot and label for y-axis
ax1 = fig.add_subplot(111, ylabel = 'Price in $')
# Plot the closing price
df['Close'].plot(ax = ax1, color = 'r', lw = 2.)

# Plot the short and long moving averages
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

# Plot the buy signals
ax1.plot(signals.loc[signals.positions == 1.0].index,
         signals.short_mavg[signals.positions == 1.0],
         '^',
         markersize=10,
         color='m')

# Plot the sell signals
ax1.plot(signals.loc[signals.positions == -1.0].index,
         signals.short_mavg[signals.positions == -1.0],
         'v',
         markersize=10,
         color='k')

# Show the plot
plt.show()


### BACKTESTING THE STRATEGY WITH PANDAS ------------------------------

# choose which time sample to take (minutely, hourly, daily)
df = daily

# Set the initial capital
initial_capital = float(10000.0)
# Create an empty DataFrame with the same index: `positions`
positions = pd.DataFrame(index=signals.index).fillna(0.0)
# Note how many btc you buy and add it to positions
positions['btc'] = 1 * signals['signal']
# Initialize the portfolio with value owned
portfolio = positions.multiply(df['Close'], axis=0) # axis = 0 means rows
# Store the difference in shares owned
pos_diff = positions.diff()
# Add `holdings` to portfolio
portfolio['holdings'] = (positions.multiply(df['Close'],
                                            axis=0)).sum(axis=1)
# Add `cash` to portfolio, we have a minus because a buy will get us less money
portfolio['cash'] = initial_capital - (pos_diff.multiply(
    df['Close'], axis=0)).sum(axis=1).cumsum()
# Add `total` to portfolio
portfolio['total'] = portfolio['cash'] + portfolio['holdings']
# Add `returns` to portfolio
portfolio['returns'] = portfolio['total'].pct_change()


## PLOT
fig = plt.figure(figsize = (10, 10))
ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')

# Plot the equity curve in dollars
portfolio['total'].plot(ax=ax1, lw=2.)

# Plot the "buy" trades against the equity curve
ax1.plot(portfolio.loc[signals.positions == 1.0].index,
         portfolio.total[signals.positions == 1.0],
         '^',
         markersize=10,
         color='m')

# Plot the "sell" trades against the equity curve
ax1.plot(portfolio.loc[signals.positions == -1.0].index,
         portfolio.total[signals.positions == -1.0],
         'v',
         markersize=10,
         color='k')

# Show the plot
plt.show()


### EVALUATE THE RESULTS --------------------------

# choose which time sample to take (minutely, hourly, daily)
df = daily


# SHARPE RATIO
# Isolate the returns of your strategy
returns = portfolio['returns']
# dailyized Sharpe ratio
sharpe_ratio = np.sqrt(60*24) * (returns.mean() / returns.std())
# Print the Sharpe ratio
print(sharpe_ratio)


# MAXIMUM DRAWDOWN
# Define a trailing 252 trading day window
window = 252
# Calculate the max drawdown in the past window days for each day
rolling_max = df['Close'].rolling(window, min_periods = 1).max()
hourly_drawdown = df['Close'] / rolling_max - 1.0
# Calculate the minimum (negative) daily drawdown
max_hourly_drawdown = hourly_drawdown.rolling(window, min_periods=1).min()
# Plot the results
hourly_drawdown.plot()
max_hourly_drawdown.plot()
plt.show()