import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import pandas_ta as pdt # pandas wrapper for technical analysis
import backtrader as bt
from datetime import datetime
from sklearn.ensemble import *
import xgboost as xgb
from statsmodels.tsa.stattools import adfuller


# import the functions from functions.py
import Functions


# import dataframe
df_raw = pd.read_csv('/Users/tgraf/Google Drive/Uni SG/Master/Smart Data Analytics/00 Group Project/df_raw.csv')

# take a subset of Dec 19 of the data for faster computation
index = df_raw[df_raw['Time']=='2019-12-31 00:00:00'].index.values[0]
x = index
y = len(df_raw)
df_subset = df_raw[x:y]
print('Number of rows: {}, Number of columns: {}'.format(*df_subset.shape))


### TECHNICAL ANALYSIS OF DATA SOURCE 3 ------------------------------

"""
# BOLLINGER BANDS -------------------------------------
# https://towardsdatascience.com/trading-technical-analysis-with-pandas-43e737a17861


# Periods
n = 20 

# calculate Simple Moving Average with 20 days window
sma = df_subset['close'].rolling(window=n).mean()

# calculate the standard deviation
rstd = df_subset['close'].rolling(window=n).std()

upper_band = pd.DataFrame(sma + 2 * rstd)
upper_band = upper_band.rename(columns={'close': 'upper'})
lower_band = pd.DataFrame(sma - 2 * rstd)
lower_band = lower_band.rename(columns={'close': 'lower'})

# plot the bollinger bands
df_bands = pd.merge(upper_band, lower_band,left_index=True, right_index=True, how='left')
df_bands = pd.merge(df_bands, df_subset['close'],left_index=True, right_index=True, how='left')
df_bands = pd.merge(df_bands, df_subset['time'],left_index=True, right_index=True, how='left')
ax = df_bands.plot(title='{} Price and BB'.format('btc'))
ax.fill_between(df_bands.index, lower_band['lower'], upper_band['upper'], color='#ADCCFF', alpha='0.4')
ax.set_xlabel('date')
ax.set_ylabel('SMA and BB')
ax.grid()
plt.show()


# BOLLINGER BANDS 2 -------------------------------------
from ta.volatility import *

# Initialize Bollinger Bands Indicator
indicator_bb = BollingerBands(close=df_subset["close"], window=20, window_dev=2)

# Add Bollinger Bands features
df_subset['bb_bbm'] = indicator_bb.bollinger_mavg()
df_subset['bb_bbh'] = indicator_bb.bollinger_hband()
df_subset['bb_bbl'] = indicator_bb.bollinger_lband()

# Add Bollinger Band high indicator
df_subset['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()

# Add Bollinger Band low indicator
df_subset['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

# plot
with pd.plotting.plot_params.use('x_compat', True):
    df_subset['close'].plot(color='r')
    df_subset['bb_bbm'].plot(color='g')
    df_subset['bb_bbh'].plot(color='b')
"""

# STATIONARITY Analysis ----------------------

# Historgram
df_subset['Close'].hist()
plt.show() 
# clearly we have a non-gaussian distribution

# Mean and Variance
X = df_subset['Close'].values
split = round(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
# running this tests shows different variances for different timeframes

# Augmented Dickey-Fuller Test
X = df_subset['Close'].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# Null Hypothesis (H0): If failed to be rejected, it suggests the time series 
# has a unit root, meaning it is non-stationary. It has some time dependent structure.


# XGBOOST ----------------------
df_subset.info()
df_subset = df_subset.drop(columns='old_index')

excl = ['Close', 'Time', 'Open', 'High', 'Volume', 'day', 'Week', 'Weekday', 'month', 'year']
cols = [c for c in df_subset.columns if c not in excl]

y_test = df_subset[round(0.9*len(df_subset)):len(df_subset)]['Close']
y_train = df_subset[0:round(0.9*len(df_subset))]
y_mean = np.mean(y_train)

xgb_params = {
    'n_trees': 800,
    'eta': 0.0045,
    'max_depth': 20,
    'subsample': 0.95,
    'colsample_bytree': 0.95,
    'colsample_bylevel': 0.95,
    'objective': 'multi:softmax',
    'num_class' : 3,
    'eval_metric': 'mlogloss', # 'merror', # 'rmse',
    'base_score': 0,
    'silent': 1
}