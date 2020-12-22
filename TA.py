import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import os 
# from zipline.api import order, record, symbol # for algo trading
# import pyfolio as pf
import pandas_ta as pdt # pandas wrapper for technical analysis
from ta import add_all_ta_features
import backtrader as bt
from datetime import datetime

import Functions



### TECHNICAL ANALYSIS OF DATA SOURCE 3 ------------------------------

# import dataframe
df_raw = pd.read_csv('df_raw.csv')


# take a subset of Dec 19 of the data for faster computation
df_raw.loc[df_raw['time'] == '2019-12-01 00:00:00']
x = 2587233
y = len(df_raw)
df_subset = df_raw[x:y]

"""
# add all 84 technical features
# this takes a while
df_subset = add_all_ta_features(
    df_subset, open="open", high="high", 
    low="low", close="close", volume="volume", fillna=True)

df_subset = df_subset.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')
df_subset = df_subset.rename(columns={"index": "old_index"})
df_subset = df_subset.sort_values(by=['time'], ascending = True, na_position = 'last')
df_subset.info()
# pd.DataFrame.to_csv(df, 'df_with_ta.txt', sep=',', na_rep='.', index=False)

# calculate returns
# this takes a while
df_subset['returns'] = 'NA'
for i in range(len(df_subset)):
    df_subset['returns'][i] = np.log(df_subset['close'][i+1]/df_subset['close'][i])
print(df_subset)
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
import pandas as pd
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