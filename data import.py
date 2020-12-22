import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import os 
# from zipline.api import order, record, symbol # for algo trading
# import pyfolio as pf
import ta # technical analysis
import pandas_ta as pdt # pandas wrapper for technical analysis


### DATA SOURCE 1: KAGGLE ----------------------------
# import dataframe
df = pd.read_csv('/Users/tgraf/Google Drive/Uni SG/Master/Smart Data Analytics/00 Group Project/Repository/Single-Timeseries-Crypto-Bot/Data/400 - 1m - Trading Pairs (2013-2020)/btcusd.csv')
print(df.info())

# convert values from timestamp to date
df['time'] = pd.to_datetime(df['time'], unit='ms')
print(df['time'])

# this doesn't work well, we have multiple obs with the same timestamp but different price values

### DATA SOURCE 2: CRYPTODATADOWNLOAD.com ----------------------------
# import dataframe
df2 = pd.read_csv('/Users/tgraf/Google Drive/Uni SG/Master/Smart Data Analytics/00 Group Project/Repository/Single-Timeseries-Crypto-Bot/Data/Bitfinex_BTCUSD_minute.csv', 
    header=1)
print(df2.info())

# convert values from timestamp to date
df2['unix'] = pd.to_datetime(df2['unix'], unit = 'ms')
print(df2['unix'])

# we only have data until 15th of November, so less than a month


### DATA SOURCE 3: GITHUB  ----------------------------
#https://github.com/Zombie-3000/Bitfinex-historical-data

headers = ['time', 'open', 'close', 'high', 'low', 'volume']
df_13 = pd.read_csv('https://raw.githubusercontent.com/Zombie-3000/Bitfinex-historical-data/master/BTCUSD/Candles_1m/2013/merged.csv?raw=true', names = headers)
df_14 = pd.read_csv('https://raw.githubusercontent.com/Zombie-3000/Bitfinex-historical-data/master/BTCUSD/Candles_1m/2014/merged.csv?raw=true', names = headers)
df_15 = pd.read_csv('https://raw.githubusercontent.com/Zombie-3000/Bitfinex-historical-data/master/BTCUSD/Candles_1m/2015/merged.csv?raw=true', names = headers)
df_16 = pd.read_csv('https://raw.githubusercontent.com/Zombie-3000/Bitfinex-historical-data/master/BTCUSD/Candles_1m/2016/merged.csv?raw=true', names = headers)
df_17 = pd.read_csv('https://raw.githubusercontent.com/Zombie-3000/Bitfinex-historical-data/master/BTCUSD/Candles_1m/2017/merged.csv?raw=true', names = headers)
df_18 = pd.read_csv('https://raw.githubusercontent.com/Zombie-3000/Bitfinex-historical-data/master/BTCUSD/Candles_1m/2018/merged.csv?raw=true', names = headers)
df_19 = pd.read_csv('https://raw.githubusercontent.com/Zombie-3000/Bitfinex-historical-data/master/BTCUSD/Candles_1m/2019/merged.csv?raw=true', names = headers)

# Merge dataframes
data_frames = [df_13, df_14, df_15, df_16, df_17, df_18, df_19]
df_merged = pd.concat(data_frames)

#convert timestamp to datae
df_merged['time'] = pd.to_datetime(df_merged['time'], unit = 'ms')
print(df_merged)

# Write csv of merged files
# pd.DataFrame.to_csv(df_merged, 'Zoombie_merged.txt', sep=',', na_rep='.', index=False)

# we should have a df with 6'832'800 rows (13y*365d*24h*60m)but only have 2'630'217 rows

# plot the open prices against time
# this takes a while
"""
df_merged.plot(x ='time', y='open', kind = 'scatter')
plt.show()
"""

# reset index and sort values according to time
df_merged = df_merged.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')
df_merged = df_merged.rename(columns={"index": "old_index"})
df_merged = df_merged.sort_values(by=['time'], ascending = True, na_position = 'last')
df_merged

### TECHNICAL ANALYSIS OF DATA SOURCE 3 -------------------
from ta import add_all_ta_features
from ta.utils import dropna

# add all 84 technical features
# this takes a while
"""
df = add_all_ta_features(
    df_merged, open="open", high="high", 
    low="low", close="close", volume="volume", fillna=True)

df = df.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')
df = df.rename(columns={"index": "old_index"})
df = df.sort_values(by=['time'], ascending = True, na_position = 'last')
df.info()
pd.DataFrame.to_csv(df, 'df_with_ta.txt', sep=',', na_rep='.', index=False)
"""


i = 1
df_merged['returns'][i] = np.log(df_merged['close'][i+1]/df_merged['close'][i])

# calculate returns
df_merged['returns'] = 'NA'
for i in range(len(df_merged)):
    df_merged['returns'][i] = np.log(df_merged['close'][i+1]/df_merged['close'][i])