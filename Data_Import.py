import pandas as pd
# from ta import add_all_ta_features
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np

"""
### DATA SOURCE 1: KAGGLE ------------------------------
# import dataframe
df_merged = pd.read_csv('./Data/400 - 1m - Trading Pairs (2013-2020)/btcusd.csv')
print(df_merged.info())

# convert values from timestamp to date
df_merged['time'] = pd.to_datetime(df_merged['time'], unit='ms')
print(df_merged['time'])

# this doesn't work well, we have multiple obs with the same timestamp but different price values

### DATA SOURCE 2: CRYPTODATADOWNLOAD.com ------------------------------
# import dataframe
df2 = pd.read_csv('./Data/Bitfinex_BTCUSD_minute.csv', 
    header=1)
print(df2.info())

# convert values from timestamp to date
df2['unix'] = pd.to_datetime(df2['unix'], unit = 'ms')
print(df2['unix'])

# we only have data until 15th of November, so less than a month
"""

### DATA SOURCE 3: GITHUB  ------------------------------
#https://github.com/Zombie-3000/Bitfinex-historical-data

headers = ['Time', 'Open', 'Close', 'High', 'Low', 'Volume']
df_13 = pd.read_csv('https://raw.githubusercontent.com/Zombie-3000/Bitfinex-historical-data/master/BTCUSD/Candles_1m/2013/merged.csv?raw=true', names = headers)
df_14 = pd.read_csv('https://raw.githubusercontent.com/Zombie-3000/Bitfinex-historical-data/master/BTCUSD/Candles_1m/2014/merged.csv?raw=true', names = headers)
df_15 = pd.read_csv('https://raw.githubusercontent.com/Zombie-3000/Bitfinex-historical-data/master/BTCUSD/Candles_1m/2015/merged.csv?raw=true', names = headers)
df_16 = pd.read_csv('https://raw.githubusercontent.com/Zombie-3000/Bitfinex-historical-data/master/BTCUSD/Candles_1m/2016/merged.csv?raw=true', names = headers)
df_17 = pd.read_csv('https://raw.githubusercontent.com/Zombie-3000/Bitfinex-historical-data/master/BTCUSD/Candles_1m/2017/merged.csv?raw=true', names = headers)
df_18 = pd.read_csv('https://raw.githubusercontent.com/Zombie-3000/Bitfinex-historical-data/master/BTCUSD/Candles_1m/2018/merged.csv?raw=true', names = headers)
df_19 = pd.read_csv('https://raw.githubusercontent.com/Zombie-3000/Bitfinex-historical-data/master/BTCUSD/Candles_1m/2019/merged.csv?raw=true', names = headers)

# check length and compare to theoretical value
#print('Theoretical maximum obervations:', 24*365*7)
#print('Actual observations of df:' len())

# Merge dataframes
data_frames = [df_13, df_14, df_15, df_16, df_17, df_18, df_19]
df_merged = pd.concat(data_frames)

#convert timestamp to datae
df_merged['Time'] = pd.to_datetime(df_merged['Time'], unit = 'ms')

# reset index and sort values according to Time
df_merged = df_merged.sort_values(by=['Time'], ascending = True, na_position = 'last')
df_merged = df_merged.reset_index(level = None, drop = True, inplace = False, col_level = 0, col_fill='')
df_merged.rename(columns={'Time': 'time', 'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}, inplace = True)

#print output
print(df_merged.info())

# save the new file under the folder Data
df_merged.to_csv('Data/BTC_USD/df_raw.csv', sep = ',', na_rep = '.', index = False)

# create and save subsets of Decembers
def make_subset (df, start_window, end_window, name):
    folder = './Data/BTC_USD/'
    if os.path.isdir(folder):
        pass
    else:
        os.mkdir(folder)
    mask = df['time'].between(start_window, end_window)
    df = df_merged[mask]
    df.reset_index(drop = True, inplace = True)
    name_temp = name + '.csv'
    df.to_csv(folder + name_temp)
    return df

# create subsets
Nov17 = make_subset(df_merged, '2017-11-01 00:00:00', '2017-11-30 23:59:00', 'Nov17')
Dec17 = make_subset(df_merged, '2017-12-01 00:00:00', '2017-12-31 23:59:00', 'Dec17')
Nov18 = make_subset(df_merged, '2018-11-01 00:00:00', '2018-11-30 23:59:00', 'Nov18')
Dec18 = make_subset(df_merged, '2018-12-01 00:00:00', '2018-12-31 23:59:00', 'Dec18')
Nov19 = make_subset(df_merged, '2019-11-01 00:00:00', '2019-11-30 23:59:00', 'Nov19')
Dec19 = make_subset(df_merged, '2019-12-01 00:00:00', '2019-12-31 23:59:00', 'Dec19')

# plot the full timeframe
figure = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
ax1 = figure.add_subplot(111, ylabel='Price in USD')
ax1.plot(df_merged.time, df_merged.close)
ax1.plot(df_merged.time, df_merged.close)
ax1.set_title('BTC PRICE OVER ALL DATA')

# plot the price
figure = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
ax1 = figure.add_subplot(111, ylabel='Price in USD')
ax1.plot(Nov17.index,
         np.log(1 + Nov17['close'].pct_change()).cumsum(),
         label='Nov17')
ax1.plot(Dec17.index,
         np.log(1 + Dec17['close'].pct_change()).cumsum(),
         label='Dec17')
ax1.plot(Nov18.index,
         np.log(1 + Nov18['close'].pct_change()).cumsum(),
         label='Nov18')
ax1.plot(Dec18.index,
         np.log(1 + Dec18['close'].pct_change()).cumsum(),
         label='Dec18')
ax1.plot(Nov19.index,
         np.log(1 + Nov19['close'].pct_change()).cumsum(),
         label='Nov19')
ax1.plot(Dec19.index,
         np.log(1 + Dec19['close'].pct_change()).cumsum(),
         label='Dec19')
ax1.set_title('BTC PRICE OVER DIFFERENT DECEMBER PERIODS')
plt.legend()
