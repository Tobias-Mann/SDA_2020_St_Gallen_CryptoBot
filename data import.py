import pandas as pd
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from functools import reduce
import pyfolio as pf # for technial analysis
from zipline.api import order, record, symbol




### DATA SOURCE 1: KAGGLE ----------------------------
# import dataframe
df = pd.read_csv('/Users/tgraf/Google Drive/Uni SG/Master/Smart Data Analytics/00 Group Project/Repository/Single-Timeseries-Crypto-Bot/Data/400 - 1m - Trading Pairs (2013-2020)/btcusd.csv')
print(df.info())

# convert values from timestamp to date
df['time'] = pd.to_datetime(df['time'], unit='ms')
print(df['time'])
#this doesn't work well, we have multiple obs with the same timestamp but different price values

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

headers = ['time', 'open', 'close', 'highest', 'lowest', 'volume']
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

# Write csv of merged files
pd.DataFrame.to_csv(df_merged, 'Zoombie_merged.txt', sep=',', na_rep='.', index=False)

# we should have a df with 6'832'800 rows (13y*365d*24h*60m)but only have 2'630'217 rows 
