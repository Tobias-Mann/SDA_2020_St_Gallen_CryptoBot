import pandas as pd
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


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
# Please download the data manually as it is too large for the repo

df3 = pd.read_csv('/Users/tgraf/Google Drive/Uni SG/Master/Smart Data Analytics/00 Group Project/Data/Bitfinex-historical-data-master/BTCUSD/Candles_1m/2013')
