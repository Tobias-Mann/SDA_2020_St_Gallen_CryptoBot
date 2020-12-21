import pandas as pd

# import dataframe
df = pd.read_csv('/Users/tgraf/Google Drive/Uni SG/Master/Smart Data Analytics/00 Group Project/Repository/Single-Timeseries-Crypto-Bot/Data/400 - 1m - Trading Pairs (2013-2020)/btcusd.csv')
print(df.info())

# convert values from timestamp to date
print(df['time'])
pd.to_datetime(df['time'])


pd.Timestamp(df['time'])
df['time'][2]

'{:.20f}'.format(df['time'])