import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np

### FETCH DATA FROM GITHUB  ------------------------------
#https://github.com/Zombie-3000/Bitfinex-historical-data

headers = ['Time', 'Open', 'Close', 'High', 'Low', 'Volume']
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

# check length and compare to theoretical value
#print('Theoretical maximum obervations:', 60*24*365*7)
#print('Actual observations of df:' len(df_merged))

#convert timestamp to datae
df_merged['Time'] = pd.to_datetime(df_merged['Time'], unit = 'ms')

# reset index and sort values according to Time
df_merged = df_merged.sort_values(by=['Time'], ascending = True, na_position = 'last')
df_merged = df_merged.reset_index(level = None, drop = True, inplace = False, col_level = 0, col_fill='')
df_merged.rename(columns={'Time': 'time', 'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}, inplace = True)

#print output
print(df_merged.info())

# save the new file under the folder Data
df_merged.to_csv('./Data/df_raw.csv', sep = ',', na_rep = '.', index = False)

### MAKE SUBSETS OF DATA  ------------------------------
# create and save subsets of Decembers
def make_subset (df, start_window, end_window, name):
    folder = './Data/'
    if os.path.isdir(folder):
        pass
    else:
        os.mkdir(folder)
    mask = df['time'].between(start_window, end_window)
    df = df_merged[mask]
    df.reset_index(drop = True, inplace = True)
    name_temp = name + '.csv'
    df.to_csv(folder + name_temp, index = False)
    return df

# create subsets
Nov17 = make_subset(df_merged, '2017-11-01 00:00:00', '2017-11-30 23:59:00', 'Nov17')
Nov18 = make_subset(df_merged, '2018-11-01 00:00:00', '2018-11-30 23:59:00', 'Nov18')
Dec19 = make_subset(df_merged, '2019-12-01 00:00:00', '2019-12-31 23:59:00', 'Dec19')


### MAKE PLOTS FROM THE DATA  ------------------------------
PATHNAME = './SDA_2020_St_Gallen_01_DataImport/'

# plot the full timeframe
fig = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(111, ylabel='BTC Price in USD')
ax1.plot(df_merged.time, df_merged.close)
ax1.plot(df_merged.time, df_merged.close)
ax1.set_title('BTC PRICE OVER ALL DATA')
plt.show()
fig.savefig(PATHNAME + 'FULL_BTC_USD.png', dpi=1200)

# plot the different timeframes
fig = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(111, ylabel='Cummulative Returns', xlabel = 'Observation Steps')
ax1.plot(Nov17.index,
         np.log(1 + Nov17['close'].pct_change()).cumsum(),
         label='Nov17')
ax1.plot(Nov18.index,
         np.log(1 + Nov18['close'].pct_change()).cumsum(),
         label='Nov18')
ax1.plot(Dec19.index,
         np.log(1 + Dec19['close'].pct_change()).cumsum(),
         label='Dec19')
ax1.set_title('BTC RETURNS OVER DIFFERENT PERIODS')
plt.legend()
plt.show()
fig.savefig(PATHNAME +'3PERIODS_BTC_USD.png', dpi=1200)

# plot every year
for i in (data_frames):
    i['Time'] = pd.to_datetime(i['Time'], unit = 'ms')
    fig = plt.figure(num=None,
                    figsize=(10, 5),
                    dpi=80,
                    facecolor='w',
                    edgecolor='k')
    ax1 = fig.add_subplot(111, ylabel='BTC Price in USD')
    ax1.yaxis.grid(color='gray', linestyle='dashed')
    TIME = str(i['Time'][0])[:-15]
    ax1.set_title('Timeframe: ' + TIME)
    ax1.plot(i['Time'], i['Close'])
    plt.show()
    fig.savefig(PATHNAME + TIME + '_BTC_USD.png', dpi=1200)
