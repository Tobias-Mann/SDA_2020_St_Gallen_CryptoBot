import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.pylab import date2num
from mplfinance.original_flavor import candlestick_ohlc


# Data Import --------------------------

df = pd.read_csv('Data/strategies.csv')
df1 = pd.read_csv('Data/Dec19.csv')
df['open'] = df1['open']
df['high'] = df1['high']
df['low'] = df1['low']
df['close'] = df1['close']
df['volume'] = df1['volume']
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True, drop=True)
cols = ['open', 'high', 'low', 'close', 'volume', 'macd', 'signal', 'short_ma', 'long_ma',
    'z_value', 'rsi']
df = df[cols]
df = df.head(1000)


# Functions --------------------

def moving_average(x, n, type='simple'):
    """
    compute an n period moving average.

    type is 'simple' | 'exponential'

    """
    x = np.asarray(x)
    if type == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a

# Simple Moving Averages--------------------------

short_window = 12
long_window = 26
df['signal_point'] = 0.0

# Create signals, 1.0 = Signal, 0.0 = No signal
df['signal_point'][short_window:] = np.where(
    df['short_ma'][short_window:] > df['long_ma'][short_window:], 1.0, 0.0)

# Generate trading orders
df['positions'] = df['signal_point'].diff()
df.loc[df['positions'] == 1]
df.loc[df['positions'] == 0]
df.loc[df['positions'] == -1]
# 0 = do nothing, 1 = Buy, 2 = Sell

# Initialize the plot figure
fig = plt.figure(num=None,
                 figsize=(10, 10),
                 dpi=80,
                 facecolor='w',
                 edgecolor='k')
# Add a subplot and label for y-axis
ax1 = fig.add_subplot(111, ylabel='Price in $')
ax1.set_title('Simple MA')

#ax1.margins(x=-0.4, y=--0.4)
# Plot the closing price
df['close'].plot(ax=ax1, color='black', lw=2.)

# Plot the short and long moving averages
df[['short_ma', 'long_ma']].plot(ax=ax1, lw=2.)

# Plot the buy signals
ax1.plot(df.loc[df.positions == 1.0].index,
         df.short_ma[df.positions == 1.0],
         '^',
         markersize=10,
         color='g')

# Plot the sell signals
ax1.plot(df.loc[df.positions == -1.0].index,
         df.short_ma[df.positions == -1.0],
         'v',
         markersize=10,
         color='r')

# Show the plot
plt.show()

# MACD --------------------------

fast = 12
slow = 26
signal = 9

df['signal_point'] = 0.0

# Create signals, 1.0 = Signal, 0.0 = No signal
df['signal_point'][slow:] = np.where(
    df['signal'][slow:] > df['macd'][slow:], 1.0, 0.0)

# Generate trading orders
df['positions'] = df['signal_point'].diff()
df.loc[df['positions'] == 1]
df.loc[df['positions'] == 0]
df.loc[df['positions'] == -1]
# 0 = do nothing, 1 = Buy, 2 = Sell

# Upper Subplot
fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 15), sharex = True)
plt.subplots_adjust(wspace=0, hspace=0)
ax1.set_title('BTC Price and MACD')
ax1.plot(df.index, df['close'], color = 'black')
ax1.set_ylabel('close')

# Lower Subplot
ax2.plot(df.index, df['signal'], color='orange', label = 'signal')
ax2.plot(df.index, df['macd'], color='blue', label = 'macd')
ax2.set_ylabel('MACD')
ax2.set_xlabel('Time')

# PRICE PLOT ---
# Plot the Buy Signals
ax1.plot(df.loc[df.positions == 1.0].index,
         df.close[df.positions == 1.0],
         '^',
         markersize=10,
         color='g')

# Plot the sell signals
ax1.plot(df.loc[df.positions == -1.0].index,
         df.close[df.positions == -1.0],
         'v',
         markersize=10,
         color='r')

# MACD PLOT ---
# Plot the Buy Signals
ax2.plot(df.loc[df.positions == 1.0].index,
         df.signal[df.positions == 1.0],
         '^',
         markersize=10,
         color='g')

# Plot the sell signals
ax2.plot(df.loc[df.positions == -1.0].index,
         df.signal[df.positions == -1.0],
         'v',
         markersize=10,
         color='r')

# add legend
plt.legend()



# RSI -----------------------------------

# Configure variables
FIGSIZE = (15, 15)
FILLCOLOR = 'darkgoldenrod'

# Define variables
textsize = 16
prices = df['close']
rsi = df['rsi']

# Set up plot
fig = plt.figure(figsize = FIGSIZE)
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax2t = ax1.twinx()  # Create a new Axes instance with an invisible x-axis and an independent y-axis positioned opposite to the original one (i.e. at right)
plt.subplots_adjust(wspace=0, hspace=0.01)
ax1.set_title('BTC PRICE, MA, RSI')
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))


# Plot the Price
ax1.plot(df.index, df['close'], color='black')
ax1.set_ylabel('Price')

# Plot the MA
ma1 = moving_average(prices, 20, type='simple')
ma2 = moving_average(prices, 200, type='simple')
linema20, = ax1.plot(df.index, ma1, color='blue', lw=2, label='MA (20)')
linema200, = ax1.plot(df.index, ma2, color='red', lw=2, label='MA (200)')

# Plot the volume
volume = (df.close * df.volume) / 1e6  # dollar volume in millions
vmax = volume.max()
ax2t.fill_between(df.index, volume, label='Volume', facecolor=FILLCOLOR, edgecolor=FILLCOLOR)
ax2t.set_ylim(0, 5 * vmax)
ax2t.set_yticks([]) # sets secondary axis = 0
# ax2t.set_ylabel("Volume")

# plot the rsi
ax2.plot(df.index, rsi, color='orange')
ax2.axhline(70, color='orange')
ax2.axhline(30, color='orange')
ax2.text(0.8, 0.9,'>70 = overbought', va='top', transform=ax2.transAxes, fontsize=textsize-2)
ax2.text(0.8, 0.1,'<30 = oversold', transform=ax2.transAxes, fontsize=textsize-2)
ax2.set_ylim(0, 100)
ax2.set_yticks([30, 70])
ax2.text(0.025, 0.95, 'RSI', va='top',transform=ax2.transAxes,fontsize=textsize)

# add legend
ax1.legend()

# show plot
plt.show()


# MEAN REVERSION -------------------------------------------

df['signal_point'] = 0.0
z_values = df['z_value']

"""
# Create signals, 1.0 = Signal, 0.0 = No signal
df['signal_point_buy'] = np.where((df['z_value'] < -2), 1.0, 0.0)
df['signal_point_sell'] = np.where((df['z_value'] > 2), 1.0, 0.0)

# Generate trading orders
df['positions_buy'] = df['signal_point_buy'].diff()
df['positions_sell'] = df['signal_point_sell'].diff()

df.loc[df['positions_buy'] == 1]
df.loc[df['positions_buy'] == 0]
df.loc[df['positions_buy'] == -1]
# 0 = do nothing, 1 = Buy, 2 = Sell
"""
# Upper Subplot
fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 15), sharex=True)
plt.subplots_adjust(wspace=0, hspace=0)
ax1.set_title('BTC Price and Z-Value')
ax1.plot(df.index, df['close'], color='black')
ax1.set_ylabel('Price')

# Lower Subplot
ax2.plot(df.index, z_values, color='orange', label='Z-Value')
ax2.axhline(2, color=FILLCOLOR)
ax2.axhline(-2, color=FILLCOLOR)
ax2.set_ylabel('Z-Value')
ax2.text(0.8, 0.9,'>2 = overvalued: SELL', va='top', transform=ax2.transAxes, fontsize=textsize-2)
ax2.text(0.8, 0.1,'<2 = undervalued: BUY', transform=ax2.transAxes, fontsize=textsize-2)

"""
# PRICE PLOT ---
# Plot the Buy Signals
ax1.plot(df.loc[df.positions_buy == 1.0].index,
         df.price[df.positions_buy == 1.0],
         '^',
         markersize=10,
         color='g')

# Plot the sell signals
ax1.plot(df.loc[df.positions_sell == -1.0].index,
         df.price[df.positions_sell == -1.0],
         'v',
         markersize=10,
         color='r')

# Z-Value PLOT ---
# Plot the Buy Signals
ax2.plot(df.loc[df.positions_buy == 1.0].index,
         df.z_value[df.positions_buy == 1.0],
         '^',
         markersize=10,
         color='g')

# Plot the sell signals
ax2.plot(df.loc[df.positions_sell == -1.0].index,
         df.z_value[df.positions_sell == -1.0],
         'v',
         markersize=10,
         color='r')
"""


# ALL TOGETHER PLOTS --------------------

# Create figure and set axes for subplots
fig = plt.figure(figsize=FIGSIZE)
gs = gridspec.GridSpec(nrows = 4, ncols =1, height_ratios=[2, 1, 1, 1])
ax_candle = plt.subplot(gs[0])
ax_candle.set_xticks(df.index)
ax_candle.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
ax_macd = plt.subplot(gs[1], sharex = ax_candle)
ax_rsi = plt.subplot(gs[2], sharex = ax_candle)
ax_vol = plt.subplot(gs[3], sharex = ax_candle)

# Format x-axis ticks as dates
# ax_candle.xaxis_date()

# Get nested list of date, open, high, low and close prices
ohlc = []
for date, row in df.iterrows():
    openp, highp, lowp, closep = row[:4]
    ohlc.append([date2num(date), openp, highp, lowp, closep])

# Plot candlestick chart
ax_candle.plot(df.index, df["short_ma"], label="MA 12")
ax_candle.plot(df.index, df["long_ma"], label="MA 26")
candlestick_ohlc(ax_candle, ohlc, colorup="g", colordown="r", width=0.001,)
ax_candle.set_title('Candlestick (OHLC)')
ax_candle.legend()

# Plot MACD
ax_macd.plot(df.index, df["macd"], label="macd")
#ax_macd.bar(df.index, df["macd_hist"] * 3, label="hist")
ax_macd.plot(df.index, df["signal"], label="signal")
ax_macd.legend()

# Plot RSI
# Above 70% = overbought, below 30% = oversold
ax_rsi.set_ylabel("(RSI %)")
ax_rsi.plot(df.index, [70] * len(df.index), label="overbought")
ax_rsi.plot(df.index, [30] * len(df.index), label="oversold")
ax_rsi.plot(df.index, df["rsi"], label="rsi")
ax_rsi.legend()

# Show volume in millions
volume = df['volume'] / 1e6
ax_vol.fill_between(df.index, volume, label='Volume in Mio.', facecolor=FILLCOLOR, edgecolor=FILLCOLOR)
ax_vol.set_ylabel("(Million)")
vmax = volume.max()
ax_vol.set_ylim(0, 1.2 * vmax)
ax_vol.legend()

plt.show()


# Save the chart as PNG
#fig.savefig("charts/" + ticker + ".png", bbox_inches="tight")