import matplotlib as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.font_manager as font_manager


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
df['price'].plot(ax=ax1, color='black', lw=2.)

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
ax1.plot(df.index, df['price'], color = 'black')
ax1.set_ylabel('Price')

# Lower Subplot
ax2.plot(df.index, df['signal'], color='orange', label = 'signal')
ax2.plot(df.index, df['macd'], color='blue', label = 'macd')
ax2.set_ylabel('MACD')
ax2.set_xlabel('Time')

# PRICE PLOT ---
# Plot the Buy Signals
ax1.plot(df.loc[df.positions == 1.0].index,
         df.price[df.positions == 1.0],
         '^',
         markersize=10,
         color='g')

# Plot the sell signals
ax1.plot(df.loc[df.positions == -1.0].index,
         df.price[df.positions == -1.0],
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
import matplotlib.dates as mdates

# Configure variables
FIGSIZE = (15, 15)
FILLCOLOR = 'darkgoldenrod'

# Define variables
textsize = 16
prices = df['price']
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
ax1.plot(df.index, df['price'], color='black')
ax1.set_ylabel('Price')

# Plot the MA
ma1 = moving_average(prices, 20, type='simple')
ma2 = moving_average(prices, 200, type='simple')
linema20, = ax1.plot(df.index, ma1, color='blue', lw=2, label='MA (20)')
linema200, = ax1.plot(df.index, ma2, color='red', lw=2, label='MA (200)')

# Plot the volume
volume = (df.price * df.volume) / 1e6  # dollar volume in millions
vmax = volume.max()
poly = ax2t.fill_between(df.index, volume, label='Volume', facecolor=FILLCOLOR, edgecolor=FILLCOLOR)
ax2t.set_ylim(0, 5 * vmax)
ax2t.set_yticks([]) # sets secondary axis = 0
# ax2t.set_ylabel("Volume")

# plot the rsi
ax2.plot(df.index, rsi, color=FILLCOLOR)
ax2.axhline(70, color=FILLCOLOR)
ax2.axhline(30, color=FILLCOLOR)
ax2.fill_between(df.index, rsi, 70, where=(rsi >= 70), facecolor=FILLCOLOR, edgecolor=FILLCOLOR)
ax2.fill_between(df.index, rsi, 30, where=(rsi <= 30), facecolor=FILLCOLOR, edgecolor=FILLCOLOR)
ax2.text(0.8, 0.9,'>70 = overbought', va='top', transform=ax2.transAxes, fontsize=textsize-2)
ax2.text(0.8, 0.1,'<30 = oversold', transform=ax2.transAxes, fontsize=textsize-2)
ax2.set_ylim(0, 100)
ax2.set_yticks([30, 70])
ax2.text(0.025, 0.95, 'RSI', va='top',transform=ax2.transAxes,fontsize=textsize)

# add legend
ax1.legend()

# show plot
plt.show()
