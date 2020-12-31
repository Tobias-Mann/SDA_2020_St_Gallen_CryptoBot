import matplotlib as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data Import --------------------------

df = pd.read_csv('Data/strategies.csv')
df1 = pd.read_csv('Data/Dec19.csv')
df['open'] = df1['open']
df['high'] = df1['high']
df['low'] = df1['low']
df['close'] = df1['close']
df['volume'] = df1['volume']
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
#ax1.margins(x=-0.4, y=--0.4)
# Plot the closing price
df['price'].plot(ax=ax1, color='r', lw=2.)

# Plot the short and long moving averages
df[['short_ma', 'long_ma']].plot(ax=ax1, lw=2.)

# Plot the buy signals
ax1.plot(df.loc[df.positions == 1.0].index,
         df.short_ma[df.positions == 1.0],
         '^',
         markersize=10,
         color='m')

# Plot the sell signals
ax1.plot(df.loc[df.positions == -1.0].index,
         df.short_ma[df.positions == -1.0],
         'v',
         markersize=10,
         color='k')

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
fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 15))
ax1.set_title('BTC Price and MACD')
ax1.plot(df['time'], df['price'], color = 'black')
ax1.set_ylabel('Price')
ax1.set_xlabel('Time')

ax2.plot(df['time'], df['signal'], color='green')
ax2.plot(df['time'], df['macd'], color='red')
ax2.set_ylabel('MACD')


# Plot the Buy Signals
plt.plot(df.loc[df.positions == 1.0].index,
         df.signal[df.positions == 1.0],
         '^',
         markersize=10,
         color='m')

# Plot the sell signals
plt.plot(df.loc[df.positions == -1.0].index,
         df.signal[df.positions == -1.0],
         'v',
         markersize=10,
         color='k')



# RSI -----------------------------------
startdate = df.iloc[0].time
today = df.iloc[-1].time

fig = plt.figure(facecolor='white')
axescolor = '#f6f6f6'  # the axes background color
ax1 = fig.add_axes(rect1, facecolor=axescolor)  # left, bottom, width, height
ax2 = fig.add_axes(rect2, facecolor=axescolor, sharex=ax1)
ax2t = ax2.twinx()
ax3 = fig.add_axes(rect3, facecolor=axescolor, sharex=ax1)

plt.rc('axes', grid=True)
plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)

textsize = 9
left, width = 0.1, 0.8
rect1 = [left, 0.7, width, 0.2]
rect2 = [left, 0.3, width, 0.4]
rect3 = [left, 0.1, width, 0.2]

# plot the RSI
fig = plt.figure(facecolor='white')
axescolor = '#f6f6f6'  # the axes background color

ax1 = fig.add_axes(rect1, facecolor=axescolor)  # left, bottom, width, height
prices = df['price']
rsi = df['rsi']
fillcolor = 'darkgoldenrod'

ax1.plot(df['time'], rsi, color=fillcolor)
ax1.axhline(70, color=fillcolor)
ax1.axhline(30, color=fillcolor)
ax1.fill_between(df['time'], rsi, 70, where=(rsi >= 70), facecolor=fillcolor, edgecolor=fillcolor)
ax1.fill_between(df['time'], rsi, 30, where=(rsi <= 30), facecolor=fillcolor, edgecolor=fillcolor)
ax1.text(0.6, 0.9, '>70 = overbought', va='top', transform=ax1.transAxes, fontsize=textsize)
ax1.text(0.6, 0.1, '<30 = oversold', transform=ax1.transAxes, fontsize=textsize)
ax1.set_ylim(0, 100)
ax1.set_yticks([30, 70])
ax1.text(0.025, 0.95, 'RSI (14)', va='top', transform=ax1.transAxes, fontsize=textsize)
ax1.set_title('RSI')

"""
# plot the price and volume data
import matplotlib.font_manager as font_manager

dx = df['price']
low = df['low']
high = df['high']

deltas = np.zeros_like(prices)
deltas[1:] = np.diff(prices)
up = deltas > 0
ax2.vlines(df['time'][up], low[up], high[up], color='black', label='_nolegend_')
ax2.vlines(df['time'][~up], low[~up], high[~up], color='black', label='_nolegend_')
ma20 = moving_average(prices, 20, type='simple')
ma200 = moving_average(prices, 200, type='simple')

linema20, = ax2.plot(df['time'], ma20, color='blue', lw=2, label='MA (20)')
linema200, = ax2.plot(df['time'], ma200, color='red', lw=2, label='MA (200)')

props = font_manager.FontProperties(size=10)
leg = ax2.legend(loc='center left', shadow=True, fancybox=True, prop=props)
leg.get_frame().set_alpha(0.5)

volume = (df.close*df.volume)/1e6  # dollar volume in millions
vmax = volume.max()
poly = ax2t.fill_between(df['time'], volume, 0, label='Volume', facecolor=fillcolor, edgecolor=fillcolor)
ax2t.set_ylim(0, 5*vmax)
ax2t.set_yticks([])

plt.show()
"""



# SENTDEX MACD--------------------
ax2 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, facecolor='#07000d') # you could ass shareax = ax1
fillcolor = '#00ffe8'
ax2.plot(df['time'], df['signal'], color='#4ee6fd', lw=2)
ax2.plot(df['time'], df['macd'], color='#e1edf9', lw=1)
ax2.fill_between(df['time'], df['macd']-df['signal'], 0, alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)
ax2.spines['bottom'].set_color("#5998ff")
ax2.spines['top'].set_color("#5998ff")
ax2.spines['left'].set_color("#5998ff")
ax2.spines['right'].set_color("#5998ff")
ax2.tick_params(axis='x', colors='w')
ax2.tick_params(axis='y', colors='w')
plt.ylabel('MACD', color='w')
plt.show()
