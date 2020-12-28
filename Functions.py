import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


def Simple_MA(df, short_window, long_window):
    # make all columns lowercase
    df.columns = df.columns.str.lower()

    # Initialize the `signals` DataFrame with the `signal` column
    signals = pd.DataFrame(index = df.index)
    signals['position'] = 0.0
    signals['datetime'] = df['datetime']

    # Create short simple moving average over the short window
    signals['short_mavg'] = df['close'].rolling(window = short_window,
                                                min_periods = 1,
                                                center = False).mean()

    # Create long simple moving average over the long window
    signals['long_mavg'] = df['close'].rolling(window = long_window,
                                            min_periods = 1,
                                            center = False).mean()

    # Create signals, 1.0 = Signal, 0.0 = No signal
    signals['position'][short_window:] = np.where(
        signals['short_mavg'][short_window:] >
        signals['long_mavg'][short_window:], 1.0, 0.0)

    # Generate trading orders
    signals['action'] = signals['position'].diff()
    signals.loc[signals['action'] == 1]
    signals.loc[signals['action'] == 0]
    signals.loc[signals['action'] == -1]
    # 0 = do nothing, 1 = Buy, 2 = Sell

    # Initialize the plot figure
    fig = plt.figure(num=None,
                    figsize=(10, 10),
                    dpi=80,
                    facecolor='w',
                    edgecolor='k')

    # Add a subplot and label for y-axis
    ax1 = fig.add_subplot(111, ylabel='Price in $')
    # Plot the closing price
    df['close'].plot(ax=ax1, color='r', lw=2.)

    # Plot the short and long moving averages
    signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

    # Plot the buy signals
    ax1.plot(signals.loc[signals.action == 1.0].index,
             signals.short_mavg[signals.action == 1.0],
             '^',
             markersize=10,
             color='m')

    # Plot the sell signals
    ax1.plot(signals.loc[signals.action == -1.0].index,
             signals.short_mavg[signals.action == -1.0],
             'v',
             markersize=10,
             color='k')

    # Show the plot
    name = ('MA LONG:', long_window, '| MA SHORT:', short_window)
    plt.title(name)
    plt.show()

    return signals


def MACD(df, ema_short, ema_long, signal):

    #ema_long = 26
    #ema_short = 12
    #signal = 9

    ema_name_short = 'ema' + str(ema_short)
    ema_name_long = 'ema' + str(ema_long)

    signals = pd.DataFrame(index = df.index)
    signals['position'] = 0.0

    signals['datetime'] = df['datetime']

    # MACD
    signals[ema_name_short] = df['close'].ewm(span = ema_short).mean()
    signals[ema_name_long] = df['close'].ewm(span = ema_long).mean()
    signals['macd'] = signals[ema_name_short] - signals[ema_name_long]

    # Signal Line
    signals['signal'] = signals['macd'].ewm(span = signal).mean()

    # Crossovers, 1.0 = Signal or True, 0.0 = No signal or False
    #    signals['position'][ema_long:] = np.where(
    #       signals['macd'][ema_long:] > signals['signal'][ema_long:], 1.0, 0.0)

    # Crossovers, 1.0 = Signal or True, 0.0 = No signal or False
    signals.loc[:,'position'][ema_long:] = np.where(
        signals.loc[:, 'macd'][ema_long:] > signals.loc[:, 'signal'][ema_long:], 1.0, 0.0)

    # Generate trading orders from Crossovers
    signals['action'] = signals['position'].diff()
    #signals.loc[signals['action'] == 1]  # buy
    #signals.loc[signals['action'] == 0]  # hold
    #signals.loc[signals['action'] == -1]  # sell

    # Set up the figures
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize = (10,10))
    gs = fig.add_gridspec(ncols = 1, nrows = 2 , hspace=0)
    ax1.set_title("BTC PRICE")
    ax2.set_title("MACD")

    # plot the btc price & macd and signal line
    ax1.plot(df['datetime'], df['close'])
    ax2.plot(signals['datetime'], signals['macd'])
    ax2.plot(signals['datetime'], signals['signal'])

    # Plot the buy signals
    ax2.plot(signals.datetime[signals.loc[signals.action == 1.0].index],
                signals.signal[signals.action == 1.0],
                '^',
                markersize=10,
                color='g')

    # Plot the sell signals
    ax2.plot(signals.datetime[signals.loc[signals.action == -1.0].index],
             signals.signal[signals.action == -1.0],
             'v',
             markersize=10,
             color='r')

    return signals



def calc_portfolio(df, df_signals, strategy_name):

    df = hourly
    df_signals = signals_macd
    strategy_name = 'macd'

    initial_capital = float('10000')
    positions = pd.DataFrame(index = df.index).fillna(0.0)

    # Note how many btc you buy and add it to positions
    positions['btc'] = 1 * df_signals['action']

    # Initialize the portfolio with value owned
    portfolio = positions.multiply(df['close'], axis=0)  # axis = 0 means rows

    # Store the difference in shares owned
    pos_diff = positions.diff()

    # Add `holdings` to portfolio
    portfolio['holdings'] = (positions.multiply(df['close'],
                                                axis=0)).sum(axis=1)

    # Add `cash` to portfolio, we have a minus because a buy will get us less money
    portfolio['cash'] = initial_capital - (pos_diff.multiply(
        df['close'], axis=0)).sum(axis=1).cumsum()

    # Add `total` and 'returns' to portfolio
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()

    portfolio['datetime'] = df['datetime']

    print('Beginning PF Value: ', portfolio['total'][0])
    print('Ending PF Value: ', portfolio['total'].iloc[-1])

    # calc the holdings with strategy and with simply buy and hodl
    pct_change = df['close'].pct_change()
    cum_return = pd.DataFrame((1 + pct_change).cumprod())
    cum_return *= initial_capital

    # Set up the figures
    fig, ax1 = plt.subplots(1, sharex=True, figsize=(10, 10))
    gs = fig.add_gridspec(ncols=1, nrows=2, hspace=0)
    title = str(strategy_name) + ": HOLDINGS WITH STRATEGY vs. BUY AND HODL"
    ax1.set_title(title)

    ax1.plot(portfolio.datetime, cum_return['close'], color = 'black', label = 'BUY AND HODL')
    ax1.plot(portfolio.datetime, portfolio['total'], color = 'green', label = strategy_name)
    plt.legend(loc='upper left')

    # Create a tear sheet
    rows = ['Start Date', 'End Date', 'Total Periods', 'CAGR', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Avg Period Holdings']
    cols = [strategy_name]
    start_date = portfolio['datetime'][0]
    end_date = portfolio['datetime'].iloc[-1]
    total_periods = len(portfolio)
    cagr = (portfolio['total'].iloc[-1]/portfolio['total'][0])**(1/total_periods)-1
    volatility = portfolio['total'].std()
    sharpe_ratio = cagr / volatility
    max_drawdown = (portfolio['total'].min() - portfolio['total'].max()) / portfolio['total'].max()

    values = [start_date, end_date, total_periods, cagr, volatility, sharpe_ratio, max_drawdown]
    
    return portfolio, values

# IMPORT DATA FRAMES -------------------------------
df = pd.read_csv('Data/Dec19.csv')

df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', drop=True, inplace=True)
hourly = df.resample('60T').apply(lambda x: x[-1])
hourly.reset_index(inplace = True)
hourly['datetime'] = pd.to_datetime(hourly['datetime'])


# RUN THE FUNCTIONS ----------------------------------------

signals_simplema = Simple_MA(hourly, 12, 26)
signals_macd = MACD(hourly, 12, 26, 9)

pf1[200:210]
pf1, values1 = calc_portfolio(hourly, signals_simplema, 'SIMPLE MOVING AVERAGE')
pf2, values2 = calc_portfolio(hourly, signals_macd, 'MACD')