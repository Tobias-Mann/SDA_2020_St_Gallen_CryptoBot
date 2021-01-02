import numpy as np
import pandas as pd
import simulator


class buyandhold(simulator.decisionmaker):
    def __init__(self, environment):
        super(buyandhold, self).__init__(environment)
        self.memory = []
    
    def make_decision(self, row):
        closing_price = row[-1]
        if self.env.portfolio.usd>=closing_price and self.env.portfolio.btc == 0:
        # The BTC == 0 restriction ensures no additional BTC will be bought after the initial buying decision is made
            quantity = self.env.portfolio.usd//closing_price
            self.env.orderbook.new_marketorder(quantity)
    

class meanreversion(simulator.decisionmaker):
    def __init__(self, environment):
        super(meanreversion, self).__init__(environment)
        self.memory = []
        self.z_memory = []
        self.__critical_deviation__ = 2

    def change_critical_deviation(self, new):
        self.__critical_deviation__ = new

    def make_decision(self, row):
        closing_price = row[-1]
        self.memory.append(closing_price)
        n = 50
        if len(self.memory) >= n:
            values = np.array(self.memory[-n:])
            mean = np.mean(values)
            std = np.std(values)
            z = (closing_price - mean)/std
            self.z_memory.append(z)
            if z > self.__critical_deviation__ and self.env.portfolio.btc >0 :
                # sell at market
                quantity = self.env.portfolio.btc
                self.env.orderbook.new_marketorder(quantity, False)
            elif z < -self.__critical_deviation__ and self.env.portfolio.usd>=closing_price:
                # buy at market
                quantity = self.env.portfolio.usd//closing_price
                self.env.orderbook.new_marketorder(quantity)

class SimpleMA(simulator.decisionmaker):
    def __init__(self, environment):
        super(SimpleMA, self).__init__(environment)
        self.memory = []
        self.short_memory = []
        self.long_memory = []

    # function for calculating a moving average with numpy arrays
    def moving_average(self, array, periods):
        weights = np.ones(periods) / periods
        return (array * weights).sum()

    def make_decision(self, row):
        closing_price = row[-1]
        self.memory.append(closing_price)
        short_window = 12
        long_window = 26
        if len(self.memory) >= long_window:
            values = np.array(self.memory[-long_window:])

            # calculate the moving averages
            values_short = self.moving_average(values[-short_window:], short_window)
            values_long = self.moving_average(values, long_window)
            self.short_memory.append(values_short)
            self.long_memory.append(values_long)

            if values_short < values_long and self.env.portfolio.btc > 0:
                # sell at market
                quantity = self.env.portfolio.btc
                self.env.orderbook.new_marketorder(quantity, False)
            elif values_short > values_long and self.env.portfolio.usd >= closing_price > 0:
                # buy at market
                quantity = self.env.portfolio.usd // closing_price
                self.env.orderbook.new_marketorder(quantity)

class MACD(simulator.decisionmaker):
    def __init__(self, environment):
        super(MACD, self).__init__(environment)
        self.memory = []
        self.macd_memory = []
        self.signal_memory = []

    def ExpMovingAverage(self, values, window):
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        a = (weights * values).sum()
        return a

    def computeMACD(self, x, slow, fast, signal):
        # compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
        # return value is emaslow, emafast, macd which are len(x) arrays
        emaslow = self.ExpMovingAverage(x, slow)
        emafast = self.ExpMovingAverage(x[-fast:], fast)
        macd = emafast - emaslow
        return macd

    def make_decision(self, row):
        closing_price = row[-1]
        self.memory.append(closing_price)
        slow = 26
        fast = 12
        signal_length = 9

        # calculate macd only if we have enough values
        if len(self.memory) >= slow:
            values = np.array(self.memory[-slow:])
            macd = self.computeMACD(values, slow, fast, signal_length)
            self.macd_memory.append(macd)

        # calculate signal line only when we have enough macd values
        if len(self.macd_memory) >= signal_length:
            values = np.array(self.macd_memory[-signal_length:])
            signal = self.ExpMovingAverage(values, signal_length)
            self.signal_memory.append(signal)

            if signal < macd and self.env.portfolio.btc > 0:
                # sell at market
                quantity = self.env.portfolio.btc
                self.env.orderbook.new_marketorder(quantity, False)
            elif signal > macd and self.env.portfolio.usd >= closing_price > 0:
                # buy at market
                quantity = self.env.portfolio.usd // closing_price
                self.env.orderbook.new_marketorder(quantity)


class relativestrength(simulator.decisionmaker):
    def __init__(self, environment):
        super(relativestrength, self).__init__(environment)
        self.memory = []
        self.rsi_memory = []
        self.overbought = 70
        self.oversold = 30
        self.period = 14

    def make_decision(self, row):
        closing_price = row[-1]
        self.memory.append(closing_price)
        if len(self.memory) >1:
            values = self.memory[-(self.period+1):]
            U, D = zip(*[(max(0, values[i+1]-values[i]), max(0, values[i]-values[i+1])) for i in range(len(values)-1)])
            U, D = np.array(U).mean(), np.array(D).mean()
            rs =  U/D if (D != 0) else 1
            rsi = 100 - (100 / (1 + rs))
            self.rsi_memory.append(rsi)
            if rsi >= self.overbought:
                # sell at market
                quantity = self.env.portfolio.btc
                self.env.orderbook.new_marketorder(quantity, False)
            elif rsi <= self.oversold:
                # buy at market
                quantity = self.env.portfolio.usd//closing_price
                self.env.orderbook.new_marketorder(quantity)
