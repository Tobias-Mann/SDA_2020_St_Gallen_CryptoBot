import numpy as np
import pandas as pd 
import simulator

class meanreversion(simulator.decisionmaker):
    def __init__(self, environment):
        super(meanreversion, self).__init__(environment)
        self.memory = []
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
            if z > self.__critical_deviation__ and self.env.portfolio.btc >0 :
                # sell at market
                quantity = self.env.portfolio.btc
                self.env.orderbook.new_marketorder(quantity, False)
            elif z < -self.__critical_deviation__ and self.env.portfolio.usd>=closing_price:
                # buy at market
                quantity = self.env.portfolio.usd//closing_price
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
        if len(self.memory) <= 1 or closing_price != self.memory[-1]:
            self.memory.append(closing_price)
        values = self.memory[-(self.period+1):]
        returns = np.diff(values)/values[:-1]
        returns = returns[~np.isnan(returns)]
        select = returns>0
        avg_gain = np.mean(returns[select])
        avg_loss = np.mean(returns[~select])
        rsi = 100
        if avg_loss != 0 and np.any(~np.isnan([avg_gain, avg_loss])):
            rsi = 100 - (100 / (1 + avg_gain/avg_loss))
        self.rsi_memory.append(rsi)
        if rsi >= self.overbought:
            # sell at market
            quantity = self.env.portfolio.btc
            self.env.orderbook.new_marketorder(quantity, False)
        elif rsi <= self.oversold:
            # buy at market
            quantity = self.env.portfolio.usd//closing_price
            self.env.orderbook.new_marketorder(quantity)
        