import pandas as pd
import numpy as np
import simulator 

# specify decision maker
class meanreversion(simulator.decisionmaker):
    def __init__(self, environment):
        super(meanreversion, self).__init__(environment)
        self.memory = np.array([])
        self.__critical_deviation__ = 2
        
    def make_decision(self, row):
        self.memory = np.append(self.memory, row["close"])
        n = 50
        values = self.memory[-n:]
        if self.memory.size >= n:
            mean = np.mean(values)
            std = np.std(values)
            z = (row["close"] - mean)/std
            if z > self.__critical_deviation__ and self.env.portfolio.btc >0 :
                # sell at market
                quantity = self.env.portfolio.usd//row["close"]
                self.env.orderbook.new_marketorder(quantity, False)
            elif z < -self.__critical_deviation__ and self.env.portfolio.usd>=row["close"]:
                # buy at market
                quantity = self.env.portfolio.btc
                self.env.orderbook.new_marketorder(quantity)
            

# create simulator
sim = simulator.simulator_environment()
#initialize decision maker
sim.initialize_decisionmaker(meanreversion)

# read in data
data = pd.read_csv("./Data/400 - 1m - Trading Pairs (2013-2020)/btcusd.csv")
data['time'] = pd.to_datetime(data['time'], unit = 'ms')
# start simulation
sim.simulate_on_aggregate_data(data)
print(sim.portfolio.portfolio_over_time())

