import pandas as pd
import numpy as np
import simulator 

# specify decision maker
class meanreversion(simulator.decisionmaker):
    def __init__(self, environment):
        super(meanreversion, self).__init__(environment)
        self.memory = []
        self.__critical_deviation__ = 2
        
    def change_critical_deviation(self, new):
        self.__critical_deviation__ = new
        
    def make_decision(self, row):
        self.memory.append(row[-1])
        n = 50
        if len(self.memory) >= n:
            values = np.array(self.memory[-n:])
            mean = np.mean(values)
            std = np.std(values)
            z = (row[-1] - mean)/std
            if z > self.__critical_deviation__ and self.env.portfolio.btc >0 :
                # sell at market
                quantity = self.env.portfolio.btc
                self.env.orderbook.new_marketorder(quantity, False)
            elif z < -self.__critical_deviation__ and self.env.portfolio.usd>=row[-1]:
                # buy at market
                quantity = self.env.portfolio.usd//row[-1]
                self.env.orderbook.new_marketorder(quantity)


# create simulator
sim = simulator.simulator_environment()
#initialize decision maker
sim.initialize_decisionmaker(meanreversion)

# read in data
data = pd.read_csv("./Data/400 - 1m - Trading Pairs (2013-2020)/btcusd.csv")
data['time'] = pd.to_datetime(data['time'], unit = 'ms')

# start simulation --> This will take some time!!!
sim.simulate_on_aggregate_data(data.dropna().head(200000))

# retrieve Portfolio performance over simulated time
print(sim.env.portfolio.portfolio_over_time)

