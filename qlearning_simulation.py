import numpy as np
import pandas as pd
import qlearning as ql
import simulator
import smartstrategies

# set random seed
np.random.seed(0)

# define actions
n = 3
asp = ql.actionspace([1/(n-1)*x for x in range(n)])

# define features
class pct_change_lag(ql.feature):
    def __init__(self, lag):
        super(pct_change_lag, self).__init__()
        self.lag = lag
        self.min_observations = max(1, abs(lag))
        self.low = -1
        self.high = 1

    def calculate(self, observations):
        return (observations[-1]/observations[-self.lag])-1

class z_score_lag(ql.feature):
    def __init__(self, lag):
        super(z_score_lag, self).__init__()
        self.lag = lag
        self.min_observations = max(1, abs(lag))
        self.low = -4
        self.high = 4

    def calculate(self, observations):
        std = observations[-self.lag:].std()
        m_mean = observations[-self.lag:].mean()
        return (observations[-1] - m_mean) / std

class relativestrength_lag(ql.feature):
    def __init__(self, lag):
        super(relative_strength, self).__init__()
        self.lag = lag
        self.min_observations = max(1, abs(lag))
        self.low = -1
        self.high = 1

    def calculate(self, observations):
        returns = np.diff(observations[-self.lag:])/observations[-self.lag:]
        returns = returns[~np.isnan(returns)]
        select = returns > 0
        avg_gain = np.mean(returns[select])
        avg_loss = np.mean(returns[~select])
        rsi = 100
        if avg_loss != 0 and np.any(~np.isnan([avg_gain, avg_loss])):
            rsi = 100 - (100 / (1 + avg_gain/avg_loss))
        return rsi

class simplema_lag(ql.feature):
    def __init__(self, lag):
        super(simplema_lag, self).__init__()
        self.lag = lag
        self.min_observations = max(1, abs(lag))
        self.low = -1
        self.high = 1

    def calculate(self, observations):
        short_window = self.lag
        long_window = self.lag * 2
        ma_short = np.convolve(observations[-short_window:],
            np.ones(short_window)/short_window, mode = 'valid')
        ma_long = np.convolve(observations[-short_window:],
            np.ones(long_window)/long_window, mode = 'valid')
        return ma_short
        return ma_long
"""
class macd_lag(ql.feature):
    def __init__(self, lag):
        super(macd_lag, self).__init__()
        self.lag = lag
        self.min_observations = max(1, abs(lag))
        self.low = -1
        self.low = 1
        self.macd_memory = []
    
    def ExpMovingAverage(self, values, window):
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        a = (weights * values).sum()
        return a

    def calculate(self, observations):
        fast = self.lag
        slow = self.lag * 2
        signal_length = 
        emaslow = self.ExpMovingAverage(observations, slow)
        emafast = self.ExpMovingAverage(observations, fast)
        macd = emafast - emaslow
        self.macd_memory.append(macd)

        if len(self.macd_memory) >= signal_length
        signal = self.ExpMovingAverage(observations)
        return macd[self.lag]
        return signal[self.lag]
"""


# define observationspace
osp = ql.observationspace()
osp.features.append(pct_change_lag(1))
osp.features.append(pct_change_lag(60))
osp.features.append(z_score_lag(60))

# Build q-environment
env = ql.environment(osp, asp)

# Build agent
agent = ql.agent(env)

# setup simulator
sim = simulator.simulator_environment()
# link simulator to smartbalancer
sim.initialize_decisionmaker(smartstrategies.smartbalancer)

# assign agent to smart balancer
sim.decisionmaker.agent = agent

# read in data
data = pd.read_csv("./Data/Dec19.csv")

# start simulator
sim.simulate_on_aggregate_data(data.dropna())


# Show Portfolio Performance
data.columns = ["time", "open","high","low","close","volume"]
print("\n", sim.env.portfolio.portfolio_over_time)