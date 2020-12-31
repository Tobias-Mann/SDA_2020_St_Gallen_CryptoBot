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
        return max(self.low, min((observations[-1] - m_mean) / std, self.high))

class relativestrength_lag(ql.feature):
    def __init__(self, lag):
        super(relativestrength_lag, self).__init__()
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

# lag is here defined as the short moving average, and long_ma is = 2 * lag
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
        ma_long = np.convolve(observations[-long_window:],
            np.ones(long_window)/long_window, mode = 'valid')
        return ma_short
        return ma_long

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
        ema = (weights * values).sum()
        return ema

    def calculate(self, observations):
        fast = 12
        slow = 26
        signal_length = 9
        if len(observations) >= slow:
            emaslow = self.ExpMovingAverage(observations[-slow:], slow)
            emafast = self.ExpMovingAverage(observations[-fast:], fast)
            macd = emafast - emaslow
            self.macd_memory.append(macd)

        if len(self.macd_memory) >= signal_length:
            signal = self.ExpMovingAverage(self.macd_memory[-signal_length:], signal_length)
            return self.macd_memory[-1]
            return signal



# define observationspace
osp = ql.observationspace()


osp.features.append(pct_change_lag(1))
osp.features.append(pct_change_lag(60))

big_osp = ql.observationspace()
big_osp.features.append(pct_change_lag(1))
big_osp.features.append(pct_change_lag(60))
big_osp.features.append(z_score_lag(20))
big_osp.features.append(z_score_lag(60))
osp.features.append(macd_lag(5))

# Build q-environment
env = ql.environment(osp, asp)
big_env = ql.environment(big_osp, asp)

# Build agent
agent = ql.agent(env)
agent2 = ql.agent(big_env)

# setup simulator
sim = simulator.simulator_environment()
sim2 = simulator.simulator_environment()

# link simulator to smartbalancer
sim.initialize_decisionmaker(smartstrategies.smartbalancer)
sim2.initialize_decisionmaker(smartstrategies.smartbalancer)

# assign agent to smart balancer
sim.decisionmaker.agent = agent
sim2.decisionmaker.agent = agent2

# read in data
data = pd.read_csv("./Data/Dec19.csv")

# start simulator
sim.simulate_on_aggregate_data(data.dropna(), verbose=True)
sim2.simulate_on_aggregate_data(data.dropna(), verbose=True)


# Show Portfolio Performance
data.columns = ["time", "open","high","low","close","volume"]
print("\nPortfolio 1:\n", sim.env.portfolio.portfolio_over_time)
print("\nPortfolio 2:\n", sim2.env.portfolio.portfolio_over_time)