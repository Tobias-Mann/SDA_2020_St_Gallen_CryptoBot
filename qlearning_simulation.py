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