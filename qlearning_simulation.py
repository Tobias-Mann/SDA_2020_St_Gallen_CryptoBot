import numpy as np
import pandas as pd
import qlearning as ql
import simulator
import smartstrategies
import matplotlib.pyplot as plt 
from matplotlib import style
from tqdm import tqdm
import multiprocessing as mp 

style.use("seaborn")
plt.close("all")


# set random seed to allow reproducable results
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

class rsi(ql.feature):
    def __init__(self, periods):
        super(rsi, self).__init__()
        self.periods = periods
        self.min_observations = max(1, abs(periods+1))
        self.low = 0
        self.high = 100
    
    def calculate(self, observations):
        values = observations[-(self.periods+1):]
        U, D = zip(*[(max(0, values[i+1]-values[i]), max(0, values[i]-values[i+1])) for i in range(len(values)-1)])
        U, D = np.array(U).mean(), np.array(D).mean()
        rs =  U/D if (D != 0) else 1
        rsi = 100 - (100 / (1 + rs))
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
        # we return ma_short / ma_long as opposed to returning the features seperately
        return ma_short / ma_long

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
            # we return signal / macd_memory as opposed to returning the features seperately
            return signal / self.macd_memory[-1]



# define observationspace
osp = ql.observationspace()


osp.features.append(pct_change_lag(1))
osp.features.append(pct_change_lag(60))
osp.features.append(macd_lag(5))

big_osp = ql.observationspace()
big_osp.features.append(pct_change_lag(1))
big_osp.features.append(pct_change_lag(60))
big_osp.features.append(z_score_lag(20))
big_osp.features.append(z_score_lag(60))
big_osp.features.append(rsi(14))

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
#sim.simulate_on_aggregate_data(data.dropna(), verbose=True)
#sim2.simulate_on_aggregate_data(data.dropna(), verbose=True)


# Show Portfolio Performance
data.columns = ["time", "open","high","low","close","volume"]
#print("\nPortfolio 1:\n", sim.env.portfolio.portfolio_over_time)
#print("\nPortfolio 2:\n", sim2.env.portfolio.portfolio_over_time)

# define function for performance plot
def save_plot(name, portfolios, data):
    rep = pd.DataFrame(index=data.index, columns=portfolios.keys())
    for name, portfolio in portfolios.items():
        rep[name] = portfolio.portfolio_repricing(data)
    rep["BTC_Returns"] = np.log(1+data.set_index("time")["close"].pct_change()).cumsum()
    rep.plot().get_figure().savefig(name)

#save_plot("Q_Learning", {"Sim":sim.env.portfolio,"Sim2":sim.env.portfolio}, data)

'''
# perform montecarlo simulation ----------------------------
# to speed up the mc simulation multiprocessing is used

def perform_mc_simulation(env, data, repetitions = 1000):
    # the q_table is asigned with the random starting values when the agent is initialized,
    # thus the same ql.environment can be reused in the simulation but the agent needs to be initialized each time
    performance_aggregator = pd.DataFrame(index=data.index, columns=range(repetitions))
    manager = mp.Manager()
    return_dict = manager.dict()
    jobs = []
    inputs = []
    print(f"Starting Monte Carlo Simulation with {repetitions} repetitions, this will take a long time:\n")
    pbar = tqdm(total=repetitions)
    # define function for one simulation
    def simple_simulation(i, env, data, return_dict):
            # each simulation enforces the process to use a different seed, otherwise the random numbers in use will be the same for each simulation
            np.random.seed(i)
            agent = ql.agent(env)
            sim = simulator.simulator_environment()
            sim.initialize_decisionmaker(smartstrategies.smartbalancer)
            sim.decisionmaker.agent = agent
            sim.simulate_on_aggregate_data(data, verbose=False)
            return_dict[i] = sim.env.portfolio.portfolio_repricing(data)["cumreturn"].values
            pbar.update(1)
    
    for i in range(repetitions):
        p = mp.Process(target=simple_simulation, args=(i, env, data, return_dict))
        jobs.append(p)
        p.start()
    
    for proc in jobs:
        proc.join()
    
    print(f"\nDone with Generating Paths!\nAppending...\n")
    for i, cumreturn in return_dict.items():
        performance_aggregator[i] = cumreturn
    data = data.set_index("time")
    performance_aggregator.index = data.index
    performance_aggregator.to_csv("./Data/lastmontecarlosimulation.csv")
    return performance_aggregator

monti = perform_mc_simulation(big_env, data.dropna(), 100)
'''