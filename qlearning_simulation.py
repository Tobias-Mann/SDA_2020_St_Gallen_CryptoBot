import numpy as np
import pandas as pd
import qlearning as ql
import features
import simulator
import smartstrategies
import matplotlib.pyplot as plt
from matplotlib import style
from tqdm import tqdm
import multiprocessing as mp
import os

# set random seed to allow reproducable results
np.random.seed(0)

# define actions
n = 3
asp = ql.actionspace([1/(n-1)*x for x in range(n)])



# define observationspace
osp = ql.observationspace()
big_osp = ql.observationspace()

# define features for small obsverationspace
osp.features.append(features.pct_change_lag(1))
osp.features.append(features.pct_change_lag(60))
osp.features.append(features.macd_lag(5))

# define features for large obsverationspace
big_osp.features.append(features.pct_change_lag(60))
big_osp.features.append(features.pct_change_lag(1))
big_osp.features.append(features.z_score_lag(20))
big_osp.features.append(features.z_score_lag(60))
big_osp.features.append(features.rsi(14))

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
data = pd.read_csv("./Data/BTC_USD/Dec19.csv")

# start simulator
sim.simulate_on_aggregate_data(data.dropna(), verbose=True)
sim2.simulate_on_aggregate_data(data.dropna(), verbose=True)


# Show Portfolio Performance
data.columns = ["time", "open","high","low","close","volume"]
print("\nPortfolio 1:\n", sim.env.portfolio.portfolio_repricing(data))
print("\nPortfolio 2:\n", sim2.env.portfolio.portfolio_repricing(data))

# save Portfolios Performance and Tearsheet of Q-Learning Agents:
def save_df(df, folder_path, name, folder_time_name):
    folder = folder_path + folder_time_name + '/'
    if os.path.isdir(folder):
        pass
    else:
        os.mkdir(folder)
    name_temp = name + '.csv'
    df.to_csv(folder + name_temp)

save_df(sim.env.portfolio.portfolio_repricing(data), 'Data/Portfolios/', 'QL1', 'Dec19')
save_df(sim2.env.portfolio.portfolio_repricing(data), 'Data/Portfolios/', 'QL2', 'Dec19')
save_df(sim.env.portfolio.tearsheet(data), 'Data/Tearsheets/', 'QL1', 'Dec19')
save_df(sim2.env.portfolio.tearsheet(data), 'Data/Tearsheets/', 'QL2', 'Dec19')



# define function for performance plot
def save_plot(file, portfolios, data):
    rep = pd.DataFrame(index=data.index, columns=portfolios.keys())
    for name, portfolio in portfolios.items():
        rep[name] = portfolio.portfolio_repricing(data)
    rep["BTC_Returns"] = np.log(1+data.set_index("time")["close"].pct_change()).cumsum()
    rep.plot().get_figure().savefig(file)

save_plot("Q_Learning", {"Sim":sim.env.portfolio,"Sim2":sim.env.portfolio}, data)

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