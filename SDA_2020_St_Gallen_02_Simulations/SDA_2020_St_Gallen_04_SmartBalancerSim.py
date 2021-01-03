import numpy as np
import pandas as pd
import simulator
import qlearning as ql
import features
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
# sim.decisionmaker.agent = agent
sim2.decisionmaker.agent = agent2

# read in data
data = pd.read_csv("./Data/BTC_USD/df_raw.csv")

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

FOLDER_TIME_NAME = '2013-2019'
save_df(sim.env.portfolio.portfolio_repricing(data), 'Data/Portfolios/', 'QL1', FOLDER_TIME_NAME)
save_df(sim2.env.portfolio.portfolio_repricing(data), 'Data/Portfolios/', 'QL2', FOLDER_TIME_NAME)
save_df(sim.env.portfolio.tearsheet(data), 'Data/Tearsheets/', 'QL1', FOLDER_TIME_NAME)
save_df(sim2.env.portfolio.tearsheet(data), 'Data/Tearsheets/', 'QL2', FOLDER_TIME_NAME)



# define function for performance plot
def save_plot(file, portfolios, data):
    rep = pd.DataFrame(index=data.index, columns=portfolios.keys())
    for name, portfolio in portfolios.items():
        rep[name] = portfolio.portfolio_repricing(data)
    rep["BTC_Returns"] = np.log(1+data.set_index("time")["close"].pct_change()).cumsum()
    rep.plot().get_figure().savefig("./Images/"+file)

save_plot("Q_Learning", {"Sim":sim.env.portfolio,"Sim2":sim.env.portfolio}, data)
