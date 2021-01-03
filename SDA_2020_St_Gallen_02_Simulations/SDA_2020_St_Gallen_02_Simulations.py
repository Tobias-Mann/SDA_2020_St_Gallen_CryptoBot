import numpy as np
import pandas as pd
from pandas import DataFrame
import simulator
import strategies
import os
from os import listdir
from os.path import isfile, join

# Additional imports for qlearning in the second half
import qlearning as ql
import features
import smartstrategies
from tqdm import tqdm
import multiprocessing as mp
import os

# SETUP FOR SIMPLE TRADING STRATEGIES -------------------------------------------------
# Read in data
data = pd.read_csv("../Data/df_raw.csv")
data = data[["time", "open","high","low","close","volume"]]
data = data[pd.to_datetime(data.time).agg(lambda x: x.year != 2013).values]

# Define variables
TIMEPERIOD = 'Output_2014-2019'
PATH_PFS = './'
PATH_STRATEGIES = './'
PATH_TEARSHEETS = './'

# Define stregies to be tested
STRATEGIESCOLLECTION = {
    "SimpleMA": strategies.SimpleMA,
    "MACD": strategies.MACD,
    "RSI": strategies.relativestrength,
    "meanreversion": strategies.meanreversion,
    "BuyAndHold":strategies.buyandhold
}
"""
# SIMULATE MULTIPLE SIMPLE TRADING STRATEGIES -------------------------------------------------
portfolios = {}
memories = {}
for name, strategy in STRATEGIESCOLLECTION.items():
    sim = simulator.simulator_environment()
    sim.initialize_decisionmaker(strategy)
    print("\n"+name,"\n")
    sim.simulate_on_aggregate_data(data)
    portfolios[name] = sim.env.portfolio
    memories[name] = sim.decisionmaker
print()

def dataframebycolumn(column):
    colzip = [(name, p.portfolio_repricing(data)[column].values) for name, p in portfolios.items()]
    columns, colvals = list(zip(*colzip))
    return pd.DataFrame(list(zip(*colvals)), columns=columns, index=data["time"])

df_cumulative = dataframebycolumn("cumreturn")
df_Absolute = dataframebycolumn("value")


# GENERAL FUNCTIONS -------------------------------------------------
# get files with a .csv ending
def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


# merge function to get NaN at the beginning
def merge_basedonlength(df1, df2, column_name):
    len1 = df1.shape[0]
    len2 = df2.shape[0]
    df1[column_name] = np.NaN
    df1.loc[:, column_name][-len2:] = df2.iloc[:, 0]

# merge dfs
def merge_dfs(path, filenames, column):
    count = 0
    df = None
    for name in filenames:
        if filenames[count] == 'merged_cumreturn.csv' or filenames[count] == 'merged_tearsheet.csv' or 'strategies' in filenames[count]:
            count += 1
        else:
            df1 = pd.read_csv(path + filenames[count])
            if df is None:
                df = pd.DataFrame(index = df1.iloc[:, 0], columns = None)
            name = name.rstrip('.csv')
            df[name] = df1[column].values
            count += 1
    return df

### SAVE SHEETS FOR LATER -------------------------------------------------

# Tearsheets: Save single sheets
def save_tearsheets (folder_time_name):
    for name, strategy in STRATEGIESCOLLECTION.items():
        folder = PATH_TEARSHEETS + folder_time_name + '/'
        if os.path.isdir(folder):
            pass
        else:
            os.mkdir(folder)
        name_temp = name + '.csv'
        df = pd.DataFrame(portfolios[name].tearsheet(data))
        df.to_csv(folder + name_temp)

save_tearsheets(TIMEPERIOD)

# Tearsheets: Merge and save all in one
filenames_ts = find_csv_filenames(PATH_TEARSHEETS + TIMEPERIOD + '/')
merged_tearsheets = merge_dfs((PATH_TEARSHEETS + TIMEPERIOD + '/'), filenames_ts , 'Performance Summary')
merged_tearsheets.to_csv(PATH_TEARSHEETS + TIMEPERIOD + '/merged_tearsheet.csv')

# Porfolios: Save signle sheets
def save_portfolios (folder_time_name):
    for name, strategy in STRATEGIESCOLLECTION.items():
        folder = PATH_STRATEGIES + folder_time_name + '/'
        if os.path.isdir(folder):
            pass
        else:
            os.mkdir(folder)
        name_temp = name + '.csv'
        df = pd.DataFrame(portfolios[name].portfolio_repricing(data))
        df.to_csv(folder + name_temp)

save_portfolios(TIMEPERIOD)


# Porfolios: Merge and save all in one
filenames_pfs = find_csv_filenames(PATH_PFS + TIMEPERIOD + '/')
merged_cumreturns = merge_dfs((PATH_PFS + TIMEPERIOD + '/'), filenames_pfs, 'cumreturn')
merged_cumreturns['time'] = data['time']
merged_cumreturns.to_csv(PATH_PFS + TIMEPERIOD + '/merged_cumreturn.csv')


# Strategies: save merged df
def save_strategies (name):

    # get values
    df = pd.DataFrame(memories['SimpleMA'].memory, columns=['price'])
    df['time'] = data['time']
    df_macd = pd.DataFrame(memories['MACD'].macd_memory, columns=['macd'])
    df_signal = pd.DataFrame(memories['MACD'].signal_memory, columns=['signal'])
    df_short = pd.DataFrame(memories['SimpleMA'].short_memory, columns=['short_ma'])
    df_long = pd.DataFrame(memories['SimpleMA'].long_memory, columns=['long_ma'])
    df_rsi = pd.DataFrame(memories['RSI'].rsi_memory, columns=['rsi'])
    df_z = pd.DataFrame(memories['meanreversion'].z_memory, columns=['z_value'])

    # merge dataframes and save them
    merge_basedonlength(df, df_macd, 'macd')
    merge_basedonlength(df, df_signal, 'signal')
    merge_basedonlength(df, df_short, 'short_ma')
    merge_basedonlength(df, df_long, 'long_ma')
    merge_basedonlength(df, df_z, 'z_value')
    merge_basedonlength(df, df_rsi, 'rsi')

    # save dataframe
    folder = PATH_STRATEGIES + TIMEPERIOD + '/'
    name_temp = name + '.csv'

    if os.path.isdir(folder):
        pass
    else:
        os.mkdir(folder)

    df.to_csv(folder + name_temp)

strategies_name = 'strategies_indicators'
save_strategies(strategies_name)

"""
### Simulations with a QLearning model ### -------------------------------------------------

# set random seed to allow reproducable results
np.random.seed(0)

# define actions
n = 3
asp = ql.actionspace([1/(n-1)*x for x in range(n)])

# define small observationspace
osp = ql.observationspace()
# define bigger observationspace
big_osp = ql.observationspace()

# define features for small obsverationspace
osp.features.append(features.pct_change_lag(1))
osp.features.append(features.pct_change_lag(60))
osp.features.append(features.macd_lag(5))

# define features for bigger obsverationspace
big_osp.features.append(features.pct_change_lag(60))
big_osp.features.append(features.pct_change_lag(1))
big_osp.features.append(features.z_score_lag(20))
big_osp.features.append(features.z_score_lag(60))
big_osp.features.append(features.rsi(14))

# Build q-environment
env = ql.environment(osp, asp)
big_env = ql.environment(big_osp, asp)

# Build agents
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

# # read in data
# data = pd.read_csv("./Data/BTC_USD/Dec.csv")

# start simulator
sim.simulate_on_aggregate_data(data.dropna(), verbose=True)
sim2.simulate_on_aggregate_data(data.dropna(), verbose=True)


# Show Portfolio Performance
data = data[["time", "open","high","low","close","volume"]]
print("\nPortfolio 1:\n", sim.env.portfolio.portfolio_repricing(data))
print("\nPortfolio 2:\n", sim2.env.portfolio.portfolio_repricing(data))

# save Portfolios Performance and Tearsheet of Q-Learning Agents:
def save_df(df, folder_path, name):
    folder = folder_path + '/'
    if os.path.isdir(folder):
        pass
    else:
        os.mkdir(folder)
    name_temp = name + '.csv'
    df.to_csv(folder + name_temp)

save_df(sim.env.portfolio.portfolio_repricing(data), TIMEPERIOD, 'cum_returns_ql1')
save_df(sim2.env.portfolio.portfolio_repricing(data), TIMEPERIOD, 'cum_returns:ql2')
save_df(sim.env.portfolio.tearsheet(data), TIMEPERIOD, 'tearsheet_ql1')
save_df(sim2.env.portfolio.tearsheet(data), TIMEPERIOD, 'tearsheet_ql2')