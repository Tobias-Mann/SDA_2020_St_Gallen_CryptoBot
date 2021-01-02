import numpy as np
import pandas as pd
from pandas import DataFrame
import simulator
import strategies
import os
from os import listdir
from os.path import isfile, join



# SETUP ---------------------------------------------
# Read in data
data = pd.read_csv("./Data/BTC_USD/Dec19.csv")
data = data.dropna().head(200000)

# Define variables
TIMEPERIOD = 'Dec19'
PATH_PFS = './Data/Portfolios/'
PATH_STRATEGIES = './Data/Strategies/'
PATH_TEARSHEETS = './Data/Tearsheets/'

# Define stregies to be tested
STRATEGIESCOLLECTION = {
    "SimpleMA": strategies.SimpleMA,
    "MACD": strategies.MACD,
    "RSI": strategies.relativestrength,
    "meanreversion": strategies.meanreversion,
    "BuyAndHold":strategies.buyandhold
}

# SIMULATE MULTIPLE STRATEGIES --------------------------
portfolios = {}
memories = {}
for name, strategy in STRATEGIESCOLLECTION.items():
    sim = simulator.simulator_environment()
    sim.initialize_decisionmaker(strategy)
    sim.simulate_on_aggregate_data(data)
    portfolios[name] = sim.env.portfolio
    memories[name] = sim.decisionmaker

data.columns = ["time", "open","high","low","close","volume"]

def dataframebycolumn(column):
    colzip = [(name, p.portfolio_repricing(data)[column].values) for name, p in portfolios.items()]
    columns, colvals = list(zip(*colzip))
    return pd.DataFrame(zip(*colvals), columns=columns, index=data["time"])

df_cumulative = dataframebycolumn("cumreturn")
df_Absolute = dataframebycolumn("value")


# GENERAL FUNCTIONS ------------------------------------
# get files with a .csv ending
def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


# merge function to get NaN at the beginning
def merge_basedonlength(df1, df2, column_name):
    len1 = len(df1)
    len2 = len(df2)
    df1[column_name] = np.NaN
    df1[column_name][-len2:] = df2.iloc[:, 0]

# merge dfs
def merge_dfs(path, filenames, column):
    count = 0
    df = None
#    filenames = filenames_ts
#    column = 'Performance Summary'
#    path = (PATH_TEARSHEETS + TIMEPERIOD + '/')
    for name in filenames:
        if filenames[count] == 'merged_cumreturn.csv' or filenames[count] == 'merged_tearsheet.csv':
            count += 1
        else:
            df1 = pd.read_csv(path + filenames[count])
            if df is None:
                df = pd.DataFrame(index = df1.iloc[:, 0], columns = None)
            name = name.rstrip('.csv')
            df[name] = df1[column].values
            count += 1
    return df

### SAVE SHEETS FOR LATER ---------------------------------

# Tearsheets: Save Signle
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

# Tearsheets: Merge and savll all in one
filenames_ts = find_csv_filenames(PATH_TEARSHEETS + TIMEPERIOD + '/') 
merged_tearsheets = merge_dfs((PATH_TEARSHEETS + TIMEPERIOD + '/'), filenames_ts , 'Performance Summary')
merged_tearsheets.to_csv(PATH_TEARSHEETS + TIMEPERIOD + '/merged_tearsheet.csv')

# Porfolios: Save Single
def save_portfolios (folder_time_name):
    for name, strategy in STRATEGIESCOLLECTION.items():
        folder = './Data/Portfolios/' + folder_time_name + '/'
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
    folder = './Data/Strategies/'
    name_temp = name + '.csv'

    if os.path.isdir(folder):
        pass
    else:
        os.mkdir(folder)

    df.to_csv(folder + name_temp)

strategies_name = 'Strategies_' + TIMEPERIOD
save_strategies(strategies_name)