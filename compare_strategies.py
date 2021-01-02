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

# Define stregies to be tested
STRATEGIESCOLLECTION = {
    "SimpleMA": strategies.SimpleMA,
    "MACD": strategies.MACD,
    "RSI": strategies.relativestrength,
    "meanreversion": strategies.meanreversion
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

### SAVE SHEETS FOR LATER ---------------------------------

# Save Tearsheets
def save_tearsheets (folder_time_name):
    for name, strategy in STRATEGIESCOLLECTION.items():
        folder = './Data/Tearsheets/' + folder_time_name + '/'
        if os.path.isdir(folder):
            pass
        else:
            os.mkdir(folder)
        name_temp = name + '.csv'
        df = pd.DataFrame(portfolios[name].tearsheet(data))
        df.to_csv(folder + name_temp)

save_tearsheets(TIMEPERIOD)

# Save Portfolios
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

# get files with a .csv ending
def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

filenames = find_csv_filenames(PATH_PFS + TIMEPERIOD + '/')

# merge all files to one sheet
def merge_cum_returns(path, filenames):
    df = pd.DataFrame(index = data.index, columns = None)
    df['time'] = data.time
    count = 0
    for name in filenames:
        if filenames[count] == 'merged_cumreturn.csv':
            count += 1
        else:
            df1 = pd.read_csv(path + filenames[count])
            df[name] = df1['cumreturn']
            count += 1
    return df

merged_cumreturns = merge_cum_returns(
    (PATH_PFS + TIMEPERIOD + '/'), filenames)

merged_cumreturns.to_csv(PATH_PFS + TIMEPERIOD + '/merged_cumreturn.csv')


# merge function to get NaN at the beginning
def merge_basedonlength(df1, df2, column_name):
    len1 = len(df1)
    len2 = len(df2)
    df1[column_name] = np.NaN
    df1[column_name][-len2:] = df2.iloc[:, 0]

# create dataframes from memories
def save_strategies (name, folder_time_name):

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
    folder = './Data/Strategies/' + folder_time_name + '/'
    name_temp = name + '.csv'

    if os.path.isdir(folder):
        pass
    else:
        os.mkdir(folder)

    df.to_csv(folder + name_temp)

strategies_name = 'Strategies_' + TIMEPERIOD
save_strategies(strategies_name, TIMEPERIOD)