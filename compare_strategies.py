import numpy as np
import pandas as pd
import simulator
import strategies
from functools import reduce

STRATEGIESCOLLECTION = {"MACD":strategies.MACD, "SimpleMA":strategies.SimpleMA,
 "meanreversion":strategies.meanreversion, "RSI":strategies.relativestrength}

# read in data
data = pd.read_csv("./Data/Dec19.csv")
data = data.dropna().head(200000)

# simulate multiple strategies
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


# merge function to get NaN at the beginning
def merge_basedonlength (df1, df2, column_name):
    len1 = len(df1)
    len2 = len(df2)
    df1[column_name] = np.NaN
    df1[column_name][-len2:] = df2.iloc[:, 0]

# create dataframes
df = pd.DataFrame(memories['MACD'].memory, columns = ['price'])
df_macd = pd.DataFrame(memories['MACD'].macd_memory, columns = ['macd'])
df_signal = pd.DataFrame(memories['MACD'].signal_memory, columns = ['signal'])
df_short = pd.DataFrame(memories['SimpleMA'].short_memory, columns=['short MA'])
df_long = pd.DataFrame(memories['SimpleMA'].long_memory, columns=['long MA'])
df_rsi = pd.DataFrame(memories['RSI'].rsi_memory, columns=['RSI'])
df_z = pd.DataFrame(memories['meanreversion'].z_memory, columns=['Z-Value'])


# merge dataframes and save them
merge_basedonlength(df, df_macd, 'macd')
merge_basedonlength(df, df_signal, 'signal')
merge_basedonlength(df, df_short, 'short MA')
merge_basedonlength(df, df_long, 'long MA')
merge_basedonlength(df, df_z, 'Z-Value')
merge_basedonlength(df, df_rsi, 'RSI')

print(df)
