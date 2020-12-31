import numpy as np
import pandas as pd
import simulator
import strategies

STRATEGIESCOLLECTION = {"SimpleMA":strategies.SimpleMA, "MACD":strategies.MACD,
    "RSI":strategies.relativestrength, "meanreversion":strategies.meanreversion}


# read in data
data = pd.read_csv("./Data/Dec19.csv")
data = data.dropna().head(200000)
data.rename(columns = {'datetime': 'time'}, inplace = True)

# simulate multiple strategies
portfolios = {}
memories = {}
for name, strategy in STRATEGIESCOLLECTION.items():
    sim = simulator.simulator_environment()
    sim.initialize_decisionmaker(strategy)
    sim.simulate_on_aggregate_data(data)
    portfolios[name] = sim.env.portfolio
    memories[name] = sim.decisionmaker

"""
data.columns = ["time", "open","high","low","close","volume"]

def dataframebycolumn(column):
    colzip = [(name, p.portfolio_repricing(data)[column].values) for name, p in portfolios.items()]
    columns, colvals = list(zip(*colzip))
    return pd.DataFrame(zip(*colvals), columns=columns, index=data["time"])

df_cumulative = dataframebycolumn("cumreturn")
df_Absolute = dataframebycolumn("value")


# merge function to get NaN at the beginning
def merge_basedonlength(df1, df2, column_name):
    len1 = len(df1)
    len2 = len(df2)
    df1[column_name] = np.NaN
    df1[column_name][-len2:] = df2.iloc[:, 0]

# create dataframes
df = pd.DataFrame(memories['MACD'].memory, columns=['price'])
df['time'] = data['time']
df_macd = pd.DataFrame(memories['MACD'].macd_memory, columns=['macd'])
df_signal = pd.DataFrame(memories['MACD'].signal_memory, columns=['signal'])
df_short = pd.DataFrame(memories['SimpleMA'].short_memory,
                        columns=['short_ma'])
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

print(df)

# save dataframe
df.to_csv('Data/strategies.csv')
"""