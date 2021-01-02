import os
import pandas as pd
from scipy import stats


def test_average_performance(paths, data):
    # This functin calculates the avgerage cumulative BTC return, and calculates the p-Value that the true cumulative simulation return is equal or below to BTC performance
    cumreturn=lambda x: x[-1]/x[0]-1
    btc = cumreturn(data.close.values)
    sumulation_cumreturns = paths.iloc[-1,:].values
    t = (btc - sumulation_cumreturns.mean()) / sumulation_cumreturns.std()
    

if __name__ == "__main__":
    if os.path.exists("./Data/lastmontecarlosimulation.csv"):
        if os.path.exists("./Data/BTC_USD/Dec19.csv"):
            data = pd.read_csv("./Data/BTC_USD/Dec19.csv")
            paths = pd.read_csv("./Data/lastmontecarlosimulation.csv").set_index("time")
        else:
            print("File is mising: ./Data/BTC_USD/Dec19.csv")
    else:
        print("There is no data fo a previous Monte Carlo Simulation. Please run first the simulation to generate the data")