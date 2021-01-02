import os
import pandas as pd
from scipy import stats


def test_average_performance(paths, data, threshold = .01):
    # This functin calculates the avgerage cumulative BTC return, and calculates the p-Value that the true cumulative simulation return is equal or below to BTC performance
    cumreturn=lambda x: x[-1]/x[0]-1
    btc = cumreturn(data.close.values)
    sumulation_cumreturns = paths.iloc[-1,:].values
    t, p = stats.ttest_1samp(sumulation_cumreturns, btc)
    if btc > sumulation_cumreturns.mean():
        print(f"The strategy appears to perform worse on average than simply holding BTC")
    elif p < threshold:
        print(f"The t-Statistic for the simulations true cumulative return being identical to the the one of BTC is {round(t,4)}, the according p-Values is {round(p*100, 2)}% and below the {int(100*threshold)}% threshold!")
    else:
        print(f"The hypothesis of a significantly different performance is rejected at the {round(threshold*100,0)}% significance level")
    
def test_volatility(paths, data, threshold = .01):
    

if __name__ == "__main__":
    if os.path.exists("./Data/lastmontecarlosimulation.csv"):
        if os.path.exists("./Data/BTC_USD/Dec19.csv"):
            data = pd.read_csv("./Data/BTC_USD/Dec19.csv")
            paths = pd.read_csv("./Data/lastmontecarlosimulation.csv").set_index("time")
            test_average_performance(paths, data)
        else:
            print("File is mising: ./Data/BTC_USD/Dec19.csv")
    else:
        print("There is no data fo a previous Monte Carlo Simulation. Please run first the simulation to generate the data")