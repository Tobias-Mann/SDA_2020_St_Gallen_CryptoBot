import os
import pandas as pd
from scipy import stats

import plotting


def test_average_performance(paths, data, threshold = .01, verbose = False):
    # This functin calculates the avgerage cumulative BTC return, and calculates the p-Value that the true cumulative simulation return is equal or below to BTC performance
    cumreturn=lambda x: x[-1]/x[0]-1
    btc = cumreturn(data.close.values)
    sumulation_cumreturns = paths.iloc[-1,:].values
    t, p = stats.ttest_1samp(sumulation_cumreturns, btc)
    if verbose:
        if btc > sumulation_cumreturns.mean():
            print(f"The strategy appears to perform worse on average than simply holding BTC")
        elif p < threshold:
            print(f"The t-Statistic for the simulations true cumulative return being identical to the the one of BTC is {round(t,4)}, the according p-Values is {round(p*100, 2)}% and below the {int(100*threshold)}% threshold!")
        else:
            print(f"The hypothesis of a significantly different performance is rejected at the {round(threshold*100,0)}% significance level")
    return (t, p)

if __name__ == "__main__":
    file = "../Data/Dec19.csv"
    paths_file = "../Data/Nov17_Paths.csv"
    if os.path.exists(paths_file ):
        if os.path.exists(file):
            data = pd.read_csv(file)
            paths = pd.read_csv(paths_file ).set_index("time")
            test_average_performance(paths, data, verbose=True)
            plotting.create_mc_dist_plot(paths.reset_index(), data, (.9, .6), output="./Images/Nov17.png", title="Qlearning Monte Carlo Simulation vs BTC Nov17")
        else:
            print(f"File is mising: {file}")
    else:
        print("There is no data from a previous Monte Carlo Simulation. Please run first the simulation to generate the data")