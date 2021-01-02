import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

monti = pd.read_csv("./mcdata/lastmontecarlosimulation.csv")
data = pd.read_csv("./Data/Dec19.csv")
data.columns = ["time", "open","high","low","close","volume"]

def create_mc_dist_plot(paths, data, quantiles):
    quantiles = set(list(quantiles) + [1])
    paths["BTC"] = np.log(data.loc[data.index[data["time"].isin(paths["time"].values)] ,["close"]].pct_change()+1).cumsum()
    paths = paths.iloc[63:,:]
    graph = pd.DataFrame(index=paths.index)
    graph["time"] = paths["time"]
    graph["BTC"] = paths["BTC"] * 100
    graph["Mean"] = paths.mean(axis=1).values *100
    quantile_packages = [(1-q, q, ((q+1)/(len(quantiles)+2)/2)) for q in quantiles]
    for q in quantile_packages:
        # Center Confidence interval
        adjust = (1-q[1])/2
        graph[q[0]] = paths.quantile(q[0]-adjust, axis=1).values * 100
        graph[q[1]] = paths.quantile(q[1]+adjust, axis=1).values * 100
    
    fig, ax = plt.subplots(1,1)
    
    mean, = ax.plot(graph.index, graph["Mean"].values, 'k', linewidth=.5)
    btc, = ax.plot(graph.index, graph["BTC"].values, 'r', linewidth=.25)
    handles= [mean, btc]
    labels = ["QLearning Performance", "BTC Performance"]
    for q in quantile_packages:
        ax.fill_between(graph.index, graph[q[0]], graph[q[1]], alpha=q[2], color="b")
        h, = ax.fill(np.NaN, np.NaN, alpha=q[2], color="b")
        handles.append(h)
        labels.append(str(q[1]*100)+'% Confidence Interval')
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(.5, 1),ncol=3, fontsize='xx-small', framealpha=0)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ticks = monti.iloc[::int(monti.index.size/5),:].index[1:].values
    plt.autoscale(tight=True)
    plt.setp(ax, xticks=ticks, xticklabels=pd.to_datetime(paths["time"]).agg(lambda x: x.date())[ticks])
    fig.suptitle("Qlearning Monte Carlo Simulation vs BTC")
    plt.savefig("./Images/MonteCarloDistribution.png")
    plt.close("all")

mg = create_mc_dist_plot(monti, data, (.90, .60))