import multiprocessing as mp
import pandas as pd 
import numpy as np

import SDA_2020_St_Gallen_02_SimpleStratSim.simulator as simulator
import SDA_2020_St_Gallen_CryptoBot.SDA_2020_St_Gallen_04_SmartBalancerSim.qlearning as ql
import SDA_2020_St_Gallen_CryptoBot.SDA_2020_St_Gallen_04_SmartBalancerSim.smartstrategies as smartstrategies
import SDA_2020_St_Gallen_CryptoBot.SDA_2020_St_Gallen_04_SmartBalancerSim.features as features

from tqdm import tqdm


def perform_mc_simulation(env, data, repetitions = 100, output="./lastmontecarlosimulation.csv"):
    # the q_table is asigned with the random starting values when the agent is initialized,
    # thus the same ql.environment can be reused in the simulation but the agent needs to be initialized each time
    performance_aggregator = pd.DataFrame(index=data.index, columns=range(repetitions))
    manager = mp.Manager()
    return_dict = manager.dict()
    jobs = []
    inputs = []
    print(f"Starting Monte Carlo Simulation with {repetitions} repetitions, this will take a long time:\n")
    # define function for one simulation
    def simple_simulation(i, env, data, return_dict):
            # each simulation enforces the process to use a different seed, otherwise the random numbers in use will be the same for each simulation
            np.random.seed(i)
            agent = ql.agent(env)
            sim = simulator.simulator_environment()
            sim.initialize_decisionmaker(smartstrategies.smartbalancer)
            sim.decisionmaker.agent = agent
            sim.simulate_on_aggregate_data(data, verbose=False)
            return_dict[i] = sim.env.portfolio.portfolio_repricing(data)["cumreturn"].values
    
    for i in range(repetitions):
        p = mp.Process(target=simple_simulation, args=(i, env, data, return_dict))
        jobs.append(p)
        p.start()
    
    for proc in jobs:
        proc.join()
    
    print(f"\nDone with Generating Paths!\nAppending...\n")
    for i, cumreturn in return_dict.items():
        performance_aggregator[i] = cumreturn
    data = data.set_index("time")
    performance_aggregator.index = data.index
    if output is not None:
        performance_aggregator.to_csv(output)
        print(f"Saved the Paths in: {output}")
    return performance_aggregator

if __name__ == "__main__":
    data = pd.read_csv("../Data/Dec19.csv")
    data =  data[["time", "open","high","low","close","volume"]]
    np.random.seed(0)
    # actionspace
    n = 3
    asp = ql.actionspace([1/(n-1)*x for x in range(n)])
    # observationspace
    big_osp = ql.observationspace()
    big_osp.features.append(features.pct_change_lag(1))
    big_osp.features.append(features.pct_change_lag(60))
    big_osp.features.append(features.z_score_lag(20))
    big_osp.features.append(features.z_score_lag(60))
    big_osp.features.append(features.rsi(14))
    big_env = ql.environment(big_osp, asp)

    monti = perform_mc_simulation(big_env, data.dropna(), 100, output="./Dec19_MC_Paths.csv")