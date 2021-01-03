import pandas as pd
import numpy as np
import SDA_2020_St_Gallen_CryptoBot.SDA_2020_St_Gallen_02_SimpleStratSim.simulator as simulator
import SDA_2020_St_Gallen_CryptoBot.SDA_2020_St_Gallen_02_SimpleStratSim.strategies as strategies


# create simulator
sim = simulator.simulator_environment()
#initialize decision maker
sim.initialize_decisionmaker(strategies.relativestrength)

# read in data
data = pd.read_csv("../Data/Dec19.csv")

# start simulation --> This will take some time!!!
sim.simulate_on_aggregate_data(data.dropna())

# retrieve Portfolio performance over simulated time
print(sim.env.portfolio.portfolio_repricing(data))

