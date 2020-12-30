import pandas as pd
import numpy as np
import simulator 
import strategies


# create simulator
sim = simulator.simulator_environment()
#initialize decision maker
sim.initialize_decisionmaker(strategies.SimpleMA)

# read in data
data = pd.read_csv("./Data/Dec19.csv")

# start simulation --> This will take some time!!!
sim.simulate_on_aggregate_data(data.dropna().head(200000))

# retrieve Portfolio performance over simulated time
print(sim.env.portfolio.portfolio_over_time)
print(sim.decisionmaker.long_memory)

