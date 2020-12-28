import numpy as np
import pandas as pd
import simulator
import strategies

STRATEGIESCOLLECTION = { "relativestrength":strategies.relativestrength, "meanreversion":strategies.meanreversion, "SimpleMA":strategies.SimpleMA,"MACD":strategies.MACD}


# read in data
data = pd.read_csv("./Data/Dec19.csv")
data = data.dropna().head(200000)

# simulate multiple strategies
portfolios = {}
for name, strategy in STRATEGIESCOLLECTION.items():
    sim = simulator.simulator_environment()
    sim.initialize_decisionmaker(strategy)
    sim.simulate_on_aggregate_data(data)
    portfolios[name] = sim.env.portfolio.portfolio_over_time

portfolios["SimpleMA"]
    