# Single-Time Series-Crypto-Bot
In this collaborative project we build a trading bot, which analyzes a single time series of crypto currency prices to identify optimal conditions for entry and exit trades in real-time.

# The Simulation Framework
In order to realistically simulate a bot which takes in price data and makes a trading decision in real time, an extensive collection of python classes has been utilized that modularize the simulation in multiple elements. The logic is best described by an actual trader observing market prices and making a decision to buy or sell. In this case the so called "decisionmaker" decides on the order to be opened, for instance "buy 100 BTC @7563.48". Such decisions are tracked in another class called "Orderbook" (the name is admittedly somewhat confusing, but does not model a full order book as other market participants entering orders are not part of the simulation). The "Simulator" class is responsible for taking in new price data, checking if any orders from the Orderbook can be executed given Open High Low and Close price of the next period and feeds in the Close price into a memory which can be accessed by the aforementioned decision maker (the decision maker currently only considers closing prices.
If the Simulator finds an order from the Orderbook that can be executed the Order is moved to a "Transactionbook". This action causes an update of the simulated "Portfolio" which keeps track of the BTC position and USD Cash position at each transaction. Furthermore, the portfolio class facilitates plotting performance data due to functions such as portfolio_repricing, or tearsheet, which allow to find the portfolio value and performance for any point in time. Additionally, the property portfolio_over_time provides easy access to the portfolio positions and prices at each time of a transaction (admittedly there is some redundancy with the transaction book).
Lastly the logic that ultimately determines the trades is defined by extensions to the decisionmaker model. Those again are divided into "simple strategies" and "smart strategies". Simple strategies are following a deterministic rule. For instance, calculate RSI for the current period and go "all in" with a market order if the calculated RSI is below 30 and there is no open position, or if there is already an open position and the RSI is above 70, sell everything by market order.
The "smart strategies'' are based on a flexible q-learning agent model and are further described in the next section. The below graphic aims to visualize the described Framework.


<p align="center"><img src="https://github.com/Tobias-Mann/SDA_2020_St_Gallen_CryptoBot/blob/main/Smart%20Data.png?raw=true" /></p>

# The Framework for QLearning Strategies
Using the simulation framework described above, this repository contains additional classes and examples to extend the functionality of the decisionmaker class with a Q-Learning Model. To facilitate such an extension an "agent" is assigned to the decision maker. This agent utilizes its own framework to suggest the best action. The provided example (smartbalancer) uses an action space of 3 actions to optimize the share of BTC in the portfolio. The possible actions are the following:

- 0% : Execute orders so there is no BTC exposure/ Portfolio is a simple cash position
- 50%: Execute orders so there is a 50% BTC exposure/ the portfolio results in an approximately equal weighting of cash and BTC
- 100%: Execute orders so the portfolio has the largest possible BTC exposure/ buy as much BTC as possible with funds in portfolio

The agent accesses all previously observed prices and chooses an action based on the price history. The action is then executed in the decision maker by filling the orderbook with the above logic. In order to use Q-Learning to choose the right action, the ever growing number of observed prices needs to be converted into a finite number of features. To get this, feature objects are defined which hold information on:
- a minimum of observations required to calculate the feature, 
- logical restrictions of maximum and minimum values (necessary to define the q-table)
- the logic to calculate the feature value at any given point in time given a sufficient number of past observations

The Q-agent class has an environment which is the collection of its possible actions and an observation space. The observation space is a class which represents in turn a collection of features. Due to this construction model it is possible to dynamically define an observation space and an actionspace, which is used by the agent to initialize its q-table.

After knowing its observation space and actionspace, the agent can be assigned to a decision maker (e.g. smartbalancer). The decisionmaker then waits until enough prices have been observed to calculate all features that the agent requires and when the agent is ready to act it enters new orders according to the portfolio weightings suggested by the agent.

Note, the agent class is designed in a way to support random exploration of its q-table. The default is an epsilon (probability of random action) of 50% which decayes each period for the first 10'000 periods. Parameters typically associated with Q-Learning can be assigned to the agent class: in particular epsilon, a discount rate for the reward function and a learning rate value for the updating of table values.


# Testing a Q-Learning Strategy by Monte Carlo Simulation

For multiple reasons, a single simulation of a Q-Learning model is not yielding a realistic idea of the expected return generated by this strategy. Some notable arguments for the necessity of running multiple simulations are the following:

- The decision rule of the agent (Q-Table) is initialized by random values, a strategy/q-table that might perform well in samples can still have no predictive power. When such a table is tested outside the sample data it might reveal that any in sample returns were purely based on a spurious correlation.
- The simulator class utilizes random numbers to simulate market prices. However, a single simulation might result in a performance that is rather driven by lucky random prices than by a feasible trading strategy. Outperformance is less likely over longer simulation periods, but there is always a small probability that a single simulation performs well because of such random prices and not because of the quality of the strategy itself.
- A single simulation might get stuck with suboptimal behavior and thus not find an optimal solution (even though this is unlikely due to the Q-Learning algorithm including random exploration).

For each of the mentioned reasons it is desirable to run multiple simulations and consider their distribution rather than a single realization of these strategies. This would also be optimal for the aforementioned "Simple Strategies". However, these follow a purely deterministic decision rule, while the q learning is even more vulnerable to random effects, due to the random initialization of the model.

The below plot shows a Q-Learning algorithm (features: 1 period return, 60 periods return, 20 periods deviation from mean, 60 period deviation from mean, 14 periods RSI, Actions: 0% BTC, 50% BTC, 100% BTC in the Portfolio) using simulated market orders. The black line indicates the average performance for Dec 2019 over 100 simulations with different random seeds. While the blue areas mark the 100%, 90%, 60% probability mass of returns generated by such a strategy. Notably, the red line displays the performance of a 100% exposure to BTC for the same period. While at first sight the strategy appears to outperform a simple long exposure to Bitcoin it must be noted that the simulation did not account for any transaction costs like commissions, or spreads and furthermore, it does not take into account liquidity constraints. Therefore, while the strategy might be capable of identifying those moments in which holding BTC is relatively desirable, it cannot be expected that deploying such an algorithm does result in outperforming BTC and might not even generate positive returns at all.
<p align="center"><img src="https://github.com/Tobias-Mann/SDA_2020_St_Gallen_CryptoBot/blob/main/SDA_2020_St_Gallen_04_VisualizeMonteCarlo/Dec19.png?raw=true" /></p>
