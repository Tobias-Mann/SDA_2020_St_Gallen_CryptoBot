import numpy as np
import pandas as pd
import qlearning
import simulator

class smartbalancer(simulator.decisionmaker):
    def __init__(self, environment):
        super(smartbalancer, self).__init__(environment)
        self.agent = None
    
    def assign_agent(self, agent):
        self.agent = agent
        self.actions = self.agent.actionspace.n
    
    @property
    def memory(self):
        return self.agent.env.observationspace.observations
     
    @property 
    def ratios(self):
        return [(1/(self.actions-1))*x for x in range(self.actions)]
    
    def features(self):
        return [self.env.portfolio.ratio]
    
    def balance(self, ratio):
        if ratio == 1:
            change = self.env.portfolio.usd // self.memory[-1]
        else:
            target =  self.env.portfolio.usd*ratio / (self.memory[-1]*(1 - ratio))
            change = target - self.env.portfolio.btc
        return change
    
    def make_decision(self, row):
        closing_price = row[-1]
        self.agent.observe(closing_price)
        if self.agent.ready_to_learn:
            r = (self.memory[-1]/self.memory[-2])-1
            w = self.env.portfolio.current_ratio(closing_price)
            if w == 0:
                reward = -r -1
            else:
                reward =  w * r -1
            self.agent.learn(reward)
        if self.agent.is_ready:
            new_ratio = self.agent.act()
            adjustment = self.balance(new_ratio)
            if adjustment < 0 and self.env.portfolio.btc > 0:
                # sell btc
                self.env.orderbook.new_marketorder(abs(adjustment), False)
            elif adjustment > 0 and self.env.portfolio.usd >= closing_price > 0:
                # buy btc
                self.env.orderbook.new_marketorder(adjustment)