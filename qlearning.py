import numpy as np

class feature:
    def __init__(self):
        self.high = 1
        self.low = 0
        self.min_observations = 1
    
    def set_space(self, high=None, low=None):
        if high is not None:
            self.high = high
        if low is not None:
            self.low = low
    
class observationspace:
    def __init__(self):
        self.features = []
        self.observations = []
        self.states = []
    
    def add_observation(self, observation):
        self.observations.append(observation)
        if self.min_observations <= len(self.observations):
            obs = np.array(self.observations)
            self.states.append(tuple([f.calculate(obs) for f in self.features]))
    
    @property
    def min_observations(self):
        return max([f.min_observations for f in self.features])
    
    @property
    def low(self):
        return np.array([f.low for f in self.features])
    
    @property
    def high(self):
        return np.array([f.high for f in self.features])
    
class actionspace:
    def __init__(self, a):
        self.actions = a
        
    @property
    def n(self):
        return len(self.actions)

class environment:
    def __init__(self, observationspace, actionspace):
        self.observationspace = observationspace
        self.actionspace = actionspace
    
    def step(self, action):
        pass # new state, reward, done, _
    
    def reset(self):
        return self.observationspace.states[0]
    
class agent:
    def __init__(self, env):
        self.env = env
        self.DISCRETE_OS_SIZE = [5] * len(env.observationspace.features) # could be improved to be feature dependent
        self.discrete_os_win_size = np.array([f.high - f.low for f in env.observationspace.features]) / self.DISCRETE_OS_SIZE
        self.q_table = np.random.uniform(low=-2, high=0, size=(self.DISCRETE_OS_SIZE + [env.actionspace.n] ))
        self.LEARNING_RATE = .1
        self.DISCOUNT = 0.95
        
        self.epsilon_start = 0.5
        self.epsilon_decay_counter = 0
        self.epsilon_decay_limit = 10000
        
        self.min_observations = self.env.observationspace.min_observations
        self.__ready_to_learn__ = False
        self.action_memory = []
    
    def get_discrete_state(self, state):
        state = [ 0 if (x is None) else x for x in state] # treatment of dauf
        discretestate = np.array([max(0, min((state - self.env.observationspace.low)[i], self.q_table.shape[i]-1)) for i in range(len(state))])
        return tuple(discretestate.astype(np.int))
    
    def find_action(self, discretestate):
        return np.argmax(self.q_table[discretestate])
    
    def learn(self, reward):
        # At this time the agent has already made a new observation, but did not act on it, thus -2 (and -1) for observations, (actions)
        # print( self.env.observationspace.states[-1])
        # print(self.get_discrete_state( self.env.observationspace.states[-1] ) )
        max_future_q = np.max( self.q_table[ self.get_discrete_state( self.env.observationspace.states[-1] ) ] )
        last_q = self.q_table[self.get_discrete_state(self.env.observationspace.states[-2])][self.action_memory[-1]]
        new_q = (1- self.LEARNING_RATE) * last_q + self.LEARNING_RATE * (reward + self.DISCOUNT * max_future_q)
        self.q_table[self.get_discrete_state(self.env.observationspace.states[-2])][self.action_memory[-1]] = new_q
            
    def observe(self, observation):
        self.env.observationspace.add_observation(observation)
    
    def act(self):
        if len(self.env.observationspace.observations) == self.min_observations:
            discrete_state = self.get_discrete_state(self.env.reset())
            self.__ready_to_learn__ = True
        elif len(self.env.observationspace.observations) > self.min_observations:
            discrete_state = self.get_discrete_state(self.env.observationspace.states[-1])
        
        if np.random.random() > self.epsilon:
            choosen_action = self.find_action(discrete_state)
        else:
            choosen_action = np.random.randint(0, self.env.actionspace.n)
        self.epsilon_decay_counter += 1
        self.action_memory.append(choosen_action)
        return self.env.actionspace.actions[choosen_action]
    
    def new_data(self):
        self.__ready_to_learn__ = False
        self.env.observationspace.observations = []
        self.env.observationspace.states = []
    
    @property
    def ready_to_learn(self):
        return self.__ready_to_learn__
    
    @property
    def is_ready(self):
        return len(self.env.observationspace.observations) >= self.min_observations
    
    @property
    def epsilon(self):
        return self.epsilon_start - self.epsilon_decay_value * min(self.epsilon_decay_counter, self.epsilon_decay_limit)
    
    @property
    def epsilon_decay_value(self):
        return self.epsilon_start / self.epsilon_decay_limit