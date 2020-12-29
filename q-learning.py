### This is jus a Trial Run of the code by SENTDEX

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make ("MountainCar-v0")
env.reset()

# Q-Learning Setups
LEARNING_RATE = 0.1
DISCOUNT = 0.95 # How much we value future rewards over current rewards (between 0 and 1) will go down 0.95*0.95*...
EPISODES = 1000 # the amount of iterations
EPSILON = 0.5 # The higher the value the more random the search will be
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2 # // is to always divide out to an integer
EPSILON_DECAY_VALUE = EPSILON / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
SHOW_EVERY = 500 # show what it is doing for every X episode

# Create a table of 20 for the oberservation values 
# Usually this 20 shouldn't always be hard-coded
DISCRETE_OBS_SIZE = [20] * len(env.observation_space.high) 
discrete_obs_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SIZE

# initialize the q-table (the variables 2 and 0 you should programm flexibly)
q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OBS_SIZE + [env.action_space.n]))

# create a list for rewards
ep_rewards = []
aggre_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

# function to get the discrete state index
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_obs_win_size
    # A tuple is a collection which is ordered and unchangeable:
    return tuple(discrete_state.astype(np.int)) 


"""
### prints -----------------

# print the restrictions
print('env.observation_space.high', env.observation_space.high) # we don't always know the values of these
print('env.observation_space.low', env.observation_space.low)
print('env.action_space.n', env.action_space.n) # how many actions we can take

print('\n discrete_obs_win_size', discrete_obs_win_size)
# each bucket will print a value that is 0.09 long and 0.007 long

# check the dimensionalities of our q_table
print('\n q_table.shape', q_table.shape) # shows us we have a 20x20x3 (3-dimensional) table
print('\n q_table', q_table)

# check what the initial discrete_state combination shows and which action it would take for the initial combination
print('discrete_state', discrete_state)
print(q_table[discrete_state]) # returns the values in the q-table for the index discrete state
print(np.argmax(q_table[discrete_state])) # returns highest values, thus which action we take
"""

for episode in range(EPISODES):
    # initialize the episode reward
    episode_rewards = 0

    # shows us for every nth episode the episode value to know that it is working
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else: 
        render = False

    # return the first value for the discrete state
    discrete_state = get_discrete_state(env.reset()) #env.reset() returns us the initial state
    done = False

    while not done: 
        # if a random number is in our epsilon range then we do the usual max function
        if np.random.random() > EPSILON: # np.random.gives gives us a number between 0 and 1
            action = np.argmax(q_table[discrete_state])
        # otherwise we just have a random action
        else:
            action = np.random.randint(0, env.action_space.n) # between 0 and env.action_space.n

        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        #update the reward
        episode_rewards += reward

        # print(reward, new_state) # prints the values that are changed 
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )] # discrete sate and (action, ) are tuples

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            print(f"we made it on episode {episode}")
            q_table[discrete_state + (action, )] = 0
        
        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        EPSILON -= EPSILON_DECAY_VALUE # -= is an operation

    # append the episode reward
    ep_rewards.append(episode_rewards)

    if not episode % SHOW_EVERY == 0: 
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggre_ep_rewards['ep'].append(episode)
        aggre_ep_rewards['avg'].append(average_reward)
        aggre_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggre_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f'Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}')

# close the environment
env.close()

# plot the findings
plt.plot(aggre_ep_rewards['ep'], aggre_ep_rewards['avg'], label = 'avg rewards')
plt.plot(aggre_ep_rewards['ep'], aggre_ep_rewards['min'], label = 'min rewards')
plt.plot(aggre_ep_rewards['ep'], aggre_ep_rewards['max'], label = 'max rewards')
plt.legend(loc=4) # 4 is the location, here the lower right
plt.show()
