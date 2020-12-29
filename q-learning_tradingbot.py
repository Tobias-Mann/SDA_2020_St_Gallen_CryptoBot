import numpy as np  # for array stuff and random
from PIL import Image  # for creating visual of our env
import cv2  # for showing our visual live
import matplotlib.pyplot as plt  # for graphing our mean rewards over time
import pickle  # to save/load Q-Tables
from matplotlib import style  # to make pretty charts because it matters.
import time  # using this to keep track of our saved Q-Tables.

style.use("ggplot")  # setting our style!

#SIZE = 10

HM_EPISODES = 25000
TRADE_PENALTY = 1 # could be for entering a trade, like a commission
LOSS_PENALTY = 300 # could be trade loss
PROFIT_REWARD = 25 # could be a trade profit
epsilon = 0.9
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 3000  # how often to play through env visually.

start_q_table = None  # None or Filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95

#PLAYER_N = 1  # player key in dict
#FOOD_N = 2  # food key in dict
#ENEMY_N = 3  # enemy key in dict

# the dict for colors!
#d = {1: (255, 175, 0), 2: (0, 255, 0), 3: (0, 0, 255)}

PRICE = 8000

class Q_learning:
    def __init__(self):
        # set random numbers where the price starts
        self.close = np.random.randint(-PRICE, PRICE)

    def __str__(self):
        # return the x and y coordinates
        return f"{self.x}, {self.y}"

    # simple formula for substraction
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self, choice):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        # this could be our actions for trades
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE - 1:
            self.x = SIZE - 1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE - 1

# if we have no q-table we initialize it with random variables from the SIZE range
# for us this could be the different price combinations for open, high, low, close
if start_q_table is None:
    # we have i, ii = player - food coordinates, iii, iiii = player - enemy coordinates
    q_table = {}
    for i in range(-SIZE + 1, SIZE):
        for ii in range(-SIZE + 1, SIZE):
            for iii in range(-SIZE + 1, SIZE):
                for iiii in range(-SIZE + 1, SIZE):
                    q_table[((i, ii), (iii, iiii))] = [
                        np.random.uniform(-5, 0) for i in range(4)
                    ]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

# can look up from Q-table with: print(q_table[((-9, -2), (3, 9))]) for example

episode_rewards = []

for episode in range(HM_EPISODES):
    player = Blob() # initializes the players coordinates
    food = Blob()
    enemy = Blob()
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(
            f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0

    # 200 is the amount of steps we are taking, this is our amount of trades
    for i in range(200):
        # this could be basically the current price observation
        obs = (player - food, player - enemy)
        #print(obs)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[obs])
        else:
            # this is a random function and could be the same for our trading bot
            action = np.random.randint(0, 4)
        # Take the action!
        player.action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

        # this could represent our penalities for profits and losses
        if player.x == enemy.x and player.y == enemy.y:
            reward = - LOSS_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = PROFIT_REWARD
        else:
            reward = - TRADE_PENALTY

        ## NOW WE KNOW THE REWARD, LET'S CALC YO
        # first we need to obs immediately after the move.
        new_obs = (player - food, player - enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3),
                           dtype=np.uint8)  # starts an rbg of our size
            env[food.x][food.y] = d[
                FOOD_N]  # sets the food location tile to green color
            env[player.x][player.y] = d[
                PLAYER_N]  # sets the player tile to blue
            env[enemy.x][enemy.y] = d[
                ENEMY_N]  # sets the enemy location to red
            img = Image.fromarray(
                env, 'RGB'
            )  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize(
                (300,
                 300))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    #print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards,
                         np.ones((SHOW_EVERY, )) / SHOW_EVERY,
                         mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)