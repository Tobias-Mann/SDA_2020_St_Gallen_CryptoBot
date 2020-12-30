import gym
import pandas as pd
import numpy as np
from gym import spaces
from sklearn import preprocessing

lookback_window_size = 5
initial_balance = 10000
comision = 0.001
# serial =
MAX_TRADING_SESSION = 100000  # ~2 months


class BitcoinTradingEnv(gym.Env):
    """A Bitcoin trading environment for OpenAI gym"""

    metadata = {'render.modes': ['live', 'file', 'none']}
    scaler = preprocessing.MinMaxScaler()
    viewer = None

    def __init__(self, df, lookback_window_size=50,
                           commission=0.00075,
                           initial_balance=10000,
                           serial=False):
        super(BitcoinTradingEnv, self).__init__()
        self.df = df.dropna().reset_index()
        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.serial = serial
        # Actions of the format Buy 1/10, Sell 3/10, Hold, etc.
        self.action_space = spaces.MultiDiscrete([3, 10])
        # Observes the OHCLV values, net worth, and trade history
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,
                        lookback_window_size + 1), dtype=np.float16)

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.btc_held = 0
        self._reset_session()
        self.account_history = np.repeat([[self.net_worth], [0], [0], [0], [0]],
                                        self.lookback_window_size + 1, # +1 to account for current step 
                                        axis=1)
        self.trades = []
        return self._next_observation()

    def _reset_session(self):
        self.current_step = 0
        if self.serial:
            self.steps_left = len(self.df) - self.lookback_window_size - 1
            self.frame_start = self.lookback_window_size
        else:
            self.steps_left = np.random.randint(1, MAX_TRADING_SESSION)
            self.frame_start = np.random.randint(self.lookback_window_size,
                                                len(self.df) - self.steps_left)
        self.active_df = self.df[self.frame_start -
                                self.lookback_window_size:self.frame_start +
                                self.steps_left]

    def _next_observation(self):
        end = self.current_step + self.lookback_window_size + 1
        obs = np.array([
            self.active_df['Open'].values[self.current_step:end],
            self.active_df['High'].values[self.current_step:end],
            self.active_df['Low'].values[self.current_step:end],
            self.active_df['Close'].values[self.current_step:end],
            self.active_df['Volume_(BTC)'].values[self.current_step:end],
        ])
        scaled_history = self.scaler.fit_transform(self.account_history)
        obs = np.append(obs, scaled_history[:, -(self.lookback_window_size + 1):], axis=0)
        return obs

    def step(self, action):
        current_price = self._get_current_price() + 0.01
        self._take_action(action, current_price)
        self.steps_left -= 1
        self.current_step += 1

        # if we are at the end of the steps, sell everything
        if self.steps_left == 0:
            self.balance += self.btc_held * current_price
            self.btc_held = 0
            self._reset_session()

        # set reward to our current net worth
        obs = self._next_observation()
        reward = self.net_worth
        done = self.net_worth <= 0
        return obs, reward, done, {}
    
    def _take_action(self, action, current_price):
        action_type = action[0]
        amount = action[1] / 10 # to get a percentage value: 10%, 20%, ...100%
        btc_bought = 0
        btc_sold = 0
        cost = 0
        sales = 0
        # BUY
        if action_type < 1:
            btc_bought = self.balance / current_price * amount
            cost = btc_bought * current_price * (1 + self.commission)
            self.btc_held += btc_bought
            self.balance -= cost
        # SELL 
        elif action_type < 2:
            btc_sold = self.btc_held * amount
            sales = btc_sold * current_price  * (1 - self.commission)
            self.btc_held -= btc_sold
            self.balance += sales
        
        # 
        if btc_sold > 0 or btc_bought > 0:
            self.trades.append({
            'step': self.frame_start + self.current_step,
            'amount': btc_sold if btc_sold > 0 else btc_bought,
            'total': sales if btc_sold > 0 else cost,
            'type': "sell" if btc_sold > 0 else "buy"
            })

        # Update Net Worth and Account History
        self.net_worth = self.balance + self.btc_held * current_price
        self.account_history = np.append(self.account_history, [
            [self.net_worth],
            [btc_bought],
            [cost],
            [btc_sold],
            [sales]
        ], axis=1)

# MANUAL -------------------------------
df = pd.read_csv('Data/Dec19.csv')

# init ---------
current_step = 0
serial = False
scaler = preprocessing.MinMaxScaler()


# Reset Session ---------
if serial:
    steps_left = len(df) - lookback_window_size - 1
    frame_start = lookback_window_size
else:
    steps_left = np.random.randint(1, MAX_TRADING_SESSION) #max amount of steps we take
    frame_start = np.random.randint(lookback_window_size,len(df) - steps_left)

active_df = df[frame_start - lookback_window_size:frame_start + steps_left]


# Next Observation ------------
end = current_step + lookback_window_size + 1

obs = np.array([
active_df['open'].values[current_step:end],
active_df['high'].values[current_step:end],
active_df['low'].values[current_step:end],
active_df['close'].values[current_step:end],
active_df['volume'].values[current_step:end],
])

net_worth = 10000
account_history = np.repeat([[net_worth], [0], [0], [0], [0]], lookback_window_size+1,axis=1)
scaled_history = scaler.fit_transform(account_history)
obs = np.append(obs, scaled_history[:, -(lookback_window_size + 1):], axis=0)
