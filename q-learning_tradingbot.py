import gym
from gym import spaces


class StockTradingEnvironment(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(low = np.array([0, 0]), high = np.array([3, 1]), dtype = np.float16)
        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(low=0, high=1, shape=(6, 6), dtype = np.float16)

