import SDA_2020_St_Gallen_CryptoBot.SDA_2020_St_Gallen_04_SmartBalancerSim.qlearning as ql
import numpy as np



# define features
class pct_change_lag(ql.feature):
    def __init__(self, lag):
        super(pct_change_lag, self).__init__()
        self.lag = lag
        self.min_observations = max(1, abs(lag))
        self.low = -1
        self.high = 1

    def calculate(self, observations):
        return (observations[-1]/observations[-self.lag])-1

class z_score_lag(ql.feature):
    def __init__(self, lag):
        super(z_score_lag, self).__init__()
        self.lag = lag
        self.min_observations = max(1, abs(lag))
        self.low = -4
        self.high = 4

    def calculate(self, observations):
        std = observations[-self.lag:].std()
        m_mean = observations[-self.lag:].mean()
        return max(self.low, min((observations[-1] - m_mean) / std, self.high))

class relativestrength_lag(ql.feature):
    def __init__(self, lag):
        super(relativestrength_lag, self).__init__()
        self.lag = lag
        self.min_observations = max(1, abs(lag))
        self.low = -1
        self.high = 1

    def calculate(self, observations):
        returns = np.diff(observations[-self.lag:])/observations[-self.lag:]
        returns = returns[~np.isnan(returns)]
        select = returns > 0
        avg_gain = np.mean(returns[select])
        avg_loss = np.mean(returns[~select])
        rsi = 100
        if avg_loss != 0 and np.any(~np.isnan([avg_gain, avg_loss])):
            rsi = 100 - (100 / (1 + avg_gain/avg_loss))
        return rsi

class rsi(ql.feature):
    def __init__(self, periods):
        super(rsi, self).__init__()
        self.periods = periods
        self.min_observations = max(1, abs(periods+1))
        self.low = 0
        self.high = 100

    def calculate(self, observations):
        values = observations[-(self.periods+1):]
        U, D = zip(*[(max(0, values[i+1]-values[i]), max(0, values[i]-values[i+1])) for i in range(len(values)-1)])
        U, D = np.array(U).mean(), np.array(D).mean()
        rs =  U/D if (D != 0) else 1
        rsi = 100 - (100 / (1 + rs))
        return rsi
# lag is here defined as the short moving average, and long_ma is = 2 * lag
class simplema_lag(ql.feature):
    def __init__(self, lag):
        super(simplema_lag, self).__init__()
        self.lag = lag
        self.min_observations = max(1, abs(lag))
        self.low = -1
        self.high = 1

    def calculate(self, observations):
        short_window = self.lag
        long_window = self.lag * 2
        ma_short = np.convolve(observations[-short_window:],
            np.ones(short_window)/short_window, mode = 'valid')
        ma_long = np.convolve(observations[-long_window:],
            np.ones(long_window)/long_window, mode = 'valid')
        # we return ma_short / ma_long as opposed to returning the features seperately
        return ma_short / ma_long

class macd_lag(ql.feature):
    def __init__(self, lag):
        super(macd_lag, self).__init__()
        self.lag = lag
        self.min_observations = max(1, abs(lag))
        self.low = -1
        self.low = 1
        self.macd_memory = []

    def ExpMovingAverage(self, values, window):
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        ema = (weights * values).sum()
        return ema

    def calculate(self, observations):
        fast = 12
        slow = 26
        signal_length = 9
        if len(observations) >= slow:
            emaslow = self.ExpMovingAverage(observations[-slow:], slow)
            emafast = self.ExpMovingAverage(observations[-fast:], fast)
            macd = emafast - emaslow
            self.macd_memory.append(macd)

        if len(self.macd_memory) >= signal_length:
            signal = self.ExpMovingAverage(self.macd_memory[-signal_length:], signal_length)
            # we return signal / macd_memory as opposed to returning the features seperately
            return signal / self.macd_memory[-1]

