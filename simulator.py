import pandas as pd 
import numpy as np
from tqdm import tqdm

class orderbook:
    def __init__(self):
        # The all_orders dict associates an order with its id
        # Each order itself is a dict with keys
        # - type: {market, limit}
        # - status:{active, filled, canceled}
        # - is_buy:{True, False}
        # - id:{self reference in the order book}
        # - and for limit orders with a limit value
        self.all_orders = dict()
        self.__last_id__ = 0
    
    def __assign_id__(self):
        self.__last_id__+=1
        return self.__last_id__
    
    def __new__order__(self):
        for key, order in self.all_orders.items():
            if order["status"] == "active":
                self.all_orders[key]["status"]="canceled"
        
    def orders_by_status(self, status):
        return dict([x for x in self.all_orders.items() if x[1]["status"]==status])
    
    @property
    def canceled_orders(self):
        [x for x in self.all_orders if x["status"]=="canceled"]
    
    def change_order_status(self, id, status):
        if status in ["active", "filled", "canceled"]:
            self.all_orders[id]["status"] = status
    
    def new_marketorder(self, quantity, is_buy = True):
        id = self.__assign_id__()
        self.__new__order__()
        self.all_orders[id] = {"type": "market", "is_buy":is_buy, "quantity":quantity, "status":"active", "id":id}
        
    def new_limitorder(self, price, quantity, is_buy = True):
        id = self.__assign_id__()
        self.__new__order__()
        self.all_orders[id] = {"type": "limit", "is_buy":is_buy, "quantity":quantity, "status":"active", "id":id, "limit": price}
    
class transactionbook:
    def __init__(self):
        # The Transaction book is based on a dictionary of transactions
        # each transaction contains a quantity {int}, price:{decimal}, time:{datetime}, is_buy{True, False}, id: {int}, origin{int}
        self.transactions = dict()
        self.__last_id__ = 0
    
    def __assign_id__(self):
        self.__last_id__+=1
        return self.__last_id__
    
    def process(self, transaction, origin):
        id = self.__assign_id__()
        transaction["id"] = id
        transaction["origin"] = origin
        self.transactions[id] = transaction

class simulator_environment:
    def __init__(self):
        #self.orderbook = orderbook()
        #self.transactionbook = transactionbook()
        #self.portfolio = portfolio()
        self.env = environment()
        self.decisionmaker = None
    
    def initialize_decisionmaker(self, decisionmakerclass):
        self.decisionmaker = decisionmakerclass(environment=self.env)
    
    def random_market_price(self, open, high, low, close):
        randoms = np.random.random(4)
        weights = randoms/randoms.sum()
        return ((np.array([ open, high, low, close]) * weights).sum()*2 + open + close)/4
    
    def process_orders(self, time, ohlc):
        # Check for open orders to be filled
        # A filled order needs to change its status from active to filled,
        # aditionally an according transaction is added to the transactions book
        for order in self.env.orderbook.orders_by_status("active").values():
            filled = False
            
            # update the transactionbook for trades
            if order["is_buy"]:
                if order["type"]=="market":
                    # buy at market
                    price = self.random_market_price(ohlc[0], ohlc[1], ohlc[2], ohlc[3])
                    transaction = self.env.portfolio.buy(time, order["quantity"], price)
                    self.env.transactionbook.process(transaction, order["id"])
                    filled = True
                elif order["limit"] >= ohlc[2]:
                    # buy at limit
                    transaction = self.env.portfolio.buy(time, order["quantity"], order["limit"])
                    self.env.transactionbook.process(transaction, order["id"])
                    filled = True
            else:
                if order["type"]=="market":
                    # sell at market
                    price = self.random_market_price(ohlc[0], ohlc[1], ohlc[2], ohlc[3])
                    transaction= self.env.portfolio.sell(time, order["quantity"], price)
                    self.env.transactionbook.process(transaction, order["id"])
                    filled = True
                elif order["limit"] <= ohlc[1]:
                    # sell at limit
                    transaction = self.env.portfolio.sell(time, order["quantity"], order["limit"])
                    self.env.transactionbook.process(transaction, order["id"])
                    filled = True
                    
            # update the orderbook for trades
            if filled:
                self.env.orderbook.change_order_status(order["id"], "filled")
        
    def performance(self):
        # Add later -> function to retrive realtime performance evaluation
        pass
    
    def simulate_on_aggregate_data(self, data):
        print("Starting Simulation:\n")
        for row in tqdm(data.values):
            self.process_orders(row[0], row[1:5])
            self.decisionmaker.make_decision(row[1:5]) # because the decision maker is initialized it can access the simulators orderbook, the function can take additional inputs
            
            
class portfolio:
    def __init__(self, usd = 10**6, btc = 0):
        self.__usd__ = usd
        self.__btc__ = btc
        self.__position_over_time__ = []
        self.__is_initialized__ = False
    
    def value(self, price):
        return self.__btc__ * price + self.__usd__
    
    def buy(self, time, quantity, price):
        if not self.__is_initialized__:
            self.__is_initialized__ = True
            self.__update__(time, price)
        cost = quantity * price
        if cost > self.__usd__:
            quantity = self.__usd__ // price
            cost = quantity * price
        self.__usd__ -= cost
        self.__btc__ += quantity
        self.__update__(time, price)
        return {"quantity":quantity, "price":price, "is_buy":True, "time":time}
    
    def sell(self, time, quantity, price):
        if not self.__is_initialized__:
            self.__is_initialized__ = True
            self.__update__(time, price)
        revenue = quantity * price
        if quantity > self.__btc__: raise Exception("Using laverage, bitcoin order exceeds btc funds")
        self.__usd__ += revenue
        self.__btc__ -= quantity
        self.__update__(time, price)
        return {"quantity":quantity, "price":price, "is_buy":False, "time":time}
    
    def __update__(self, time, price):
        self.__position_over_time__.append((time, self.__usd__, self.__btc__, price))
    
    def portfolio_repricing(self, data):
        df = pd.DataFrame(self.__position_over_time__)
        df.columns = ["Time", "USD", "BTC", "Price"]
        
        p = df.loc[1:,["USD", "BTC", "Time"]].set_index("Time")
        
        answer = pd.DataFrame(index = data["time"].values.flatten(), columns= ["USD", "BTC"])
        answer[answer.index.isin(p.index)] = p
        answer.loc[answer.index[0],:]=df.loc[0,["USD", "BTC"]]
        answer = answer.fillna(method="ffill")
        answer["price"] = data.set_index("time")["close"]
        answer["value"] = answer["price"] * answer["BTC"] + answer["USD"]
        answer["returns"] = answer["value"].pct_change()
        answer["cumreturn"] = answer["value"]/ answer.loc[answer.index[0],"value"] -1
        answer.index = pd.to_datetime(answer.index)
        return answer
    
    def tearsheet(self, data):
        repriced = self.portfolio_repricing(data)
        summary={}
        summary["Start Date"] = repriced.index.min()
        summary["End Date"] = repriced.index.max()
        summary["Total Days"] = (summary["End Date"] - summary["Start Date"]).days
        cagr = np.exp(365 * (np.power((repriced.loc[summary["End Date"], ["value"]]/repriced.loc[summary["Start Date"], ["value"]]), 1/summary["Total Days"])-1).values[0])-1
        summary["CAGR"] = f"{cagr*100}%"
        # summary["Sharpe ratio"] = (repriced.returns.mean() - repriced.price.pct_change().mean()) / (repriced.returns.std() - repriced.price.pct_change().std())
        daily_min = repriced["value"].groupby(repriced.index.date).agg(lambda x: min(x))
        daily_max = repriced["value"].groupby(repriced.index.date).agg(lambda x: max(x))
        max_drawdown = min([ daily_min[day2]/daily_max[day] - 1 for day in daily_max.index for day2 in daily_min.index[daily_min.index>day]])
        summary["Max Drawdown"] = f"{max_drawdown*100}%"
        summary["Cumulative Return"] = f"{repriced.loc[repriced.index[-1], ['cumreturn']].values[0]*100}%"
        daily_returns = repriced['value'].groupby(repriced.index.date).agg(lambda x: x[-1]).pct_change()
        summary["Annual Volatility"] = f"{daily_returns.std()*np.sqrt(365) * 100}%"
        summary["Calmar Ratio"] = cagr / abs(max_drawdown)
        summary["Skew"] = daily_returns.skew()
        summary["Kurtosis"] = daily_returns.kurtosis()
        summary["Absolute Exposure"] = ((repriced["value"] - repriced["USD"]) /repriced["value"] ).mean() * 100
        summary["Net Exposure"] = (repriced["BTC"] * repriced["price"] / repriced["value"]).mean() *100 
        summary["Average Daily Position"] = repriced["BTC"].groupby(repriced.index.date).agg(lambda x: x.mean()).mean()
        summary["Average Daily Turnover (percentage of capital)"] = 100*repriced.groupby(repriced.index.date).agg(lambda x: x["BTC"].diff().abs().sum() * x["price"][-1] / x["value"][-1] ).mean()[0]
        summary["Normalized CAGR"] = (cagr*100 / summary["Absolute Exposure"]) *100
        return pd.DataFrame(index=summary.keys(), data=summary.values(), columns=["Performance Summary"])
    
    @property
    def ratio(self):
        last = self.__position_over_time__[-1]
        exposure = last[-1] *last[-2]
        return exposure / (exposure + last[1])
    
    @property
    def portfolio_over_time(self):
        df = pd.DataFrame(self.__position_over_time__)
        df.columns = ["Time", "USD", "BTC", "Price"]
        df["Value"] = df["USD"] + df["BTC"] * df["Price"]
        df["BTCRatio"] = 1 - df["USD"]/ df["Value"]
        df["Returns"] = df["Value"].pct_change()
        df["CumulativeReturn"] = df["Value"]/(df["Value"].values[0]) - 1
        return df
    
    @property
    def usd(self):
        return self.__usd__
        
    @property
    def btc(self):
        return self.__btc__

class environment:
    def __init__(self):
        self.orderbook = orderbook()
        self. transactionbook = transactionbook()
        self.portfolio = portfolio()


class decisionmaker:
    def __init__(self, environment):
        #The decisiomaker class is a generic class so its functions are meant to be overwritten by actual trading logic
        self.env = environment
        
    def make_decision(self):
        pass

