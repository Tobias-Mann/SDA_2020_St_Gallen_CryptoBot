import pandas as pd 
import numpy as np

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
    
    def buy(self,price, qyt, time, origin):
        id = self.__assign_id__()
        self.transactions[id] = {"quantity":qyt, "price":price, "is_buy":True, "id":id, "time":time, "origin":origin}
    
    def sell(self,price, qyt, time, origin):
        id = self.__assign_id__()
        self.transactions[id] = {"quantity":qyt, "price":price, "is_buy":False, "id":id, "time":time, "origin":origin}
    
    def get_buys(self):
        pass
    
    def get_sells(self):
        pass

class simulator_environment:
    def __init__(self):
        #self.orderbook = orderbook()
        #self.transactionbook = transactionbook()
        #self.portfolio = portfolio()
        self.env = environment()
        self.decisionmaker = None
        self.closing_prices = np.array([])
    
    def initialize_decisionmaker(self, decisionmakerclass):
        self.decisionmaker = decisionmakerclass(environment=self.env)
    
    def random_market_price(self, open, high, low, close):
        randoms = np.random.random(4)
        weights = randoms/randoms.sum()
        return ((np.array([ open, high, low, close]) * weights).sum()*2 + open + close)/4
    
    def process_orders(self, time, ohcl):
        # Check for open orders to be filled
        # A filled order needs to change its status from active to filled,
        # aditionally an according transaction is added to the transactions book
        for order in self.env.orderbook.orders_by_status("active").values():
            filled = False
            
            # update the transactionbook for trades
            if order["is_buy"]:
                if order["type"]=="market":
                    # buy at market
                    price = self.random_market_price(ohcl[0], ohcl[1], ohcl[2], ohcl[3])
                    self.env.transactionbook.buy(price, order["quantity"], time, order["id"])
                    self.env.portfolio.buy(time, order["quantity"], price)
                    filled = True
                elif order["limit"] >= ohcl[2]:
                    # buy at limit
                    self.env.transactionbook.buy(order["limit"], order["quantity"], time, order["id"])
                    self.env.portfolio.buy(time, order["quantity"], order["limit"])
                    filled = True
            else:
                if order["type"]=="market":
                    # sell at market
                    price = self.random_market_price(ohcl[0], ohcl[1], ohcl[2], ohcl[3])
                    self.env.transactionbook.sell(price, order["quantity"], time, order["id"])
                    self.env.portfolio.sell(time, order["quantity"], price)
                    filled = True
                elif order["limit"] <= ohcl[1]:
                    # sell at limit
                    self.env.transactionbook.sell(order["limit"], order["quantity"], time, order["id"])
                    self.env.portfolio.sell(time, order["quantity"], order["limit"])
                    filled = True
                    
            # update the orderbook for trades
            if filled:
                self.env.orderbook.change_order_status(order["id"], "filled")
        
    def performance(self):
        # Add later -> function to retrive realtime performance evaluation
        pass
    
    def simulate_on_aggregate_data(self, data):
        for row in data.iterrows():
            self.closing_prices = np.append(self.closing_prices, row[1]["close"])
            self.decisionmaker.make_decision(row[1]) # because the decision maker is initialized it can access the simulators orderbook, the function can take additional inputs
            
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
            __update__(time, price)
        cost = quantity * price
        if cost > self.__usd__: raise Exception("Using laverage, bitcoin order exceeds usd funds")
        self.__usd__ -= cost
        self.__btc__ += quantity
        self.__update__(price)
    
    def sell(self, time, quantity, price):
        if not self.__is_initialized__:
            self.__is_initialized__ = True
            __update__(time, price)
        revenue = quantity * price
        if quantity > self.__btc__: raise Exception("Using laverage, bitcoin order exceeds btc funds")
        self.__usd__ += revenue
        self.__btc__ -= quantity
        self.__update__(price)
    
    def __update__(self, time, price):
        self.__position_over_time__.append((time, self.__usd__, self.__btc__, price))
    
    @property
    def portfolio_over_time(self):
        if len(self.__position_over_time__) == 0: raise Execption("No")
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

