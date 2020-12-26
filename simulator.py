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
        self.transactions = dict{}
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

class simulator:
    def __init__(self):
        self.orderbook = orderbook()
        self.transactionbook = transactionbook()
        self.decisionmaker = None
    
    def initialize_decisionmaker(self, decisionmaker_class):
        self.decisionmaker = decisionmaker_class(self.orderbook, self.transactionbook)
    
    def random_market_price(self, open, high, low, close):
        randoms = np.random.random(4)
        weights = randoms/randoms.sum()
        return ((np.array([ open, high, low, close]) * weights).sum()*2 + open + close)/4
    
    def process_orders(self, time, ohcl):
        # Check for open orders to be filled
        # A filled order needs to change its status from active to filled,
        # aditionally an according transaction is added to the transactions book
        for order in self.orderbook.orders_by_status("active").values():
            filled = False
            
            # update the transactionbook for trades
            if order["is_buy"]:
                if order["type"]=="market":
                    # buy at market
                    price = self.random_market_price(ohcl[0], ohcl[1], ohcl[2], ohcl[3])
                    self.transactionbook.buy(price, order["quantity"], time, order["id"])
                    filled = True
                elif order["limit"] >= ohcl[2]:
                    # buy at limit
                    self.transactionbook.buy(order["limit"], order["quantity"], time, order["id"])
                    filled = True
            else:
                if order["type"]=="market":
                    # sell at market
                    price = self.random_market_price(ohcl[0], ohcl[1], ohcl[2], ohcl[3])
                    self.transactionbook.sell(price, order["quantity"], time, order["id"])
                    filled = True
                elif order["limit"] <= ohcl[1]:
                    # sell at limit
                    self.transactionbook.sell(order["limit"], order["quantity"], time, order["id"])
                    filled = True
                    
            # update the orderbook for trades
            if filled:
                self.orderbook.change_order_status(order["id"], "filled")
        

class decisionmaker:
    def __init__(self, orderbook, transactionbook):
        #The decisiomaker class is a generic class so its functions are meant to be overwritten by actual trading logic
        self.orderbook = orderbook
        self.transactionbook = transactionbook
        
    def observe_new_price

class BestDecsionsEver(decisionmaker_class):
    