# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)


import tensorflow as tf
# from tensorflow import keras
# assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os

import gym
from gym import spaces
import pandas as pd
import random

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}
    def __init__(self, df_stock_prices, window_size=5, commission=0.1):
        super(StockTradingEnv, self).__init__()
        
        self.MAX_ACCOUNT_BALANCE = 1000000
        self.MAX_NUM_SHARES = 1000
        self.MAX_NET_WORTH = 1000000
        self.MAX_SHARE_PRICE = 1000
        self.MAX_SHARE_VALUE = 1000000

        self.df_original_prices = df_stock_prices
        self.df_original_prices['Date'] = pd.to_datetime(self.df_original_prices['Date'])
        self.df_original_prices = self.df_original_prices.sort_values(by='Date')
        self.df = self.df_original_prices[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].pct_change(periods=1)
        self.df['Date'] = self.df_original_prices.loc[1: ,'Date']
        self.df = self.df.dropna().reset_index().drop('index', axis=1)
        self.df = self.df.reset_index().drop('index', axis=1)
        self.df_original_prices=self.df_original_prices.drop(0,axis=0).reset_index().drop('index', axis=1) 

        # print(self.df.shape[0], self.df_original_prices.shape[0])

        assert(self.df.shape[0]==self.df_original_prices.shape[0])
        assert(self.df.shape[1]==self.df_original_prices.shape[1])

        self.window_size = window_size
        self.reward_range = (0, self.MAX_ACCOUNT_BALANCE)
        self.current_open_price = 0
        self.current_close_price = 0
        self.counter = 0
        self.INITIAL_ACCOUNT_BALANCE = 10000
        self.net_worth_done = self.INITIAL_ACCOUNT_BALANCE
        self.counter_done = self.counter
        self.commission = commission
        self.prev_open_price = 0 
        self.prev_balance = self.INITIAL_ACCOUNT_BALANCE
        self.prev_net_worth = self.INITIAL_ACCOUNT_BALANCE
        self.prev_shares_held = 0
        self.prev_cost_basis = 0
        self.prev_total_sales_value = 0

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the prices history in the sliding window
        # self.observation_space = spaces.Box(
        #     low=0, high=1, shape=(self.window_size+1, self.df.shape[1]-1), dtype=np.float16)

        ## modified observation_space.shape
        # self.observation_space = spaces.Box(
        #     low=0, high=1, shape=((self.window_size+1)*(self.df.shape[1]-2), ), dtype=np.float16)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(30, ), dtype=np.float16)
        
    def getState(self):        
        # Get the stock price changes for the last days (in the window excluding the current day)
        frame = np.array(self.df.loc[self.current_step-self.window_size: self.current_step-1, 
                                     ['Open', 'High', 'Low', 'Close', 'Volume']])
        frame = frame.flatten()

        obs = np.array([self.df.loc[self.current_step].at['Open'], 
                        self.net_worth / self.MAX_NET_WORTH,
                        self.balance / self.MAX_ACCOUNT_BALANCE,
                        self.shares_held / self.MAX_NUM_SHARES,
                        self.cost_basis / self.MAX_SHARE_PRICE])
        obs = np.hstack((obs, frame))

        ## flatten 
        obs = np.atleast_1d(obs.flatten())
        return obs

    def reset(self):
        # print('hello world')
        # Reset the state of the environment to an initial state
        self.balance = self.INITIAL_ACCOUNT_BALANCE
        self.net_worth = self.INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = self.INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.counter = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            self.window_size, self.df.shape[0]-90)
        
        self.current_date = self.df.loc[self.current_step, 'Date']
        # Set the current price as the open price on current day
        self.current_open_price = self.df_original_prices.loc[self.current_step, 'Open']
        self.current_close_price = self.df_original_prices.loc[self.current_step, 'Close']


        self.prev_open_price = self.current_open_price
        self.prev_balance = self.balance
        self.prev_net_worth = self.net_worth
        self.prev_shares_held = 0
        self.prev_cost_basis = 0
        self.prev_total_sales_value = 0
        
        return self.getState()
    
    
    def step(self, action):

        ## record before action
        self.prev_open_price = self.current_open_price
        self.prev_balance = self.balance
        self.prev_net_worth = self.net_worth 
        self.prev_shares_held = self.shares_held
        self.prev_cost_basis = self.cost_basis
        self.prev_total_sales_value = self.total_sales_value
        
        # Execute one time step within the environment
        action_type = action[0]
        amount = action[1]
        
        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / self.current_open_price)
            shares_bought = int(total_possible * amount)
            
#             print('buy price {:0.2f}'.format(self.current_open_price))
            
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * self.current_open_price
            transaction_fee = additional_cost * self.commission

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought
            
        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            net_worth_sold = shares_sold * self.current_open_price
            transaction_fee = net_worth_sold * self.commission
            self.balance += net_worth_sold
            
#             print('sell price {:0.2f}'.format(self.current_open_price))
            
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * self.current_open_price

        else:
            transaction_fee = 0
            
        ## after the action, the net worth is evaluated using current_close_price
        self.net_worth = self.balance + self.shares_held * self.current_close_price - transaction_fee
        
#         print('net_worth {:0.2f}'.format(self.net_worth))
        
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0
            
        # after the transaction, the environment is updated. 
        # increase current_step by 1 
        self.current_step += 1
        # increase counter by 1
        self.counter += 1
        
        # update current date for debugging
        self.current_date = self.df.loc[self.current_step, 'Date']
        # update the current prices 
        self.current_open_price = self.df_original_prices.loc[self.current_step, 'Open']
        self.current_close_price = self.df_original_prices.loc[self.current_step, 'Close']
        
        # reward design 
#         delay_modifier = (self.counter / self.MAX_STEPS)
        delay_modifier = self.counter
        # reward = (self.net_worth / INITIAL_ACCOUNT_BALANCE - 1) * delay_modifier
        reward = (self.net_worth-self.INITIAL_ACCOUNT_BALANCE)/self.counter * delay_modifier

        
        done = self.net_worth <= 6000 or self.current_step>=self.df.shape[0]-1 or self.counter >= 90

        obs = self.getState()

        if(done):
            self.net_worth_done = obs[1]*self.MAX_NET_WORTH
            self.counter_done = self.counter
        return obs, reward, done, {}
        

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
