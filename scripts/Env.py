import numpy as np
import pandas as pd
from gym import spaces
from gym_trading_env.environments import TradingEnv

import torch


class POMDPTEnv(TradingEnv):
    def __init__(self, df, window_size=60, initial_balance=100_000,transaction_cost=2.3e-5, slippage=0.2, eta=0.01):
        super().__init__(df=df)
        
        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(6 + 4 * window_size,), # OHLCV + 4 indicators + account
            dtype=np.float32
            )
        self.action_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(2,), 
            dtype=np.float32
        ) # [P long, P short]

        # Reward variables
        self.eta = eta
        self.alpha = 0
        self.beta = 0

        self.cumulative_profit = 0 

        # Initialize
        self.vectorize()
        self.reset()
        
    def _compute_dual_thrust(self, k1=0.3, k2=0.3):
        idx = self.current_step - 1

        hh = self.high_rolling_max[idx]
        hc = self.close_rolling_max[idx]
        lc = self.close_rolling_min[idx]
        ll = self.low_rolling_min[idx]

        if np.isnan(hh) or np.isnan(hc) or np.isnan(lc) or np.isnan(ll): # Check necessary for first window_size steps
            #set them to open price so that range = 0
            hh = hc = lc = ll = self.opens[idx]

        self.range = max(hh - lc, hc - ll)
        self.buy_line = self.opens[idx] + k1 * self.range
        self.sell_line = self.opens[idx] - k2 * self.range


    def _compute_differential_sharpe_ratio(self, reward, eps=1e-6):
        delta_alpha = reward - self.alpha
        delta_beta = reward**2 - self.beta

        if (self.beta - self.alpha**2) < eps:
            dsr = 0
        else:
            dsr = (self.beta*delta_alpha - 0.5*self.alpha*delta_beta) / (self.beta - self.alpha**2)**1.5

        # update
        self.alpha += self.eta * delta_alpha
        self.beta += self.eta * delta_beta

        return dsr


    def _next_observation(self):
        start_idx = self.current_step - self.window_size
        end_idx   = self.current_step
        
        prices = np.array([
            self.opens[start_idx:end_idx],
            self.highs[start_idx:end_idx],
            self.lows[start_idx:end_idx],
            self.closes[start_idx:end_idx]
        ]).flatten() # (4 * window_size)

        self._compute_dual_thrust()
        indicators = [self.buy_line, self.sell_line, self.volume[self.current_step], self.trade_count[self.current_step]]

        account = [self.position, self.cumulative_profit]
        return np.concatenate([prices, indicators, account])
    
    def reset(self):
        self.current_step = self.window_size

        self.balance = self.initial_balance
        self.position = 0 # 0: no position, 1: long, -1: short

        self.entry_price = 0
        self.buy_line = 0
        self.sell_line = 0
        
        self.alpha = 0
        self.beta = 0

        # first observation (update lines)
        self._compute_dual_thrust()
        return self._next_observation()

    def vectorize(self):
        
        self.opens = self.df['open'].values
        self.highs = self.df['high'].values
        self.lows  = self.df['low'].values
        self.closes = self.df['close'].values

        self.trade_count = self.df['trade_count'].values
        self.volume = self.df['volume'].values

        self.high_rolling_max  = self.df['high'].rolling(self.window_size).max().values
        self.close_rolling_max = self.df['close'].rolling(self.window_size).max().values
        self.close_rolling_min = self.df['close'].rolling(self.window_size).min().values
        self.low_rolling_min   = self.df['low'].rolling(self.window_size).min().values


    def step(self, action):
            done = False
            if self.current_step >= len(self.df) - 1:
                done = True
            
            desired_position = 1 if action[0] > action[1] else -1

            price_open = self.opens[self.current_step]
            price_close = self.closes[self.current_step]
            prev_close = self.closes[self.current_step-1]
            prev_position = self.position

            # Eq (1)
            rt = (price_close - prev_close - 2 * self.slippage) * prev_position - \
                    (abs(desired_position - prev_position) * self.transaction_cost * price_close)
            
            self.balance += rt
            self.position = desired_position
            self.entry_price = price_open

            self.cumulative_profit += rt
            
            dsr = self._compute_differential_sharpe_ratio(rt)

            # next step
            self.current_step += 1
            if self.current_step < len(self.df):
                self._compute_dual_thrust()
                obs = self._next_observation()
            else:
                obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            
            if self.balance <= 0:
                done = True
            
            return obs, dsr, done, {}
    

# POLICIES

def dt_policy(env):

    buy_line = env.buy_line
    sell_line= env.sell_line

    curr = env.opens[env.current_step]

    action = np.zeros(2, dtype=np.float32)

    if curr > buy_line:
        action[0] = 1.0 # Long
    elif curr < sell_line:
        action[1] = 1.0 # Short
    else: # Do nothing
        if env.position == 1:
            action[0] = 1.0
        elif env.position == -1:
            action[1] = 1.0
    
    return action
    

def intraday_greedy_actions(env):

    day_len = 391  

    high_prices = env.highs
    low_prices = env.lows
    num_steps = len(high_prices)
    actions = np.zeros(num_steps, dtype=int)

    i = env.window_size
    while i < (num_steps - 1):
        day_start = i
        day_end = min(i + day_len, num_steps)
        
        day_high = high_prices[day_start:day_end]
        day_low = low_prices[day_start:day_end]
        idx_min = np.argmin(day_high).item()  # Buy
        idx_max = np.argmax(day_low).item()  # Sell

        actions[day_start + idx_min] = 0  # Long
        actions[day_start + idx_max] = 1  # Short

        i = day_end  

    return actions
