import numpy as np
from gym import spaces
from gym_trading_env.environments import TradingEnv

import torch




class POMDPTEnv(TradingEnv):
    def __init__(self, df, window_size=60, initial_balance=100_000,transaction_cost=2.3e-5, slippage=0.2, eta=0.01, alpha=0, beta=0):
        super().__init__(df=df)
        
        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(4 + 4 * window_size,), # OHLCV + 2 indicators + account
            )
        self.action_space = spaces.Discrete(3) # buy or sell

        # Reward variables
        self.eta = eta
        self.alpha = alpha
        self.beta = beta

        # Initialize
        self.vectorize()
        self.reset()
        
    def _compute_dual_thrust(self, k1=0.3, k2=0.3):
        idx = self.current_step - 1

        hh = self.high_rolling_max[idx]
        hc = self.close_rolling_max[idx]
        lc = self.close_rolling_min[idx]
        ll = self.low_rolling_min[idx]

        if np.isnan(hh) or np.isnan(hc) or np.isnan(lc) or np.isnan(ll):
            #set them to open price so that range = 0
            hh = hc = lc = ll = self.opens[idx]

        self.range = max(hh - lc, hc - ll)
        self.buy_line = self.df['open'].iloc[-1] + k1 * self.range
        self.sell_line = self.df['open'].iloc[-1] - k2 * self.range

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
        indicators = [self.buy_line, self.sell_line]

        account = [self.position, self.balance/self.initial_balance]
        return np.concatenate((prices, indicators, account))
    
    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0 # 0: no position, 1: long, -1: short
        self.entry_price = 0
        self.buy_line = 0
        self.sell_line = 0

        # first observation (update lines)
        self._compute_dual_thrust()
        return self._next_observation()

    def vectorize(self):
        
        self.opens = self.df['open'].values
        self.highs = self.df['high'].values
        self.lows  = self.df['low'].values
        self.closes = self.df['close'].values

        self.high_rolling_max  = self.df['high'].rolling(self.window_size).max().values
        self.close_rolling_max = self.df['close'].rolling(self.window_size).max().values
        self.close_rolling_min = self.df['close'].rolling(self.window_size).min().values
        self.low_rolling_min   = self.df['low'].rolling(self.window_size).min().values



    def step(self, action):
            done = False
            if self.current_step >= len(self.df) - 1:
                done = True
            
            if action == 1:
                desired_position = 1
            elif action == 2:
                desired_position = -1
            else:
                desired_position = self.position

            price_open = self.opens[self.current_step]
            
            # Close hold open new
            if desired_position != self.position:
                # close old
                if self.position != 0:
                    old_pnl = (price_open - self.entry_price) * self.position
                    cost = (abs(self.position - desired_position) * self.transaction_cost * price_open 
                            + abs(self.position)*self.slippage)
                    self.balance += old_pnl - cost
                
                # open new
                if desired_position != 0:
                    self.entry_price = price_open
                    cost = (abs(self.position - desired_position) * self.transaction_cost * price_open
                            + abs(desired_position)*self.slippage)
                    self.balance -= cost
                
                self.position = desired_position
            
            # next step
            self.current_step += 1
            if self.current_step < len(self.df):
                self._compute_dual_thrust()
            
            price_close = self.closes[self.current_step-1]
            step_pnl = (price_close - self.entry_price) * self.position
            
            dsr = self._compute_differential_sharpe_ratio(step_pnl)
            obs = self._next_observation()  if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
            
            if self.balance <= 0:
                done = True
            
            return obs, dsr, done, {}
    

# POLICIES

def dt_policy(env):

    o = env._next_observation()
    n = env.window_size

    buy_line = o[4*n]
    sell_line= o[4*n + 1]

    curr = env.opens[env.current_step]

    if curr > buy_line:
        return 1
    elif curr < sell_line:
        return 2
    else:
        return 0
    

def intraday_greedy_actions(env_df, window_size=60, device="cuda"):

    num_steps = len(env_df)
    day_len = 240  

    open_prices = torch.tensor(env_df["open"].values, dtype=torch.float32, device=device)
    actions = torch.zeros(num_steps, dtype=torch.int, device=device)

    i = window_size
    while i < (num_steps - 1):
        day_start = i
        day_end = min(i + day_len, num_steps)
        
        day_opens = open_prices[day_start:day_end]
        idx_min = torch.argmin(day_opens).item()  # Buy
        idx_max = torch.argmax(day_opens).item()  # Sell

        actions[day_start + idx_min] = 1  # Long
        actions[day_start + idx_max] = 2  # Short

        i = day_end  

    return actions