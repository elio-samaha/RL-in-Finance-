import numpy as np
import pandas as pd
from gym import spaces
from gym_trading_env.environments import TradingEnv


class POMDPTEnv(TradingEnv):
    def __init__(self, df, window_size=5, 
                 initial_balance=100_000,
                 transaction_cost=2.3e-5, 
                 slippage=0.2, 
                 eta=0.01, 
                 k1=0.5, 
                 k2=0.5):
        
        super().__init__(df=df)

        self.window_size = window_size
        self.initial_balance = initial_balance

        self.transaction_cost = transaction_cost
        self.slippage = slippage

        self.k1 = k1
        self.k2 = k2

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
        self.df = self._preprocess_df(df)
        self.vectorize()
        self.reset()

    def _preprocess_df(self, df):
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        df['date'] = df.index.date

        daily_df = df.groupby('date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
        })
        daily_df.dropna(subset=['open','high','low','close'], inplace=True)

        daily_df['HH'] = daily_df['high'].rolling(window=self.window_size).max()
        daily_df['HC'] = daily_df['close'].rolling(window=self.window_size).max()
        daily_df['LC'] = daily_df['close'].rolling(window=self.window_size).min()
        daily_df['LL'] = daily_df['low'].rolling(window=self.window_size).min()

        daily_df['Range'] = np.maximum(daily_df['HH'] - daily_df['LC'], daily_df['HC'] - daily_df['LL'])
        daily_df['BuyLine'] = daily_df['open'] + self.k1 * daily_df['Range']
        daily_df['SellLine'] = daily_df['open'] - self.k2 * daily_df['Range']

        df = df.merge(daily_df[['BuyLine', 'SellLine']], left_on='date', right_index=True, how='left')

        df['BuyLine'] = df['BuyLine'].ffill()
        df['SellLine'] = df['SellLine'].ffill()

        self.first_valid_idx = daily_df.index[self.window_size] # Index of the first valid observation

        return df.drop(columns=['date'])

        
    def _compute_dual_thrust(self):
        idx = self.current_step

        self.buy_point = self.buy_lines[idx]
        self.sell_point = self.sell_lines[idx]


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
        indicators = [self.buy_point, 
                      self.sell_point, 
                      self.volume[self.current_step], 
                      self.trade_count[self.current_step]
                      ]

        account = [self.position, self.cumulative_profit]
        return np.concatenate([prices, indicators, account])
    
    def reset(self):
        self.current_step = np.where(self.df.index.date >= self.first_valid_idx)[0][0]

        self.balance = self.initial_balance
        self.position = 0 # 0: no position, 1: long, -1: short

        self.entry_price = 0
        self.buy_line = 0
        self.sell_line = 0
        
        self.alpha = 1e-6
        self.beta = 1e-6

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

        self.buy_lines = self.df['BuyLine'].values
        self.sell_lines = self.df['SellLine'].values


    def step(self, action):
            done = False
            if self.current_step >= len(self.df) - 1:
                done = True

            prev_position = self.position
            desired_position = 1 if action[0] > action[1] else -1

            price_open = self.opens[self.current_step]
            price_close = self.closes[self.current_step]
            prev_close = self.closes[self.current_step-1] if self.current_step > 0 else price_close

            # Eq (1)
            rt = (price_close - prev_close - 2 * self.slippage) * prev_position

            if desired_position != prev_position and prev_position != 0:
                rt -= abs(prev_position) * self.transaction_cost * price_close
                self.position = 0
            
            if desired_position != prev_position:
                rt -= abs(desired_position) * self.transaction_cost * price_close
                self.position = desired_position
                self.entry_price = price_open

            self.balance += rt
            self.cumulative_profit += rt
            
            dsr = self._compute_differential_sharpe_ratio(rt)

            # next step
            self.current_step += 1
            if self.current_step < len(self.df):
                self._compute_dual_thrust()
                obs = self._next_observation()
            else:
                obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            
            if self.balance <= 0.5 * self.initial_balance:
                done = True
            
            return obs, dsr, done, {}
    

# POLICIES

def dt_policy(env):

    buy_line = env.buy_lines[env.current_step]
    sell_line= env.sell_lines[env.current_step]

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

    day_len = 390

    low_prices = env.lows
    high_prices = env.highs
    num_steps = len(low_prices)
    actions = np.zeros(num_steps, dtype=int)

    i = env.window_size
    while i < (num_steps - 1):
        day_start = i
        day_end = min(i + day_len, num_steps)
        
        day_lows = low_prices[day_start:day_end]
        day_highs = high_prices[day_start:day_end]
        idx_min = np.argmin(day_lows)
        idx_max = np.argmax(day_highs)

        actions[day_start + idx_min] = 0
        actions[day_start + idx_max] = 1
        
        i = day_end  # Move to next day
    return actions
