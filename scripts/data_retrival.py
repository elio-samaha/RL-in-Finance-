from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import pandas as pd
from datetime import timedelta

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('API_KEY')
secret_key = os.getenv('SECRET_KEY')

API_KEY = api_key
SECRET_KEY = secret_key

def initialize_client():
    return StockHistoricalDataClient(API_KEY, SECRET_KEY)

def check_dir(path_dir:str) -> None:
    os.makedirs(path_dir, exist_ok=True)

def save_bar_data(symbol:str, timeframe, start:str, end:str, save_dir:str) -> None:

    check_dir(save_dir) # Create directory if it doesn't exist

    client = initialize_client()

    start_date = pd.Timestamp(start, tz='America/New_York')
    end_date = pd.Timestamp(end, tz='America/New_York')

    all_data = pd.DataFrame()

    current_start = start_date
    delta = timedelta(days=7)

    while current_start < end_date:
        current_end = min(current_start + delta, end_date)
        
        # Create the request parameters
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=current_start.isoformat(),
            end=current_end.isoformat(),
            limit=10000 # Max limit
        )
        
        # Fetch the data
        bars = client.get_stock_bars(request_params).df
        
        if not bars.empty:
            bars = bars.reset_index()
            bars['timestamp'] = bars['timestamp'].dt.tz_convert('America/New_York')
            bars.set_index('timestamp', inplace=True)
            
            bars = bars.between_time('09:30', '16:00')
            
            all_data = pd.concat([all_data, bars])
        
        current_start = current_end

    all_data.drop(['symbol', 'vwap'], axis=1, inplace=True)

    all_data.to_csv(f'./{save_dir}/{symbol}_{timeframe}_{start}_{end}.csv')
    print(f"Data saved to: {save_dir}/{symbol}_{timeframe}_{start}_{end}.csv\n")


save_bar_data('INTC', TimeFrame.Minute, '2024-02-01', '2025-02-01', 'data')