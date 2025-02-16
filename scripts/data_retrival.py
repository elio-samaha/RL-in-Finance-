from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

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

    # Define API request
    client = initialize_client()
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start, 
        end=end)
    
    print(f"\nRetrieving data for: {symbol} from {start} to {end}...")
    bars = client.get_stock_bars(request)
    bars = bars.df

    bars.reset_index(level='symbol', drop=True, inplace=True) # Remove multi-index

    bars.to_csv(f'./{save_dir}/{symbol}_{timeframe}_{start}_{end}.csv')
    print(f"Data saved to: {save_dir}/{symbol}_{timeframe}_{start}_{end}.csv\n")


save_bar_data('INTC', TimeFrame.Minute, '2024-02-01', '2025-02-01', 'data')