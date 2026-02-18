import pandas as pd
import requests
from datetime import datetime, timedelta

class DataCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        
    def fetch_data(self, coin, start_date, end_date):
        """
        Fetches historical price data from CoinGecko API.
        Args:
            coin (str): Cryptocurrency to fetch data for.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
        Returns:
            DataFrame: Contains the price data.
        """
        headers = {'accept': 'application/json'}
        params = {
            'vs_currency': 'usd',
            'from_date': start_date,
            'to_date': end_date
        }
        
        response = requests.get(
            f'https://api.coingecko.com/api/v3/coins/{coin}/market_data',
            headers=headers, params=params
        )
        
        if not response.ok:
            raise ValueError(f"API request failed with status {response.status_code}")
            
        data = response.json()
        df = pd.DataFrame({
            'date': pd.to_datetime(data['time_series']),
            'price': data['prices']
        })
        
        df.set_index('date', inplace=True)
        return df

    def save_to_csv(self, df, filename):
        """
        Saves DataFrame to CSV.
        Args:
            df (DataFrame): Data to save.
            filename (str): Name of the file.
        """
        df.to_csv(filename, mode='a', header=True)