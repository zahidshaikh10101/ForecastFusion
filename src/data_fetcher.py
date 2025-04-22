import requests
import json
import time
import os
from datetime import datetime, timedelta
from utils.logger import get_logger

class DataFetcher:
    """
    A class to interact with the Upstox API and fetch Indian stock market historical data.
    """
    today_date = datetime.today().strftime('%Y-%m-%d')
    def __init__(self, data_dir=f'./data/raw/{today_date}'):
        """
        Initializes the DataFetcher instance.

        Args:
            data_dir (str): Directory where fetched data will be saved.
        """
        self.logger = get_logger(__name__, log_file='logs/data_fetcher.log')
        self.base_url = "https://api.upstox.com/v3/historical-candle"
        self.instrument_keys = {
            "MAHABANK": "NSE_EQ|INE457A01014",
            "INFY": "NSE_EQ|INE009A01021",
            "KPIT": "NSE_EQ|INE04I401011",
            "NTPC": "NSE_EQ|INE733E01010",
            "CANBK": "NSE_EQ|INE476A01022"
        }
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.headers = {
            "Accept": "application/json",
            # Uncomment and add your actual token below
            # "Authorization": "Bearer YOUR_ACCESS_TOKEN"
        }

    def save_to_file(self, data, filename):
        """
        Saves the data to a local JSON file.

        Args:
            data (dict): Data to save.
            filename (str): Name of the file to save the data in.
        """
        timestamp = time.strftime("%Y%m%d%H%M%S")
        full_path = os.path.join(self.data_dir, f"upstox_{timestamp}_{filename}")

        try:
            with open(full_path, 'w') as file:
                json.dump(data, file, indent=4)
            self.logger.info(f"Data successfully saved to: {full_path}")
        except (IOError, OSError) as e:
            self.logger.error(f"Unable to save data to {full_path}: {e}")

    def fetch_historical_data(self, instrument_name, interval="1", unit="days"):
        """
        Fetches historical data for the specified instrument.

        Args:
            instrument_name (str): Name of the instrument (key in instrument_keys).
            interval (str): Interval (default "1" for daily).
            unit (str): Unit (default "days").
        """
        if instrument_name not in self.instrument_keys:
            self.logger.error(f"Instrument '{instrument_name}' not found.")
            return

        instrument_key = self.instrument_keys[instrument_name]

        to_date = datetime.today().strftime('%Y-%m-%d')
        from_date = (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')

        url = f"{self.base_url}/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}"

        self.logger.info(f"Fetching historical data for {instrument_name} from {from_date} to {to_date}...")

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            self.save_to_file(data, f"{instrument_name}_{unit}_{interval}.json")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch data for {instrument_name}: {e}")

    def fetch_all(self):
        """
        Fetch historical data for all predefined stocks.
        """
        for name in self.instrument_keys:
            self.fetch_historical_data(name)
            time.sleep(1)

if __name__ == "__main__":
    fetcher = DataFetcher()
    fetcher.fetch_all()