import os
import json
import csv
import re
from datetime import datetime
from utils.logger import get_logger

class DataPreprocessor:
    """
    A class to handle processing of raw Upstox API JSON data and converting it into CSV format.
    """
    today_date = datetime.today().strftime('%Y-%m-%d')
    def __init__(self, raw_directory=f'./data/raw/{today_date}', processed_directory=f'./data/processed/{today_date}', identifier="upstox"):
        """
        Initializes the DataPreprocessor.

        Args:
            raw_directory (str): Directory containing raw JSON files.
            processed_directory (str): Directory to store processed CSV files.
            identifier (str): Fixed identifier used in filenames.
        """
        self.logger = get_logger(__name__, log_file='logs/data_preprocessor.log')
        self.raw_directory = raw_directory
        self.processed_directory = processed_directory
        self.identifier = identifier

        os.makedirs(processed_directory, exist_ok=True)

    def get_sorted_files(self, directory):
        """
        Retrieves all files in a directory, sorted by timestamp in the filename.

        Args:
            directory (str): The directory to list files from.

        Returns:
            list: A list of filenames sorted by timestamp.
        """
        try:
            files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            files.sort(key=lambda x: int(re.search(r'_(\d+)_', x).group(1)) if re.search(r'_(\d+)_', x) else 0)
            return files
        except Exception as e:
            self.logger.error(f"Failed to retrieve files from {directory}: {e}")
            return []

    def get_unprocessed_files(self, raw_files, processed_files):
        """
        Identifies files present in the raw directory but missing in the processed directory.

        Args:
            raw_files (list): List of files in the raw directory.
            processed_files (list): List of files in the processed directory.

        Returns:
            list: A list of filenames missing from the processed directory.
        """
        processed_set = {f.replace('.csv', '.json') for f in processed_files if self.identifier in f}
        return [f for f in raw_files if f not in processed_set]

    def convert_json_to_csv(self, raw_file_path, csv_file_path):
        """
        Converts Upstox candle JSON data to CSV format with timezone-adjusted date column.
        
        Args:
            raw_file_path (str): Path to the JSON file.
            csv_file_path (str): Path to the output CSV file.
        """
        try:
            with open(raw_file_path, 'r') as json_file:
                data = json.load(json_file)

            candles = data.get("data", {}).get("candles", [])

            if not candles:
                raise ValueError("No candles data found in JSON.")

            rows = []
            for candle in candles:
                iso_timestamp = candle[0]
                # Convert to just date (YYYY-MM-DD)
                date_only = datetime.fromisoformat(iso_timestamp).date().isoformat()
                rows.append({
                    "date": date_only,
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
                    "open_interest": candle[6]
                })

            with open(csv_file_path, 'w', newline='') as csv_file:
                fieldnames = ["date", "open", "high", "low", "close", "volume", "open_interest"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            self.logger.info(f"Successfully converted {raw_file_path} to {csv_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to convert {raw_file_path} to CSV: {e}")

    def process_raw(self):
        """
        Processes all raw Upstox JSON files and converts them to CSV format.
        """
        self.logger.info("Starting processing of raw JSON files...")
        raw_files = self.get_sorted_files(self.raw_directory)
        processed_files = self.get_sorted_files(self.processed_directory)

        unprocessed_files = self.get_unprocessed_files(raw_files, processed_files)

        if not unprocessed_files:
            self.logger.warning("No new files to process.")
            return

        for raw_file in unprocessed_files:
            raw_file_path = os.path.join(self.raw_directory, raw_file)
            csv_file_name = raw_file.replace('.json', '.csv')
            csv_file_path = os.path.join(self.processed_directory, csv_file_name)

            self.logger.info(f"Processing file: {raw_file_path}")
            self.convert_json_to_csv(raw_file_path, csv_file_path)

        self.logger.info("Processing completed for all files.")

if __name__ == "__main__":
    processor = DataPreprocessor()
    processor.process_raw()