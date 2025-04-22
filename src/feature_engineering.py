import os
import pandas as pd
import re
import numpy as np
from datetime import datetime
from utils.logger import get_logger

class FeatureEngineer:
    """
    A class for feature engineering of stock market CSV data by adding
    Moving Averages, EMAs, RSI, MACD, Volatility, Returns, Bollinger Bands,
    Stochastic Oscillator, and seasonal features.
    """
    today_date = datetime.today().strftime('%Y-%m-%d')
    def __init__(self, preprocessed_directory=f'./data/processed/{today_date}', engineered_directory=f'./data/engineered/{today_date}', identifier="upstox"):
        self.identifier = identifier
        self.preprocessed_directory = preprocessed_directory
        self.engineered_directory = engineered_directory
        self.logger = get_logger(__name__, log_file='logs/feature_engineer.log')
        os.makedirs(engineered_directory, exist_ok=True)

    def get_sorted_files(self, directory):
        try:
            files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            files.sort(key=lambda x: int(re.search(r'_(\d+)_', x).group(1)) if re.search(r'_(\d+)_', x) else 0)
            return files
        except Exception as e:
            self.logger.error(f"Failed to retrieve files from {directory}: {e}")
            return []

    def get_unengineered_files(self, processed_files, engineered_files):
        engineered_set = {f for f in engineered_files if self.identifier in f}
        unengineered_files = [f for f in processed_files if f not in engineered_set]
        return unengineered_files

    def engineer_daily_dataset(self, file_path, output_directory):
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values(by='date', inplace=True)

            # Moving Averages
            df['MA_5'] = df['close'].rolling(window=5).mean()
            df['MA_10'] = df['close'].rolling(window=10).mean()
            df['MA_20'] = df['close'].rolling(window=20).mean()
            df['MA_50'] = df['close'].rolling(window=50).mean()
            df['MA_100'] = df['close'].rolling(window=100).mean()

            # Exponential Moving Averages
            df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['EMA_100'] = df['close'].ewm(span=100, adjust=False).mean()

            # RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI_14'] = 100 - (100 / (1 + rs))

            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

            # Volatility (20-day standard deviation of returns)
            df['daily_returns'] = df['close'].pct_change()
            df['volatility_20'] = df['daily_returns'].rolling(window=20).std()

            # Bollinger Bands (20-day)
            df['BB_upper'] = df['MA_20'] + 2 * df['close'].rolling(window=20).std()
            df['BB_lower'] = df['MA_20'] - 2 * df['close'].rolling(window=20).std()
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['MA_20']

            # Stochastic Oscillator (14-day)
            df['low_14'] = df['low'].rolling(window=14).min()
            df['high_14'] = df['high'].rolling(window=14).max()
            df['stochastic_k'] = 100 * (df['close'] - df['low_14']) / (df['high_14'] - df['low_14'])
            df['stochastic_d'] = df['stochastic_k'].rolling(window=3).mean()

            # Seasonal Features
            df['week_of_year'] = df['date'].dt.isocalendar().week
            df['month'] = df['date'].dt.month

            # Drop temporary columns
            df.drop(columns=['low_14', 'high_14'], inplace=True)

            # Drop rows with NaN values
            df.dropna(inplace=True)

            output_file_path = os.path.join(output_directory, os.path.basename(file_path))
            df.to_csv(output_file_path, index=False)
            self.logger.info(f"Successfully engineered daily features for {file_path} -> {output_file_path}")

        except Exception as e:
            self.logger.error(f"Failed to engineer daily dataset for {file_path}: {e}")

    def engineer_features(self):
        self.logger.info("Starting feature engineering...")
        preprocessed_files = self.get_sorted_files(self.preprocessed_directory)
        engineered_files = self.get_sorted_files(self.engineered_directory)

        unengineered_files = self.get_unengineered_files(preprocessed_files, engineered_files)

        if not unengineered_files:
            self.logger.warning("No new files to engineer.")
            return

        for preprocessed_file in unengineered_files:
            preprocessed_file_path = os.path.join(self.preprocessed_directory, preprocessed_file)
            self.engineer_daily_dataset(preprocessed_file_path, self.engineered_directory)

        self.logger.info("Feature engineering completed for all files.")

if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.engineer_features()