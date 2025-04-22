import os
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import warnings
from utils.logger import get_logger

warnings.filterwarnings("ignore", category=FutureWarning)

class ModelGenerator:
    today_date = datetime.today().strftime('%Y-%m-%d')
    def __init__(self, dataset_directory=f'./data/engineered/{today_date}', forecast_directory=f'./data/forecast/{today_date}', identifier="upstox"):
        """
        Initializes the ModelGenerator for Random Forest forecasting of multiple stocks.

        Args:
            dataset_directory (str): Directory containing engineered CSV files.
            forecast_directory (str): Directory to store forecast CSV files.
            identifier (str): File identifier (default: 'upstox').
        """
        self.logger = get_logger(__name__, log_file='logs/model_generator.log')
        self.identifier = identifier
        self.dataset_directory = dataset_directory
        self.forecast_directory = forecast_directory

        self.dataset_file_name = None
        self.target = 'close'
        self.data = None
        self.train = None
        self.val = None
        self.train_exog = None
        self.val_exog = None
        self.model = None
        self.forecast = None
        # Added to store validation metrics
        self.val_metrics = None

        os.makedirs(forecast_directory, exist_ok=True)

    def get_sorted_files(self, directory):
        """
        Retrieves sorted files from a directory based on timestamp.

        Args:
            directory (str): Directory to list files from.

        Returns:
            list: Sorted list of filenames.
        """
        try:
            files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            return sorted(files, key=lambda x: int(re.search(r'_(\d+)_', x).group(1)) if re.search(r'_(\d+)_', x) else 0)
        except Exception as e:
            self.logger.error(f"Failed to retrieve files from {directory}: {e}")
            return []

    def load_data(self):
        """
        Loads all unprocessed datasets matching the identifier.

        Returns:
            list: List of tuples (filename, pandas.DataFrame) for unprocessed datasets, or empty list if none.
        """
        dataset_files = [f for f in self.get_sorted_files(self.dataset_directory) if f.startswith(self.identifier)]
        forecast_files = [f for f in self.get_sorted_files(self.forecast_directory) if f.startswith(self.identifier)]

        latest_forecast_timestamp = max(
            (int(re.search(r'_(\d+)_', f).group(1)) for f in forecast_files if re.search(r'_(\d+)_', f)),
            default=None
        )

        datasets = []
        for dataset_file in dataset_files:
            try:
                dataset_timestamp = int(re.search(r'_(\d+)_', dataset_file).group(1))
                if latest_forecast_timestamp is None or dataset_timestamp > latest_forecast_timestamp:
                    df = pd.read_csv(os.path.join(self.dataset_directory, dataset_file))
                    if not df.empty:
                        datasets.append((dataset_file, df))
                    else:
                        self.logger.warning(f"Dataset {dataset_file} is empty, skipping.")
                else:
                    self.logger.info(f"Dataset {dataset_file} already processed, skipping.")
            except Exception as e:
                self.logger.error(f"Failed to load dataset {dataset_file}: {e}")

        if not datasets:
            self.logger.info(f"No new datasets available for processing.")
        return datasets

    def preprocess_data(self):
        """
        Preprocesses data by aggregating to weekly, splitting into train/validation sets, and adding lagged features.
        """
        self.data.dropna(inplace=True)

        # Convert 'date' to datetime and set it as index
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)

        # Resample to weekly data (last trading day of the week, typically Friday)
        weekly_data = self.data.resample('W-FRI').agg({
            'close': 'last',
            'volume': 'mean',
            'open_interest': 'mean',
            'MA_5': 'last',
            'MA_10': 'last',
            'MA_20': 'last',
            'MA_50': 'last',
            'MA_100': 'last',
            'EMA_12': 'last',
            'EMA_26': 'last',
            'EMA_50': 'last',
            'EMA_100': 'last',
            'RSI_14': 'last',
            'MACD': 'last',
            'MACD_Signal': 'last',
            'daily_returns': 'mean',
            'volatility_20': 'last',
            'BB_upper': 'last',
            'BB_lower': 'last',
            'BB_width': 'last',
            'stochastic_k': 'last',
            'stochastic_d': 'last',
            'week_of_year': 'last',
            'month': 'last'
        }).dropna()

        # Add weekly lagged features
        weekly_data['close_w-1'] = weekly_data['close'].shift(1)
        weekly_data['close_w-2'] = weekly_data['close'].shift(2)
        weekly_data['close_w-3'] = weekly_data['close'].shift(3)
        weekly_data['RSI_14_w-1'] = weekly_data['RSI_14'].shift(1)
        weekly_data.dropna(inplace=True)  # Drop rows with NaN from lagging

        self.data = weekly_data

        # Log dataset size
        self.logger.info(f"Dataset {self.dataset_file_name} has {len(self.data)} weeks after preprocessing.")

        # Split into train (80%) and validation (20%)
        train_size = int(len(self.data) * 0.8)
        self.train = self.data[self.target].iloc[:train_size]
        self.val = self.data[self.target].iloc[train_size:]

        self.exogenous_columns = [
            'volume', 'open_interest',
            'MA_5', 'MA_10', 'MA_20', 'MA_50', 'MA_100',
            'EMA_12', 'EMA_26', 'EMA_50', 'EMA_100',
            'RSI_14', 'MACD', 'MACD_Signal',
            'daily_returns', 'volatility_20',
            'BB_upper', 'BB_lower', 'BB_width',
            'stochastic_k', 'stochastic_d',
            'week_of_year', 'month',
            'close_w-1', 'close_w-2', 'close_w-3', 'RSI_14_w-1'
        ]

        self.train_exog = self.data[self.exogenous_columns].iloc[:train_size]
        self.val_exog = self.data[self.exogenous_columns].iloc[train_size:]

    def fit_random_forest(self):
        """
        Fits a Random Forest model and computes validation metrics (MAE, RMSE, R^2, MAPE).
        """
        try:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
            self.model.fit(self.train_exog, self.train)
            self.logger.info("Random Forest model successfully fitted.")

            # Compute validation metrics
            if len(self.val) > 0 and len(self.val_exog) > 0:
                val_pred = self.model.predict(self.val_exog)
                # Calculate MAE
                mae = mean_absolute_error(self.val, val_pred)
                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(self.val, val_pred))
                # Calculate R^2
                r2 = r2_score(self.val, val_pred)
                # Calculate MAPE
                if np.any(self.val == 0):
                    self.logger.warning("Zero values in validation set, MAPE undefined")
                    mape = np.nan
                else:
                    mape = np.mean(np.abs((self.val - val_pred) / self.val)) * 100

                self.val_metrics = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R^2': r2,
                    'MAPE': mape
                }
                self.logger.info(f"Validation Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R^2={r2:.4f}, MAPE={mape:.4f}%")
            else:
                self.logger.warning("Validation set is empty, skipping metric calculation")
                self.val_metrics = {'MAE': np.nan, 'RMSE': np.nan, 'R^2': np.nan, 'MAPE': np.nan}

            # Log feature importance
            feature_importance = dict(zip(self.exogenous_columns, self.model.feature_importances_))
            self.logger.info(f"Feature Importance: {{k: round(v, 4) for k, v in feature_importance.items()}}")
        except Exception as e:
            self.logger.error(f"Failed to fit Random Forest model: {e}")
            self.val_metrics = {'MAE': np.nan, 'RMSE': np.nan, 'R^2': np.nan, 'MAPE': np.nan}

    def forecast_future(self, weeks=52):
        """
        Generates weekly forecasts using the fitted Random Forest model.

        Args:
            weeks (int): Number of future weeks to forecast (default: 52).
        """
        try:
            self.logger.info(f"Generating {weeks}-week forecast for {self.dataset_file_name}. Note: Long-term forecasts may have reduced accuracy due to feature extrapolation.")
            if len(self.data) < weeks:
                self.logger.warning(f"Dataset {self.dataset_file_name} has only {len(self.data)} weeks, less than {weeks} required for forecast. Using all available data.")
            recent_data = self.data.iloc[-min(weeks, len(self.data)):].copy()
            future_exog_data = pd.DataFrame(columns=self.exogenous_columns)
            last_known = recent_data.iloc[-1]

            # Initialize with last known values, fill NaN with mean
            for col in self.exogenous_columns:
                if col in recent_data.columns and not pd.isna(last_known[col]):
                    future_exog_data.loc[0, col] = last_known[col]
                else:
                    mean_val = recent_data[col].mean() if col in recent_data.columns else 0
                    future_exog_data.loc[0, col] = mean_val
                    self.logger.warning(f"Filled NaN for {col} with mean {mean_val}")

            # Extrapolate non-lagged features
            non_lagged_features = [col for col in self.exogenous_columns if col not in ['close_w-1', 'close_w-2', 'close_w-3', 'RSI_14_w-1']]
            slopes = []
            for col in non_lagged_features:
                if col in recent_data.columns and not recent_data[col].isna().all():
                    slope = (recent_data[col].iloc[-1] - recent_data[col].iloc[0]) / (len(recent_data) - 1)
                    slopes.append((col, slope))
                else:
                    slopes.append((col, 0))
                    self.logger.warning(f"No valid data for {col}, using slope 0")

            forecasts = []
            last_close = last_known['close']
            second_last_close = recent_data['close'].iloc[-2] if len(recent_data) > 1 else last_close
            third_last_close = recent_data['close'].iloc[-3] if len(recent_data) > 2 else second_last_close
            last_rsi = last_known['RSI_14'] if not pd.isna(last_known['RSI_14']) else recent_data['RSI_14'].mean()

            for week in range(weeks):
                if week == 0:
                    future_exog_data.loc[week, 'close_w-1'] = last_close
                    future_exog_data.loc[week, 'close_w-2'] = second_last_close
                    future_exog_data.loc[week, 'close_w-3'] = third_last_close
                    future_exog_data.loc[week, 'RSI_14_w-1'] = last_rsi
                else:
                    future_exog_data.loc[week] = future_exog_data.loc[week - 1].copy()
                    future_exog_data.loc[week, 'close_w-1'] = forecasts[-1]
                    future_exog_data.loc[week, 'close_w-2'] = future_exog_data.loc[week - 1, 'close_w-1']
                    future_exog_data.loc[week, 'close_w-3'] = future_exog_data.loc[week - 1, 'close_w-2']
                    future_exog_data.loc[week, 'RSI_14_w-1'] = future_exog_data.loc[week - 1, 'RSI_14']

                for col, slope in slopes:
                    future_exog_data.loc[week, col] = last_known[col] + slope * (week + 1) if not pd.isna(last_known[col]) else future_exog_data.loc[0, col]

                pred = self.model.predict(future_exog_data.loc[[week]])[0]
                if np.isnan(pred):
                    self.logger.warning(f"NaN prediction at week {week}, using last valid close")
                    pred = forecasts[-1] if forecasts else last_close
                forecasts.append(pred)

            self.forecast = np.array(forecasts)
            self.logger.info(f"Forecast generated with {len(self.forecast)} weekly predictions.")
            if len(self.forecast) != weeks:
                self.logger.warning(f"Expected {weeks} weekly predictions, but generated {len(self.forecast)}.")
        except Exception as e:
            self.logger.error(f"Failed to generate forecast for {self.dataset_file_name}: {e}")
            self.forecast = None

    def save_forecast(self, weeks=52):
        """
        Saves the weekly forecast and validation metrics to a CSV file.

        Args:
            weeks (int): Number of forecast weeks (default: 52).
        """
        if self.forecast is not None:
            forecast_file_path = os.path.join(self.forecast_directory, self.dataset_file_name)
            try:
                last_date = self.data.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), freq='W-FRI', periods=weeks)

                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    'date': future_dates,
                    'forecast_close': self.forecast
                })

                # Append validation metrics as additional rows
                if self.val_metrics:
                    metrics_df = pd.DataFrame({
                        'date': ['Validation_Metric'] * 4,
                        'forecast_close': [
                            f"MAE: {self.val_metrics['MAE']:.4f}",
                            f"RMSE: {self.val_metrics['RMSE']:.4f}",
                            f"R^2: {self.val_metrics['R^2']:.4f}",
                            f"MAPE: {self.val_metrics['MAPE']:.4f}%"
                        ]
                    })
                    forecast_df = pd.concat([forecast_df, metrics_df], ignore_index=True)

                forecast_df.to_csv(forecast_file_path, index=False)
                self.logger.info(f"Forecast and metrics saved to {forecast_file_path} with {len(forecast_df)} rows (including metrics).")
            except Exception as e:
                self.logger.error(f"Failed to save forecast for {self.dataset_file_name}: {e}")
        else:
            self.logger.warning(f"No forecast available to save for {self.dataset_file_name}.")

    def fit(self, weeks=52):
        """
        Fits the Random Forest model and generates weekly forecasts for all unprocessed stock datasets.

        Args:
            weeks (int): Number of future weeks to forecast (default: 52).
        """
        self.logger.info(f"Processing all datasets in {self.dataset_directory} for {weeks}-week forecasts...")
        datasets = self.load_data()

        if datasets:
            for dataset_file, dataset in datasets:
                self.logger.info(f"Processing dataset: {dataset_file}")
                self.dataset_file_name = dataset_file
                self.data = dataset

                try:
                    self.preprocess_data()
                    if len(self.data) < 52:  # Ensure sufficient weeks for 52-week forecast
                        self.logger.warning(f"Dataset {dataset_file} has too few weeks ({len(self.data)}) for 52-week forecast, skipping.")
                        continue

                    self.fit_random_forest()
                    self.forecast_future(weeks=weeks)
                    self.save_forecast(weeks=weeks)
                except Exception as e:
                    self.logger.error(f"Failed to process dataset {dataset_file}: {e}")
                    continue

            self.logger.info(f"Completed processing {len(datasets)} datasets.")
        else:
            self.logger.info(f"No datasets to process.")

if __name__ == '__main__':
    model_generator = ModelGenerator()
    model_generator.fit(weeks=52)