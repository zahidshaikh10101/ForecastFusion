import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
from utils.logger import get_logger

class ModelEvaluator:
    """
    A class to evaluate Random Forest model performance by plotting actual vs predicted values
    for the validation set and extracting metrics (MAE, RMSE, MAPE) from forecast files.
    """
    today_date = datetime.today().strftime('%Y-%m-%d')
    def __init__(self, dataset_directory=f'./data/engineered/{today_date}', 
                 forecast_directory=f'./data/forecast/{today_date}', 
                 evaluation_directory=f'./data/evaluation/{today_date}', 
                 identifier="upstox"):
        """
        Initializes the ModelEvaluator.

        Args:
            dataset_directory (str): Directory containing engineered CSV files.
            forecast_directory (str): Directory containing forecast CSV files.
            evaluation_directory (str): Directory to store evaluation results and plots.
            identifier (str): File identifier (default: 'upstox').
        """
        self.logger = get_logger(__name__, log_file='logs/evaluation.log')
        self.identifier = identifier
        self.dataset_directory = dataset_directory
        self.forecast_directory = forecast_directory
        self.evaluation_directory = evaluation_directory

        os.makedirs(evaluation_directory, exist_ok=True)
        sns.set_style("whitegrid")

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
        Loads forecast and corresponding engineered datasets for evaluation.

        Returns:
            list: List of tuples (filename, forecast_df, actual_df) for evaluation.
        """
        forecast_files = [f for f in self.get_sorted_files(self.forecast_directory) if f.startswith(self.identifier)]
        dataset_files = [f for f in self.get_sorted_files(self.dataset_directory) if f.startswith(self.identifier)]

        datasets = []
        for forecast_file in forecast_files:
            try:
                # Match forecast file with corresponding dataset file
                dataset_file = next((df for df in dataset_files if df == forecast_file), None)
                if not dataset_file:
                    self.logger.warning(f"No matching dataset file for {forecast_file}, skipping.")
                    continue

                # Load forecast data
                forecast_df = pd.read_csv(os.path.join(self.forecast_directory, forecast_file))
                # Load actual data
                actual_df = pd.read_csv(os.path.join(self.dataset_directory, dataset_file))
                actual_df['date'] = pd.to_datetime(actual_df['date'])

                datasets.append((forecast_file, forecast_df, actual_df))
            except Exception as e:
                self.logger.error(f"Failed to load data for {forecast_file}: {e}")

        if not datasets:
            self.logger.info("No datasets available for evaluation.")
        return datasets

    def extract_validation_metrics(self, forecast_df):
        """
        Extracts validation metrics from the forecast DataFrame.

        Args:
            forecast_df (pd.DataFrame): DataFrame containing forecast and validation metrics.

        Returns:
            dict: Dictionary containing MAE, RMSE, MAPE, or None if metrics are not found.
        """
        try:
            metrics_df = forecast_df[forecast_df['date'] == 'Validation_Metric']
            if metrics_df.empty:
                self.logger.warning("No validation metrics found in forecast data.")
                return None

            metrics = {}
            for _, row in metrics_df.iterrows():
                metric_str = row['forecast_close']
                if 'MAE' in metric_str:
                    metrics['MAE'] = float(metric_str.split(': ')[1])
                elif 'RMSE' in metric_str:
                    metrics['RMSE'] = float(metric_str.split(': ')[1])
                elif 'MAPE' in metric_str:
                    metrics['MAPE'] = float(metric_str.split(': ')[1].rstrip('%'))
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to extract validation metrics: {e}")
            return None

    def plot_actual_vs_predicted(self, filename, val_pred, val_actual, val_dates):
        """
        Plots actual vs predicted values for the validation set and saves the plot.

        Args:
            filename (str): Name of the forecast file.
            val_pred (array-like): Predicted values for the validation set.
            val_actual (array-like): Actual values for the validation set.
            val_dates (array-like): Dates for the validation set.

        Returns:
            str: Path to the saved plot or None if plotting fails.
        """
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(val_dates, val_actual, label='Actual Close', color='blue', linewidth=2)
            plt.plot(val_dates, val_pred, label='Predicted Close', color='orange', linestyle='--', linewidth=2)
            plt.title(f'Actual vs Predicted Close Prices (Validation Set) - {filename}', fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Close Price', fontsize=12)
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

            plot_path = os.path.join(self.evaluation_directory, f"{filename.replace('.csv', '')}_validation_plot.png")
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Validation plot saved to {plot_path}")
            return plot_path
        except Exception as e:
            self.logger.error(f"Failed to generate validation plot for {filename}: {e}")
            plt.close()
            return None

    def evaluate_models(self):
        """
        Evaluates all forecast models by extracting validation metrics, generating plots for the
        validation set, and saving results to a CSV summary.
        """
        self.logger.info("Starting model evaluation...")
        datasets = self.load_data()

        if not datasets:
            self.logger.warning("No datasets to evaluate.")
            return

        summary_results = []
        for filename, forecast_df, actual_df in datasets:
            self.logger.info(f"Evaluating {filename}")
            try:
                # Extract validation metrics
                metrics = self.extract_validation_metrics(forecast_df)
                if not metrics:
                    self.logger.warning(f"No validation metrics available for {filename}, skipping.")
                    continue

                # Recompute validation predictions for plotting
                actual_df.set_index('date', inplace=True)
                weekly_data = actual_df.resample('W-FRI').agg({
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

                # Add lagged features
                weekly_data['close_w-1'] = weekly_data['close'].shift(1)
                weekly_data['close_w-2'] = weekly_data['close'].shift(2)
                weekly_data['close_w-3'] = weekly_data['close'].shift(3)
                weekly_data['RSI_14_w-1'] = weekly_data['RSI_14'].shift(1)
                weekly_data.dropna(inplace=True)

                # Split into train and validation
                train_size = int(len(weekly_data) * 0.8)
                train = weekly_data['close'].iloc[:train_size]
                val = weekly_data['close'].iloc[train_size:]
                exogenous_columns = [
                    'volume', 'open_interest', 'MA_5', 'MA_10', 'MA_20', 'MA_50', 'MA_100',
                    'EMA_12', 'EMA_26', 'EMA_50', 'EMA_100', 'RSI_14', 'MACD', 'MACD_Signal',
                    'daily_returns', 'volatility_20', 'BB_upper', 'BB_lower', 'BB_width',
                    'stochastic_k', 'stochastic_d', 'week_of_year', 'month',
                    'close_w-1', 'close_w-2', 'close_w-3', 'RSI_14_w-1'
                ]
                train_exog = weekly_data[exogenous_columns].iloc[:train_size]
                val_exog = weekly_data[exogenous_columns].iloc[train_size:]
                val_dates = val.index

                # Fit Random Forest model
                model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
                model.fit(train_exog, train)
                val_pred = model.predict(val_exog)

                # Generate plot
                plot_path = self.plot_actual_vs_predicted(filename, val_pred, val, val_dates)

                # Collect results
                result = {
                    'filename': filename,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE': metrics['MAPE'],
                    'plot_path': plot_path if plot_path else 'N/A',
                    'evaluation_date': self.today_date
                }
                summary_results.append(result)
                self.logger.info(f"Evaluation completed for {filename}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE']:.4f}%")
            except Exception as e:
                self.logger.error(f"Failed to evaluate {filename}: {e}")
                continue

        # Save summary to CSV
        if summary_results:
            summary_df = pd.DataFrame(summary_results)
            summary_path = os.path.join(self.evaluation_directory, f'evaluation_summary_{self.today_date}.csv')
            summary_df.to_csv(summary_path, index=False)
            self.logger.info(f"Evaluation summary saved to {summary_path}")
        else:
            self.logger.warning("No evaluation results to save.")

        self.logger.info("Model evaluation completed.")

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.evaluate_models()