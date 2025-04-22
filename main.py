import logging
from src.data_fetcher import DataFetcher
from src.data_preprocessor import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_generator import ModelGenerator
from src.model_evaluation import ModelEvaluator
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__, log_file='logs/main.log')

# Initialize module instances
data_fetcher = DataFetcher()
preprocessor = DataPreprocessor()
engineer = FeatureEngineer()
model_generator = ModelGenerator()
evaluator = ModelEvaluator()

# Execute Workflow
logger.info("Fetching stock data from Upstox...")
data_fetcher.fetch_all()

logger.info("Preprocessing fetched data...")
preprocessor.process_raw()

logger.info("Performing feature engineering...")
engineer.engineer_features()

logger.info("Generating forecast for next 52 weeks...")
model_generator.fit(weeks=52)

logger.info("Evaluating model performance...")
evaluator.evaluate_models()

logger.info("Workflow completed successfully.")