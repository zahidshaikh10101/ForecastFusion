# 📈 Indian Stock Price Forecasting Pipeline

This project is a comprehensive machine learning pipeline for fetching, preprocessing, engineering features, forecasting, and evaluating stock prices for Indian stocks. It leverages the **Upstox API** to fetch historical stock data and processes it through a modular workflow to predict future stock prices using **Random Forest Regression**.

---

## 📋 Table of Contents

- [Overview](#-overview)  
- [Features](#-features)  
- [Directory Structure](#-directory-structure)  
- [Installation](#-installation)  
- [Usage](#-usage)  
- [Modules](#-modules)  
- [Pending Work](#-pending-work)  
- [Future Enhancements](#-future-enhancements)  
- [License](#-license)  

---

## 🌟 Overview

This project fetches historical Indian stock data from the Upstox API, preprocesses it, engineers advanced features, and uses **Random Forest Regression** to forecast stock prices for the next 52 weeks. It includes model evaluation with metrics and visualizations, ensuring a robust end-to-end workflow. Each module generates logs for debugging and monitoring.

---

## ✨ Features

- 📊 Fetches historical Indian stock data using the **Upstox API**.
- 🔄 Converts raw JSON data into structured CSV files with handling for missing values.
- ⚙️ Engineers advanced features like **moving averages**, **EMA**, **RSI**, **MACD**, **Bollinger Bands**, and more.
- 📉 Forecasts stock prices for **52 weeks** using **Random Forest Regression** with hyperparameter tuning.
- 📈 Evaluates model performance with metrics (**MAE**, **RMSE**, **MAPE**) and visualizes actual vs. predicted prices.
- 🔌 Modular pipeline with **logging** for traceability and extensibility.

---

## 🗂 Directory Structure
indian-stock-forecast-pipeline/
│
├── data/                          # All data files
│   ├── raw/{today_date}/          # Raw JSON data
│   ├── processed/{today_date}/    # Processed CSV data
│   ├── engineered/{today_date}/   # Feature-engineered CSV data
│   ├── forecast/{today_date}/     # Forecast results CSV
│   └── evaluation/{today_date}/   # Evaluation metrics and plots
│
├── src/                           # Source code
│   ├── data_fetcher.py            # DataFetcher module
│   ├── data_preprocessor.py       # DataPreprocessor module
│   ├── feature_engineering.py     # FeatureEngineer module
│   ├── model_generator.py         # ModelGenerator module
│   └── model_evaluation.py        # ModelEvaluator module
│
├── notebooks/                     # Jupyter Notebooks for analysis
│   └── visual_reports.ipynb       # Notebook for visual reports
│
├── main.py                        # Main script to run the workflow
├── README.md                      # Project README
├── requirements.txt               # Python dependencies
└── logs/                          # Log files

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher  
- Upstox API credentials  
- Virtual environment (recommended)

### Steps

```bash
# Clone this repository:
git clone https://github.com/your-username/indian-stock-forecast-pipeline.git
cd indian-stock-forecast-pipeline

# Set up a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt
```
---

## 🛠 Usage

1.  **Configure Upstox API credentials:** Ensure you have your Upstox API credentials configured correctly in either the `main.py` file or a separate configuration file that `main.py` can access.

2.  **Run the main pipeline:**

    ```bash
    # Run the main pipeline:
    python main.py
    ```

3.  **Customize the pipeline:** You can tailor the pipeline to your specific needs by modifying the following:

    * **➕ Add or remove stock symbols:** Edit the `DataFetcher` class (`src/data_fetcher.py`) to include or exclude the specific Indian stock symbols you want to analyze.
    * **⚙️ Adjust feature engineering logic:** Modify the `FeatureEngineer` class (`src/feature_engineering.py`) to implement different or additional technical indicators based on your analysis requirements.
    * **🧠 Modify hyperparameters or forecasting periods:** Fine-tune the hyperparameters of the Random Forest Regression model or adjust the forecasting horizon within the `ModelGenerator` class (`src/model_generator.py`).
    * **📊 Customize evaluation metrics or visualizations:** Adapt the evaluation metrics used or the visualizations generated in the `ModelEvaluator` class (`src/model_evaluation.py`) to suit your preferred analysis output.

---

## 🧩 Modules

The pipeline is organized into the following modules:

1.  **DataFetcher**
    * **Location:** `src/data_fetcher.py`
    * **Purpose:** Fetches historical Indian stock market data from the Upstox API. The fetched raw JSON data is saved in the `data/raw/{today_date}/` directory.
    * **Key Method:**
        ```python
        fetch_stock_data(symbol)
        ```
    * **Logging:** Records fetch operations, including successful retrievals and any errors encountered during the process.

2.  **DataPreprocessor**
    * **Location:** `src/data_preprocessor.py`
    * **Purpose:** Takes the raw JSON data fetched by `DataFetcher` and converts it into a cleaner, structured CSV format. The processed CSV files are saved in `data/processed/{today_date}/`.
    * **Key Method:**
        ```python
        process_json_to_csv()
        ```
    * **Logging:** Details the preprocessing steps undertaken and any actions taken to handle missing values within the data.

3.  **FeatureEngineer**
    * **Location:** `src/feature_engineering.py`
    * **Purpose:** Generates relevant technical indicators from the processed stock data. These engineered features are crucial for the model's learning process and are saved in the `data/engineered/{today_date}/` directory.
    * **Key Method:**
        ```python
        engineer_features()
        ```
    * **Logging:** Tracks the steps involved in the generation of various technical indicators.

4.  **ModelGenerator**
    * **Location:** `src/model_generator.py`
    * **Purpose:** Utilizes the engineered features to train a Random Forest Regression model for forecasting stock prices. The trained models and their predictions are saved in the `data/forecast/{today_date}/` directory.
    * **Key Method:**
        ```python
        train_and_predict(symbol, horizon)
        ```
    * **Logging:** Records the model training process and the generation of price predictions for the specified horizon.

5.  **ModelEvaluator**
    * **Location:** `src/model_evaluation.py`
    * **Purpose:** Assesses the accuracy and performance of the generated forecasting models. It also visualizes the results to provide insights into the model's predictions. The evaluation metrics and visualizations are outputted to the `data/evaluation/{today_date}/` directory.
    * **Key Method:**
        ```python
        evaluate_model()
        ```
    * **Logging:** Details the evaluation metrics calculated and the generation of any performance plots.

---

## 🚧 Pending Work

* **Improve model accuracy:** Focus on enhancing the forecasting model's accuracy through more sophisticated hyperparameter tuning techniques and the exploration of different feature sets.
* **Optimize pipeline execution:** Streamline the pipeline to achieve faster execution times, potentially through code optimization or parallel processing.
* **Automate daily runs:** Implement a mechanism to automate the pipeline's execution on a daily basis for generating real-time stock price predictions.
* **Build a visualization dashboard:** Develop an interactive dashboard to effectively visualize the historical data and the generated stock price forecasts.

---

## 🔮 Future Enhancements

* **Integrate advanced models:** Explore and integrate more advanced time-series forecasting models such as Long Short-Term Memory (LSTM) networks or XGBoost to potentially improve prediction accuracy.
* **Add more technical indicators:** Incorporate a wider range of technical indicators, such as Fibonacci retracement levels and the Ichimoku Cloud, to enrich the feature set.
* **Automate scheduling:** Implement robust scheduling of the pipeline using cron jobs or cloud-based scheduling services.
* **Cloud deployment:** Deploy the entire pipeline on cloud platforms like AWS or GCP for scalability and reliability.
* **Develop a web interface:** Create a user-friendly web interface to allow users to interact with the system, select stocks, and view forecasts.

