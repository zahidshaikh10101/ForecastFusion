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

🛠 Usage
1. Configure Upstox API credentials in main.py or a configuration file.

2. Run the pipeline:
python main.py

3. Customize the pipeline:
  -  Modify stock symbols in the DataFetcher class.
  - Adjust feature engineering logic in FeatureEngineer.
  - Tweak hyperparameters or forecast horizon in ModelGenerator.
  - Customize evaluation logic in ModelEvaluator.

