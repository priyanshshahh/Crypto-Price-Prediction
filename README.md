# Crypto-Price-Prediction
This project implements a comprehensive cryptocurrency price prediction system using Python, Regression, Time-Series Forecasting, and Machine Learning techniques. The system analyzes Bitcoin, Ethereum, and Dogecoin price data to provide accurate predictions and market insights.

## Project Structure

```
crypto_prediction/
├── data/                      # Raw cryptocurrency data
├── data/preprocessed/         # Preprocessed data files
├── models/                    # Saved models and results
├── visualizations/            # Generated visualizations
├── dashboard/                 # Interactive dashboard files
├── data_collection.py         # Data collection script
├── data_preprocessing.py      # Data preprocessing script
├── regression_models.py       # Regression model implementation
├── clustering_analysis.py     # Clustering analysis implementation
├── time_series_models.py      # ARIMA and LSTM model implementation
├── create_dashboard.py        # Dashboard creation script
└── README.md                  # Project documentation
```

## Key Features

1. **Data Collection**: Automated collection of historical cryptocurrency data using Yahoo Finance API
2. **Data Preprocessing**: Feature engineering including technical indicators (MA, RSI, MACD, Bollinger Bands)
3. **Regression Models**: Multiple regression algorithms with hyperparameter tuning
4. **Clustering Analysis**: Market regime identification using unsupervised learning
5. **Time Series Forecasting**: ARIMA and LSTM models for price prediction
6. **Interactive Dashboard**: Comprehensive visualization of all analyses and predictions

## Model Performance

### Regression Models (R² Score)
- **Bitcoin**: 98.45% accuracy (ElasticNet)
- **Ethereum**: 94.19% accuracy (ElasticNet)
- **Dogecoin**: 98.00% accuracy (Linear Regression)

### Time Series Models (R² Score)
- **Bitcoin**: 96.39% accuracy (LSTM)
- **Ethereum**: 81.31% accuracy (LSTM)
- **Dogecoin**: 96.42% accuracy (LSTM)

## Technical Implementation Details

### Data Collection
The system collects historical price data for Bitcoin, Ethereum, and Dogecoin using the Yahoo Finance API. The data includes daily OHLCV (Open, High, Low, Close, Volume) values.

### Data Preprocessing
- **Feature Engineering**: Creation of technical indicators (Moving Averages, RSI, MACD, Bollinger Bands)
- **Target Variables**: Generation of next-day, next-week, and next-month price targets
- **Data Normalization**: Scaling features for model training
- **Missing Value Handling**: Interpolation and forward filling

### Regression Models
Multiple regression algorithms are implemented and evaluated:
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest
- Gradient Boosting
- Support Vector Regression (SVR)

### Clustering Analysis
Market regimes are identified using:
- K-Means Clustering
- Gaussian Mixture Models
- Agglomerative Clustering

### Time Series Forecasting
Two approaches are implemented:
- **ARIMA**: Statistical time series modeling with parameter optimization
- **LSTM**: Deep learning sequence prediction with TensorFlow

### Visualization Dashboard
An interactive dashboard is created with:
- Price history visualizations
- Model performance comparisons
- Technical indicator charts
- Clustering visualizations
- Price forecasts
- Tableau Dashboard link:https://us-east-1.online.tableau.com/t/priyanshshah-7ae2fd725b/views/Dasboard/Dashboard1/90db5220-4278-4134-bf7d-4a3b677e2bf2/36580626-190f-4ab6-9b66-39ee13f69e70

## Running the Project

1. Install required packages:
```
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels tensorflow yfinance plotly
```

2. Run the data collection script:
```
python data_collection.py
```

3. Run the data preprocessing script:
```
python data_preprocessing.py
```

4. Train and evaluate the regression models:
```
python regression_models.py
```

5. Perform clustering analysis:
```
python clustering_analysis.py
```

6. Train and evaluate the time series models:
```
python time_series_models.py
```

7. Create the visualization dashboard:
```
python create_dashboard.py
```

8. Open the dashboard:
```
open dashboard/index.html
```

## Future Enhancements

1. Real-time data integration for live predictions
2. Sentiment analysis from social media and news
3. Portfolio optimization based on predictions
4. Reinforcement learning for trading strategies
5. Expanded cryptocurrency coverage

## Conclusion

This cryptocurrency price prediction system demonstrates the power of combining multiple machine learning approaches for financial forecasting. The high accuracy achieved (94%+ R² scores) shows the effectiveness of the implemented models and features.
