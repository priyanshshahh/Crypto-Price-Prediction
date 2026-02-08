# Cryptocurrency Price Prediction - Web Application

A live, interactive web application for cryptocurrency price prediction and analysis using machine learning.

## Features

- **Real-time Price Tracking**: Monitor Bitcoin, Ethereum, and Dogecoin prices with 90-day historical data
- **Model Performance Dashboard**: Compare regression models (Ridge, Lasso, ElasticNet, SVR, Random Forest, Gradient Boosting)
- **Price Forecasting**: View 30-day predictions using ARIMA and LSTM models
- **Clustering Analysis**: Identify market regimes using KMeans, DBSCAN, Agglomerative, and GMM algorithms
- **Interactive Charts**: Built with Recharts for responsive, beautiful visualizations
- **Real-time Data**: Powered by Supabase for instant data synchronization

## Technology Stack

- **Frontend**: React + Vite
- **Styling**: Tailwind CSS v4
- **Charts**: Recharts
- **Database**: Supabase (PostgreSQL)
- **Icons**: Lucide React
- **Deployment**: Ready for Vercel, Netlify, or any static host

## Getting Started

The application is already built and ready to use. The database has been seeded with sample data.

### Running Locally

```bash
npm run dev
```

Visit `http://localhost:5173` to view the application.

### Building for Production

```bash
npm run build
```

The production build will be in the `dist/` directory.

## Project Structure

```
src/
├── components/
│   ├── Dashboard.jsx          # Main dashboard container
│   ├── PriceChart.jsx         # Historical price visualization
│   ├── ModelPerformance.jsx   # ML model comparison
│   ├── ForecastView.jsx       # Future price predictions
│   └── ClusteringView.jsx     # Market regime analysis
├── lib/
│   └── supabase.js            # Supabase client configuration
├── App.jsx                    # Root component
├── main.jsx                   # Application entry point
└── index.css                  # Global styles
```

## Database Schema

The application uses the following tables:

- `cryptocurrencies`: Crypto metadata (BTC, ETH, DOGE)
- `price_history`: Historical OHLCV data
- `regression_models`: ML model performance metrics
- `forecasts`: Future price predictions (ARIMA, LSTM)
- `clustering_results`: Market regime clustering data

## Key Features Explained

### Overview Tab
- 90-day price history with interactive area charts
- High-Low range visualization
- Key metrics: Current price, 90-day high/low, average volume

### Models Tab
- Performance comparison across 7 regression models
- R² score, RMSE, and MAE metrics
- Best model highlighting
- Detailed model comparison table

### Forecast Tab
- 30-day price predictions
- Toggle between ARIMA and LSTM models
- Expected price change calculation
- Side-by-side forecast comparison

### Clustering Tab
- Market regime identification
- Silhouette score comparison
- Cluster distribution visualization
- Algorithm performance metrics

## Model Performance

Based on the original research:

| Crypto    | Best Model | R² Score | RMSE     | MAE      |
|-----------|------------|----------|----------|----------|
| Bitcoin   | Ridge      | 0.9827   | 2087.57  | 1493.16  |
| Ethereum  | Lasso      | 0.9685   | 105.35   | 75.30    |
| Dogecoin  | Lasso      | 0.9843   | 0.0132   | 0.0082   |

## Data Source

The application is seeded with synthetic data for demonstration purposes. To use real data:

1. Implement the CryptoCompare API integration from the original notebooks
2. Run the data collection and preprocessing pipeline
3. Update the database with actual historical data

## License

MIT License - See LICENSE file for details

## Original Research

This web application is based on the cryptocurrency prediction research project that includes:
- Data collection via CryptoCompare API
- Feature engineering and preprocessing
- Multiple ML models (regression, time-series, clustering)
- Comprehensive analysis and visualization

For the full research implementation, see the Jupyter notebooks in the project root.
