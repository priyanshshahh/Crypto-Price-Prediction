import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from datetime import datetime, timedelta
import joblib

# Create directory for dashboard
os.makedirs('dashboard', exist_ok=True)

# Define the cryptocurrencies to analyze
cryptocurrencies = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum', 
    'DOGE-USD': 'Dogecoin'
}

# Function to load data
def load_data():
    """
    Load all the necessary data for the dashboard.
    """
    data = {}
    
    # Load original data
    for symbol, name in cryptocurrencies.items():
        file_path = f'data/{symbol.replace("-", "_")}.csv'
        # Skip the first 3 rows which contain metadata
        df = pd.read_csv(file_path, skiprows=3)
        
        # The first column is the date
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        
        # The second column is the price/close
        df.rename(columns={df.columns[1]: 'Close'}, inplace=True)
        
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Convert numeric columns to float
        for col in df.columns:
            if col not in ['Symbol', 'Name']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Print column names for debugging
        print(f"Columns for {name}: {df.columns.tolist()}")
        
        data[f'{name}_original'] = df
    
    # Load preprocessed data
    for symbol, name in cryptocurrencies.items():
        try:
            file_path = f'data/preprocessed/{symbol.replace("-", "_")}_preprocessed.csv'
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            data[f'{name}_preprocessed'] = df
        except:
            print(f"Preprocessed data for {name} not found.")
    
    # Load clustering results
    for symbol, name in cryptocurrencies.items():
        try:
            file_path = f'data/preprocessed/{name.replace(" ", "_")}_clustering.csv'
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            data[f'{name}_clustering'] = df
        except:
            print(f"Clustering data for {name} not found.")
    
    # Load regression results
    try:
        with open('models/regression_results.json', 'r') as f:
            data['regression_results'] = json.load(f)
    except:
        print("Regression results not found.")
    
    # Load clustering results
    try:
        with open('models/clustering_results.json', 'r') as f:
            data['clustering_results'] = json.load(f)
    except:
        print("Clustering results not found.")
    
    # Load time series results
    try:
        with open('models/time_series_results.json', 'r') as f:
            data['time_series_results'] = json.load(f)
    except:
        print("Time series results not found.")
    
    return data

# Function to create price history visualization
def create_price_history_visualization(data):
    """
    Create interactive price history visualization for all cryptocurrencies.
    """
    # Create figure
    fig = go.Figure()
    
    # Add traces for each cryptocurrency
    for symbol, name in cryptocurrencies.items():
        df = data[f'{name}_original']
        
        # Add line chart for Close price
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name=f'{name} Close',
                line=dict(width=2)
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Cryptocurrency Price History',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=600,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    # Save the figure
    fig.write_html('dashboard/price_history.html')
    
    return fig

# Function to create normalized price comparison
def create_normalized_price_comparison(data):
    """
    Create normalized price comparison visualization.
    """
    # Create figure
    fig = go.Figure()
    
    # Add traces for each cryptocurrency
    for symbol, name in cryptocurrencies.items():
        df = data[f'{name}_original']
        
        # Normalize prices to start at 100
        normalized_price = df['Close'] / df['Close'].iloc[0] * 100
        
        # Add line chart
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=normalized_price,
                mode='lines',
                name=name,
                line=dict(width=2)
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Normalized Price Comparison (Base 100)',
        xaxis_title='Date',
        yaxis_title='Normalized Price',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    # Save the figure
    fig.write_html('dashboard/normalized_price_comparison.html')
    
    return fig

# Function to create volume comparison
def create_volume_comparison(data):
    """
    Create volume comparison visualization.
    """
    # Create figure
    fig = go.Figure()
    
    # Add traces for each cryptocurrency
    for symbol, name in cryptocurrencies.items():
        df = data[f'{name}_original']
        
        # Check if Volume column exists
        if 'Volume' in df.columns:
            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name=name,
                    opacity=0.7
                )
            )
    
    # Update layout
    fig.update_layout(
        title='Trading Volume Comparison',
        xaxis_title='Date',
        yaxis_title='Volume',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis_type="log"  # Log scale for better visualization
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    # Save the figure
    fig.write_html('dashboard/volume_comparison.html')
    
    return fig

# Function to create correlation heatmap
def create_correlation_heatmap(data):
    """
    Create correlation heatmap between cryptocurrencies.
    """
    # Extract closing prices
    close_prices = pd.DataFrame()
    
    for symbol, name in cryptocurrencies.items():
        df = data[f'{name}_original']
        close_prices[name] = df['Close']
    
    # Calculate correlation
    correlation = close_prices.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation.values,
        x=correlation.columns,
        y=correlation.index,
        colorscale='Viridis',
        zmin=-1,
        zmax=1,
        text=correlation.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 14}
    ))
    
    # Update layout
    fig.update_layout(
        title='Cryptocurrency Price Correlation',
        height=500,
        template='plotly_white'
    )
    
    # Save the figure
    fig.write_html('dashboard/correlation_heatmap.html')
    
    return fig

# Function to create regression model comparison
def create_regression_model_comparison(data):
    """
    Create regression model comparison visualization.
    """
    if 'regression_results' not in data:
        print("Regression results not found. Skipping regression model comparison.")
        return None
    
    regression_results = data['regression_results']
    
    # Extract R² scores for each cryptocurrency and model
    cryptos = []
    models = []
    r2_scores = []
    
    for crypto, result in regression_results.items():
        for model, metrics in result['Model Results'].items():
            cryptos.append(crypto)
            models.append(model)
            r2_scores.append(metrics['Test R²'])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Cryptocurrency': cryptos,
        'Model': models,
        'R² Score': r2_scores
    })
    
    # Create figure
    fig = px.bar(
        df,
        x='Model',
        y='R² Score',
        color='Cryptocurrency',
        barmode='group',
        title='Regression Model Performance Comparison',
        height=500,
        template='plotly_white'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Model',
        yaxis_title='R² Score',
        legend_title='Cryptocurrency',
        yaxis=dict(range=[0, 1])
    )
    
    # Save the figure
    fig.write_html('dashboard/regression_model_comparison.html')
    
    return fig

# Function to create clustering visualization
def create_clustering_visualization(data):
    """
    Create clustering visualization.
    """
    # Create a figure for each cryptocurrency
    for symbol, name in cryptocurrencies.items():
        if f'{name}_clustering' not in data:
            print(f"Clustering data for {name} not found. Skipping clustering visualization.")
            continue
        
        df = data[f'{name}_clustering']
        
        # Check if the clustering data has the expected columns
        if 'Best_Cluster' not in df.columns:
            print(f"Best_Cluster column not found in clustering data for {name}. Skipping clustering visualization.")
            continue
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['Best_Cluster'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Cluster')
                ),
                text=df.index.strftime('%Y-%m-%d'),
                name='Clusters'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f'{name} - Price Clusters Over Time',
            xaxis_title='Date',
            yaxis_title='Price',
            height=500,
            template='plotly_white'
        )
        
        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        # Save the figure
        fig.write_html(f'dashboard/{name.replace(" ", "_")}_clustering.html')
    
    return None

# Function to create time series forecast visualization
def create_time_series_forecast_visualization(data):
    """
    Create time series forecast visualization.
    """
    if 'time_series_results' not in data:
        print("Time series results not found. Skipping time series forecast visualization.")
        return None
    
    time_series_results = data['time_series_results']
    
    # Create a figure for each cryptocurrency
    for crypto, result in time_series_results.items():
        # Get the original data
        df = data[f'{crypto}_original']
        
        # Get the forecasts
        arima_forecast = result['ARIMA']['Forecast']
        lstm_forecast = result['LSTM']['Forecast']
        
        # Create date range for forecasts
        last_date = df.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(arima_forecast))
        
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(
            go.Scatter(
                x=df.index[-90:],
                y=df['Close'][-90:],
                mode='lines',
                name='Historical Data',
                line=dict(width=2)
            )
        )
        
        # Add ARIMA forecast
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=arima_forecast,
                mode='lines',
                name='ARIMA Forecast',
                line=dict(width=2, dash='dash', color='red')
            )
        )
        
        # Add LSTM forecast
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=lstm_forecast,
                mode='lines',
                name='LSTM Forecast',
                line=dict(width=2, dash='dash', color='green')
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f'{crypto} - 30-Day Price Forecast',
            xaxis_title='Date',
            yaxis_title='Price',
            height=500,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Save the figure
        fig.write_html(f'dashboard/{crypto.replace(" ", "_")}_forecast.html')
    
    return None

# Function to create model performance comparison
def create_model_performance_comparison(data):
    """
    Create model performance comparison visualization.
    """
    if 'regression_results' not in data or 'time_series_results' not in data:
        print("Results not found. Skipping model performance comparison.")
        return None
    
    regression_results = data['regression_results']
    time_series_results = data['time_series_results']
    
    # Extract best model R² scores for each cryptocurrency
    cryptos = []
    regression_r2 = []
    arima_r2 = []
    lstm_r2 = []
    
    for crypto in cryptocurrencies.values():
        if crypto in regression_results and crypto in time_series_results:
            cryptos.append(crypto)
            regression_r2.append(regression_results[crypto]['Best Model R²'])
            arima_r2.append(time_series_results[crypto]['ARIMA']['R²'])
            lstm_r2.append(time_series_results[crypto]['LSTM']['R²'])
    
    # Create figure
    fig = go.Figure()
    
    # Add bar charts
    fig.add_trace(
        go.Bar(
            x=cryptos,
            y=regression_r2,
            name='Regression',
            text=[f'{r:.4f}' for r in regression_r2],
            textposition='auto'
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=cryptos,
            y=arima_r2,
            name='ARIMA',
            text=[f'{r:.4f}' for r in arima_r2],
            textposition='auto'
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=cryptos,
            y=lstm_r2,
            name='LSTM',
            text=[f'{r:.4f}' for r in lstm_r2],
            textposition='auto'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Model Performance Comparison (R² Score)',
        xaxis_title='Cryptocurrency',
        yaxis_title='R² Score',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(range=[0, 1])
    )
    
    # Save the figure
    fig.write_html('dashboard/model_performance_comparison.html')
    
    return fig

# Function to create feature importance visualization
def create_feature_importance_visualization(data):
    """
    Create feature importance visualization.
    """
    # Create a figure for each cryptocurrency
    for symbol, name in cryptocurrencies.items():
        if f'{name}_preprocessed' not in data:
            print(f"Preprocessed data for {name} not found. Skipping feature importance visualization.")
            continue
            
        df = data[f'{name}_preprocessed']
        
        # Calculate correlation with target
        if 'Target_Next_Day' not in df.columns:
            print(f"Target_Next_Day column not found in preprocessed data for {name}. Skipping feature importance visualization.")
            continue
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation with target
        target_correlation = numeric_df.corr()['Target_Next_Day'].sort_values(ascending=False)
        
        # Remove target variables from the correlation
        target_correlation = target_correlation.drop(['Target_Next_Day', 'Target_Next_Week', 'Target_Next_Month'], errors='ignore')
        
        # Create figure
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=target_correlation.index[:15],
                y=target_correlation.values[:15],
                text=[f'{r:.4f}' for r in target_correlation.values[:15]],
                textposition='auto'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f'{name} - Top 15 Features Correlation with Next Day Price',
            xaxis_title='Feature',
            yaxis_title='Correlation',
            height=500,
            template='plotly_white'
        )
        
        # Save the figure
        fig.write_html(f'dashboard/{name.replace(" ", "_")}_feature_importance.html')
    
    return None

# Function to create technical indicators visualization
def create_technical_indicators_visualization(data):
    """
    Create technical indicators visualization.
    """
    # Create a figure for each cryptocurrency
    for symbol, name in cryptocurrencies.items():
        if f'{name}_preprocessed' not in data:
            print(f"Preprocessed data for {name} not found. Skipping technical indicators visualization.")
            continue
            
        df = data[f'{name}_preprocessed']
        
        # Check if the preprocessed data has the expected columns
        required_columns = ['Close', 'MA7', 'MA30', 'RSI', 'MACD']
        if not all(col in df.columns for col in required_columns):
            print(f"Required columns not found in preprocessed data for {name}. Skipping technical indicators visualization.")
            continue
        
        # Create figure with subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price and Moving Averages', 'RSI', 'MACD', 'Volume')
        )
        
        # Add price and moving averages
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Close',
                line=dict(width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MA7'],
                mode='lines',
                name='MA7',
                line=dict(width=1.5)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MA30'],
                mode='lines',
                name='MA30',
                line=dict(width=1.5)
            ),
            row=1, col=1
        )
        
        # Add RSI
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(width=1.5, color='purple')
            ),
            row=2, col=1
        )
        
        # Add RSI overbought/oversold lines
        fig.add_trace(
            go.Scatter(
                x=[df.index[0], df.index[-1]],
                y=[70, 70],
                mode='lines',
                name='Overbought',
                line=dict(width=1, color='red', dash='dash')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[df.index[0], df.index[-1]],
                y=[30, 30],
                mode='lines',
                name='Oversold',
                line=dict(width=1, color='green', dash='dash')
            ),
            row=2, col=1
        )
        
        # Add MACD
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                mode='lines',
                name='MACD',
                line=dict(width=1.5, color='blue')
            ),
            row=3, col=1
        )
        
        if 'MACD_Signal' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD_Signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(width=1.5, color='red')
                ),
                row=3, col=1
            )
        
        if 'MACD_Hist' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['MACD_Hist'],
                    name='Histogram',
                    marker_color='green'
                ),
                row=3, col=1
            )
        
        # Add Volume if available
        if 'Volume' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color='lightblue'
                ),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{name} - Technical Indicators',
            height=800,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            row=4, col=1
        )
        
        # Save the figure
        fig.write_html(f'dashboard/{name.replace(" ", "_")}_technical_indicators.html')
    
    return None

# Function to create main dashboard HTML
def create_dashboard_html():
    """
    Create the main dashboard HTML file.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cryptocurrency Price Prediction Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f8f9fa;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                background-color: #343a40;
                color: white;
                padding: 20px 0;
                text-align: center;
                margin-bottom: 30px;
            }
            .card {
                margin-bottom: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .card-header {
                background-color: #007bff;
                color: white;
                font-weight: bold;
                border-radius: 10px 10px 0 0;
            }
            .nav-tabs {
                margin-bottom: 15px;
            }
            iframe {
                width: 100%;
                border: none;
                height: 500px;
            }
            .summary-card {
                height: 100%;
            }
            .model-metrics {
                font-size: 1.1em;
                margin-bottom: 15px;
            }
            .metric-value {
                font-weight: bold;
                color: #007bff;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="container">
                <h1>Cryptocurrency Price Prediction Dashboard</h1>
                <p>Interactive visualization and analysis of Bitcoin, Ethereum, and Dogecoin</p>
            </div>
        </div>
        
        <div class="container">
            <!-- Overview Section -->
            <div class="card">
                <div class="card-header">
                    Market Overview
                </div>
                <div class="card-body">
                    <ul class="nav nav-tabs" id="marketTabs" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="price-history-tab" data-bs-toggle="tab" data-bs-target="#price-history" type="button" role="tab" aria-controls="price-history" aria-selected="true">Price History</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="normalized-tab" data-bs-toggle="tab" data-bs-target="#normalized" type="button" role="tab" aria-controls="normalized" aria-selected="false">Normalized Comparison</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="volume-tab" data-bs-toggle="tab" data-bs-target="#volume" type="button" role="tab" aria-controls="volume" aria-selected="false">Volume</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="correlation-tab" data-bs-toggle="tab" data-bs-target="#correlation" type="button" role="tab" aria-controls="correlation" aria-selected="false">Correlation</button>
                        </li>
                    </ul>
                    <div class="tab-content" id="marketTabsContent">
                        <div class="tab-pane fade show active" id="price-history" role="tabpanel" aria-labelledby="price-history-tab">
                            <iframe src="price_history.html"></iframe>
                        </div>
                        <div class="tab-pane fade" id="normalized" role="tabpanel" aria-labelledby="normalized-tab">
                            <iframe src="normalized_price_comparison.html"></iframe>
                        </div>
                        <div class="tab-pane fade" id="volume" role="tabpanel" aria-labelledby="volume-tab">
                            <iframe src="volume_comparison.html"></iframe>
                        </div>
                        <div class="tab-pane fade" id="correlation" role="tabpanel" aria-labelledby="correlation-tab">
                            <iframe src="correlation_heatmap.html"></iframe>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Model Performance Section -->
            <div class="card">
                <div class="card-header">
                    Model Performance
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-8">
                            <iframe src="model_performance_comparison.html"></iframe>
                        </div>
                        <div class="col-md-4">
                            <div class="card summary-card">
                                <div class="card-body">
                                    <h5 class="card-title">Model Summary</h5>
                                    <div class="model-metrics">
                                        <p><strong>Bitcoin:</strong> <span class="metric-value">98.45%</span> accuracy (ElasticNet)</p>
                                        <p><strong>Ethereum:</strong> <span class="metric-value">94.19%</span> accuracy (ElasticNet)</p>
                                        <p><strong>Dogecoin:</strong> <span class="metric-value">98.00%</span> accuracy (Linear Regression)</p>
                                    </div>
                                    <p>The regression models achieved high accuracy in predicting cryptocurrency prices, with R² scores above 94% for all three cryptocurrencies. The time series models (ARIMA and LSTM) provide forecasting capabilities for future price movements.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Bitcoin Analysis Section -->
            <div class="card">
                <div class="card-header">
                    Bitcoin Analysis
                </div>
                <div class="card-body">
                    <ul class="nav nav-tabs" id="bitcoinTabs" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="btc-forecast-tab" data-bs-toggle="tab" data-bs-target="#btc-forecast" type="button" role="tab" aria-controls="btc-forecast" aria-selected="true">Price Forecast</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="btc-clustering-tab" data-bs-toggle="tab" data-bs-target="#btc-clustering" type="button" role="tab" aria-controls="btc-clustering" aria-selected="false">Market Regimes</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="btc-features-tab" data-bs-toggle="tab" data-bs-target="#btc-features" type="button" role="tab" aria-controls="btc-features" aria-selected="false">Feature Importance</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="btc-technical-tab" data-bs-toggle="tab" data-bs-target="#btc-technical" type="button" role="tab" aria-controls="btc-technical" aria-selected="false">Technical Indicators</button>
                        </li>
                    </ul>
                    <div class="tab-content" id="bitcoinTabsContent">
                        <div class="tab-pane fade show active" id="btc-forecast" role="tabpanel" aria-labelledby="btc-forecast-tab">
                            <iframe src="Bitcoin_forecast.html"></iframe>
                        </div>
                        <div class="tab-pane fade" id="btc-clustering" role="tabpanel" aria-labelledby="btc-clustering-tab">
                            <iframe src="Bitcoin_clustering.html"></iframe>
                        </div>
                        <div class="tab-pane fade" id="btc-features" role="tabpanel" aria-labelledby="btc-features-tab">
                            <iframe src="Bitcoin_feature_importance.html"></iframe>
                        </div>
                        <div class="tab-pane fade" id="btc-technical" role="tabpanel" aria-labelledby="btc-technical-tab">
                            <iframe src="Bitcoin_technical_indicators.html"></iframe>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Ethereum Analysis Section -->
            <div class="card">
                <div class="card-header">
                    Ethereum Analysis
                </div>
                <div class="card-body">
                    <ul class="nav nav-tabs" id="ethereumTabs" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="eth-forecast-tab" data-bs-toggle="tab" data-bs-target="#eth-forecast" type="button" role="tab" aria-controls="eth-forecast" aria-selected="true">Price Forecast</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="eth-clustering-tab" data-bs-toggle="tab" data-bs-target="#eth-clustering" type="button" role="tab" aria-controls="eth-clustering" aria-selected="false">Market Regimes</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="eth-features-tab" data-bs-toggle="tab" data-bs-target="#eth-features" type="button" role="tab" aria-controls="eth-features" aria-selected="false">Feature Importance</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="eth-technical-tab" data-bs-toggle="tab" data-bs-target="#eth-technical" type="button" role="tab" aria-controls="eth-technical" aria-selected="false">Technical Indicators</button>
                        </li>
                    </ul>
                    <div class="tab-content" id="ethereumTabsContent">
                        <div class="tab-pane fade show active" id="eth-forecast" role="tabpanel" aria-labelledby="eth-forecast-tab">
                            <iframe src="Ethereum_forecast.html"></iframe>
                        </div>
                        <div class="tab-pane fade" id="eth-clustering" role="tabpanel" aria-labelledby="eth-clustering-tab">
                            <iframe src="Ethereum_clustering.html"></iframe>
                        </div>
                        <div class="tab-pane fade" id="eth-features" role="tabpanel" aria-labelledby="eth-features-tab">
                            <iframe src="Ethereum_feature_importance.html"></iframe>
                        </div>
                        <div class="tab-pane fade" id="eth-technical" role="tabpanel" aria-labelledby="eth-technical-tab">
                            <iframe src="Ethereum_technical_indicators.html"></iframe>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Dogecoin Analysis Section -->
            <div class="card">
                <div class="card-header">
                    Dogecoin Analysis
                </div>
                <div class="card-body">
                    <ul class="nav nav-tabs" id="dogecoinTabs" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="doge-forecast-tab" data-bs-toggle="tab" data-bs-target="#doge-forecast" type="button" role="tab" aria-controls="doge-forecast" aria-selected="true">Price Forecast</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="doge-clustering-tab" data-bs-toggle="tab" data-bs-target="#doge-clustering" type="button" role="tab" aria-controls="doge-clustering" aria-selected="false">Market Regimes</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="doge-features-tab" data-bs-toggle="tab" data-bs-target="#doge-features" type="button" role="tab" aria-controls="doge-features" aria-selected="false">Feature Importance</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="doge-technical-tab" data-bs-toggle="tab" data-bs-target="#doge-technical" type="button" role="tab" aria-controls="doge-technical" aria-selected="false">Technical Indicators</button>
                        </li>
                    </ul>
                    <div class="tab-content" id="dogecoinTabsContent">
                        <div class="tab-pane fade show active" id="doge-forecast" role="tabpanel" aria-labelledby="doge-forecast-tab">
                            <iframe src="Dogecoin_forecast.html"></iframe>
                        </div>
                        <div class="tab-pane fade" id="doge-clustering" role="tabpanel" aria-labelledby="doge-clustering-tab">
                            <iframe src="Dogecoin_clustering.html"></iframe>
                        </div>
                        <div class="tab-pane fade" id="doge-features" role="tabpanel" aria-labelledby="doge-features-tab">
                            <iframe src="Dogecoin_feature_importance.html"></iframe>
                        </div>
                        <div class="tab-pane fade" id="doge-technical" role="tabpanel" aria-labelledby="doge-technical-tab">
                            <iframe src="Dogecoin_technical_indicators.html"></iframe>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Footer -->
            <div class="card">
                <div class="card-body text-center">
                    <h5>Cryptocurrency Price Prediction Project</h5>
                    <p>Built with Python, Regression, Time-Series Forecasting, and Machine Learning</p>
                    <p>© 2025</p>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    # Save the HTML file
    with open('dashboard/index.html', 'w') as f:
        f.write(html_content)
    
    return html_content

# Main function to create the dashboard
def create_dashboard():
    """
    Create the complete visualization dashboard.
    """
    print("Loading data...")
    data = load_data()
    
    print("Creating price history visualization...")
    create_price_history_visualization(data)
    
    print("Creating normalized price comparison...")
    create_normalized_price_comparison(data)
    
    print("Creating volume comparison...")
    create_volume_comparison(data)
    
    print("Creating correlation heatmap...")
    create_correlation_heatmap(data)
    
    print("Creating regression model comparison...")
    create_regression_model_comparison(data)
    
    print("Creating clustering visualization...")
    create_clustering_visualization(data)
    
    print("Creating time series forecast visualization...")
    create_time_series_forecast_visualization(data)
    
    print("Creating model performance comparison...")
    create_model_performance_comparison(data)
    
    print("Creating feature importance visualization...")
    create_feature_importance_visualization(data)
    
    print("Creating technical indicators visualization...")
    create_technical_indicators_visualization(data)
    
    print("Creating main dashboard HTML...")
    create_dashboard_html()
    
    print("Dashboard created successfully!")

# Run the dashboard creation
if __name__ == "__main__":
    create_dashboard()
