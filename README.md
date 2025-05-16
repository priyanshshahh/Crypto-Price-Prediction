
# 🪙 Cryptocurrency Price Prediction & Analysis

A comprehensive pipeline to analyze and predict the price of **Bitcoin**, **Ethereum**, and **Dogecoin** using regression models, time-series forecasting, and clustering. Includes data collection, processing, model training, and visual analytics in both Python and Tableau.

---

## 📂 Project Structure

| File / Notebook                   | Description                                               |
|----------------------------------|-----------------------------------------------------------|
| `cryp_dta_collection.ipynb`      | Collects OHLCV data via CryptoCompare API                |
| `cryp_dta_preprocessing.ipynb`   | Cleans and prepares features for modeling                |
| `crypt_regression_model.ipynb`   | Predicts prices using Ridge, Lasso, RF, SVR, etc.         |
| `crypt_timeseries_model.ipynb`   | Time-Series Forecasting using ARIMA and LSTM             |
| `crypt_clustering_analysis.ipynb`| Groups crypto behavior via KMeans, Agglomerative         |
| `regression_results.json`        | Stores regression model scores                           |
| `time_series_results.json`       | Forecast results (ARIMA and LSTM)                        |
| `clustering_results.json`        | Best clustering results and silhouette scores            |
| `create_dashboard.py`            | (Optional) Generates charts for model comparisons        |

---

## 📈 Best Regression Model Results

| Crypto    | Best Model | R² Score | RMSE     | MAE      |
|-----------|------------|----------|----------|----------|
| Bitcoin   | Ridge      | 0.9827   | 2087.57  | 1493.16  |
| Ethereum  | Lasso      | 0.9685   | 105.35   | 75.30    |
| Dogecoin  | Lasso      | 0.9843   | 0.0132   | 0.0082   |

---

## ⏳ Best Time-Series Forecasting Results

| Crypto    | Best Model | R² Score | RMSE     | MAE      |
|-----------|------------|----------|----------|----------|
| Bitcoin   | LSTM       | 0.9269   | 4309.50  | 3521.21  |
| Ethereum  | LSTM       | 0.9494   | 135.76   | 100.27   |
| Dogecoin  | LSTM       | 0.9779   | 0.0158   | 0.0105   |

---

## 🧩 Clustering Summary

| Crypto    | Best Algorithm  | Clusters | Silhouette Score |
|-----------|-----------------|----------|------------------|
| Bitcoin   | KMeans          | 4        | 0.2168           |
| Ethereum  | KMeans          | 3        | 0.2090           |
| Dogecoin  | Agglomerative   | 2        | **0.7283**       |

---

## 📊 Visual Dashboard

Explore the **interactive Tableau dashboard** for insights:
👉 [View Dashboard]([https://us-east-1.online.tableau.com/t/priyanshshah-7ae2fd725b/views/Dasboard/Dashboard1/90db5](https://us-east-1.online.tableau.com/t/priyanshshah-7ae2fd725b/views/Dasboard/Dashboard1))

---

## 📌 Recommended Visual Charts

You can generate these in Python or Tableau for added insights:
- 📊 Bar chart: Top regression model R² by coin
- ⏳ Line plot: LSTM forecast vs actual prices
- 🔍 Feature importance chart from RF/GBR
- 🧩 Clustering silhouette comparison per coin

Notebook or Python script like `create_dashboard.py` can automate these visualizations.

---

## ⚙️ How to Run

```bash
git clone https://github.com/priyanshshahh/crypto-price-prediction.git
cd crypto-price-prediction
pip install -r requirements.txt
````

Run notebooks in order:

1. `cryp_dta_collection.ipynb`
2. `cryp_dta_preprocessing.ipynb`
3. `crypt_regression_model.ipynb`
4. `crypt_clustering_analysis.ipynb`
5. `crypt_timeseries_model.ipynb`

---

## 📦 Dependencies

Install using:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
tensorflow
keras
jupyter
requests
```

*(Already included in `requirements.txt`)*

---

## 🔮 Future Additions

* Social sentiment integration (Reddit/Twitter)
* Live crypto price streaming via WebSockets
* Streamlit-based dashboard for model deployment

---

## 👨‍💻 Author

**Priyansh Shah**
🎓 Stony Brook University | B.S. Applied Mathematics and Statistics
📧 [priyansh.shah@stonybrook.edu](mailto:priyansh.shah@stonybrook.edu)
🔗 [LinkedIn](https://linkedin.com/in/priyansh-shah)

---

## 📜 License

For academic and educational use only. No financial advice.
