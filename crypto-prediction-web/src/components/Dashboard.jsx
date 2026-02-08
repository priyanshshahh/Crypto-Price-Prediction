import { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { TrendingUp, TrendingDown, Activity, BarChart3 } from 'lucide-react';
import PriceChart from './PriceChart';
import ModelPerformance from './ModelPerformance';
import ForecastView from './ForecastView';
import ClusteringView from './ClusteringView';

export default function Dashboard() {
  const [cryptos, setCryptos] = useState([]);
  const [priceData, setPriceData] = useState({});
  const [loading, setLoading] = useState(true);
  const [selectedCrypto, setSelectedCrypto] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const { data: cryptoData, error: cryptoError } = await supabase
        .from('cryptocurrencies')
        .select('*');

      if (cryptoError) throw cryptoError;

      setCryptos(cryptoData);
      if (cryptoData.length > 0 && !selectedCrypto) {
        setSelectedCrypto(cryptoData[0]);
      }

      const priceDataMap = {};
      for (const crypto of cryptoData) {
        const { data: prices, error: priceError } = await supabase
          .from('price_history')
          .select('*')
          .eq('crypto_id', crypto.id)
          .order('date', { ascending: false })
          .limit(90);

        if (!priceError) {
          priceDataMap[crypto.id] = prices.reverse();
        }
      }
      setPriceData(priceDataMap);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getCryptoColor = (symbol) => {
    const colors = {
      'BTC': 'bg-orange-500',
      'ETH': 'bg-blue-500',
      'DOGE': 'bg-yellow-500'
    };
    return colors[symbol] || 'bg-gray-500';
  };

  const getCryptoTextColor = (symbol) => {
    const colors = {
      'BTC': 'text-orange-600',
      'ETH': 'text-blue-600',
      'DOGE': 'text-yellow-600'
    };
    return colors[symbol] || 'text-gray-600';
  };

  const getLatestPrice = (cryptoId) => {
    const prices = priceData[cryptoId];
    if (!prices || prices.length === 0) return null;
    return prices[prices.length - 1];
  };

  const getPriceChange = (cryptoId) => {
    const prices = priceData[cryptoId];
    if (!prices || prices.length < 2) return 0;
    const latest = prices[prices.length - 1];
    const previous = prices[prices.length - 2];
    return ((latest.close - previous.close) / previous.close) * 100;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div className="text-center">
          <Activity className="w-12 h-12 animate-spin mx-auto mb-4 text-blue-600" />
          <p className="text-gray-600">Loading cryptocurrency data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Cryptocurrency Price Prediction
              </h1>
              <p className="mt-1 text-sm text-gray-600">
                ML-powered analysis and forecasting for Bitcoin, Ethereum, and Dogecoin
              </p>
            </div>
            <BarChart3 className="w-12 h-12 text-blue-600" />
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          {cryptos.map((crypto) => {
            const latestPrice = getLatestPrice(crypto.id);
            const priceChange = getPriceChange(crypto.id);
            const isPositive = priceChange >= 0;

            return (
              <div
                key={crypto.id}
                onClick={() => setSelectedCrypto(crypto)}
                className={`bg-white rounded-lg shadow-md p-6 cursor-pointer transition-all hover:shadow-lg ${
                  selectedCrypto?.id === crypto.id ? 'ring-2 ring-blue-500' : ''
                }`}
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className={`w-10 h-10 rounded-full ${getCryptoColor(crypto.symbol)} flex items-center justify-center text-white font-bold`}>
                      {crypto.symbol.substring(0, 2)}
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900">{crypto.name}</h3>
                      <p className="text-sm text-gray-500">{crypto.symbol}</p>
                    </div>
                  </div>
                  {isPositive ? (
                    <TrendingUp className="w-6 h-6 text-green-500" />
                  ) : (
                    <TrendingDown className="w-6 h-6 text-red-500" />
                  )}
                </div>
                {latestPrice && (
                  <div>
                    <p className="text-2xl font-bold text-gray-900">
                      ${parseFloat(latestPrice.close).toLocaleString(undefined, {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2
                      })}
                    </p>
                    <p className={`text-sm mt-1 ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
                      {isPositive ? '+' : ''}{priceChange.toFixed(2)}% (24h)
                    </p>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div className="bg-white rounded-lg shadow-md mb-8">
          <div className="border-b border-gray-200">
            <nav className="flex -mb-px">
              {['overview', 'models', 'forecast', 'clustering'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                    activeTab === tab
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              ))}
            </nav>
          </div>

          <div className="p-6">
            {activeTab === 'overview' && selectedCrypto && (
              <PriceChart
                crypto={selectedCrypto}
                priceData={priceData[selectedCrypto.id] || []}
              />
            )}
            {activeTab === 'models' && selectedCrypto && (
              <ModelPerformance cryptoId={selectedCrypto.id} cryptoName={selectedCrypto.name} />
            )}
            {activeTab === 'forecast' && selectedCrypto && (
              <ForecastView cryptoId={selectedCrypto.id} cryptoName={selectedCrypto.name} />
            )}
            {activeTab === 'clustering' && selectedCrypto && (
              <ClusteringView cryptoId={selectedCrypto.id} cryptoName={selectedCrypto.name} />
            )}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">About This Project</h2>
          <div className="prose max-w-none">
            <p className="text-gray-700 mb-4">
              This cryptocurrency prediction platform uses machine learning to analyze and forecast
              prices for Bitcoin, Ethereum, and Dogecoin. The system combines multiple approaches:
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li><strong>Regression Models:</strong> Ridge, Lasso, ElasticNet, SVR, Random Forest, and Gradient Boosting</li>
              <li><strong>Time Series Forecasting:</strong> ARIMA and LSTM neural networks</li>
              <li><strong>Clustering Analysis:</strong> KMeans, DBSCAN, Agglomerative, and GMM for market regime identification</li>
            </ul>
            <p className="text-gray-700 mt-4">
              The models achieve high accuracy with R² scores above 94% for all cryptocurrencies.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
