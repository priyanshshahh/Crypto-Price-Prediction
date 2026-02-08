import { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { format } from 'date-fns';
import { Calendar, TrendingUp } from 'lucide-react';

export default function ForecastView({ cryptoId, cryptoName }) {
  const [forecasts, setForecasts] = useState([]);
  const [historicalPrices, setHistoricalPrices] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState('ARIMA');

  useEffect(() => {
    fetchData();
  }, [cryptoId]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const { data: forecastData, error: forecastError } = await supabase
        .from('forecasts')
        .select('*')
        .eq('crypto_id', cryptoId)
        .order('forecast_date', { ascending: true });

      if (forecastError) throw forecastError;

      const { data: priceData, error: priceError } = await supabase
        .from('price_history')
        .select('*')
        .eq('crypto_id', cryptoId)
        .order('date', { ascending: true })
        .limit(30);

      if (priceError) throw priceError;

      setForecasts(forecastData || []);
      setHistoricalPrices(priceData || []);
    } catch (error) {
      console.error('Error fetching forecast data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="text-center py-8">Loading forecast data...</div>;
  }

  if (forecasts.length === 0) {
    return (
      <div className="text-center py-12">
        <Calendar className="w-12 h-12 mx-auto mb-4 text-gray-400" />
        <p className="text-gray-500 mb-4">No forecast data available yet</p>
        <p className="text-sm text-gray-400">Run the time series models to generate price forecasts</p>
      </div>
    );
  }

  const arimaForecasts = forecasts.filter(f => f.model_type === 'ARIMA');
  const lstmForecasts = forecasts.filter(f => f.model_type === 'LSTM');

  const chartData = [
    ...historicalPrices.map(price => ({
      date: format(new Date(price.date), 'MMM dd'),
      fullDate: price.date,
      historical: parseFloat(price.close),
      type: 'historical'
    })),
    ...forecasts
      .filter(f => f.model_type === selectedModel)
      .map(forecast => ({
        date: format(new Date(forecast.forecast_date), 'MMM dd'),
        fullDate: forecast.forecast_date,
        forecast: parseFloat(forecast.predicted_price),
        type: 'forecast'
      }))
  ];

  const allForecasts = forecasts.filter(f => f.model_type === selectedModel);
  const avgForecast = allForecasts.length > 0
    ? allForecasts.reduce((sum, f) => sum + parseFloat(f.predicted_price), 0) / allForecasts.length
    : 0;

  const lastHistorical = historicalPrices.length > 0
    ? parseFloat(historicalPrices[historicalPrices.length - 1].close)
    : 0;

  const forecastChange = lastHistorical > 0
    ? ((avgForecast - lastHistorical) / lastHistorical) * 100
    : 0;

  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Price Forecast for {cryptoName}
        </h3>

        <div className="flex space-x-4 mb-6">
          <button
            onClick={() => setSelectedModel('ARIMA')}
            className={`px-6 py-3 rounded-lg font-medium transition-colors ${
              selectedModel === 'ARIMA'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            ARIMA Model
          </button>
          <button
            onClick={() => setSelectedModel('LSTM')}
            className={`px-6 py-3 rounded-lg font-medium transition-colors ${
              selectedModel === 'LSTM'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            LSTM Model
          </button>
        </div>

        {allForecasts.length > 0 && (
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6 mb-6">
            <div className="flex items-start space-x-4">
              <TrendingUp className="w-8 h-8 text-blue-600 mt-1" />
              <div className="flex-1">
                <h4 className="text-lg font-semibold text-blue-900 mb-2">{selectedModel} Forecast Summary</h4>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <p className="text-sm text-blue-700">Current Price</p>
                    <p className="text-2xl font-bold text-blue-900">${lastHistorical.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-sm text-blue-700">Avg Forecast</p>
                    <p className="text-2xl font-bold text-blue-900">${avgForecast.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-sm text-blue-700">Expected Change</p>
                    <p className={`text-2xl font-bold ${forecastChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {forecastChange >= 0 ? '+' : ''}{forecastChange.toFixed(2)}%
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="date"
              stroke="#6b7280"
              style={{ fontSize: '12px' }}
            />
            <YAxis
              stroke="#6b7280"
              style={{ fontSize: '12px' }}
              tickFormatter={(value) => `$${value.toLocaleString()}`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
              }}
              formatter={(value, name) => [
                `$${parseFloat(value).toLocaleString()}`,
                name === 'historical' ? 'Historical Price' : `${selectedModel} Forecast`
              ]}
              labelFormatter={(label, payload) => {
                if (payload && payload[0]) {
                  return payload[0].payload.fullDate;
                }
                return label;
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="historical"
              stroke="#3B82F6"
              strokeWidth={2}
              dot={false}
              name="Historical"
            />
            <Line
              type="monotone"
              dataKey="forecast"
              stroke="#10B981"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={{ fill: '#10B981', r: 4 }}
              name={`${selectedModel} Forecast`}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h4 className="text-md font-semibold text-gray-900 mb-4">ARIMA Forecasts</h4>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {arimaForecasts.length > 0 ? (
              arimaForecasts.map((forecast, idx) => (
                <div key={forecast.id} className="flex justify-between items-center py-2 border-b border-gray-100">
                  <span className="text-sm text-gray-600">
                    {format(new Date(forecast.forecast_date), 'MMM dd, yyyy')}
                  </span>
                  <span className="text-sm font-semibold text-gray-900">
                    ${parseFloat(forecast.predicted_price).toLocaleString()}
                  </span>
                </div>
              ))
            ) : (
              <p className="text-sm text-gray-500 text-center py-4">No ARIMA forecasts available</p>
            )}
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h4 className="text-md font-semibold text-gray-900 mb-4">LSTM Forecasts</h4>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {lstmForecasts.length > 0 ? (
              lstmForecasts.map((forecast, idx) => (
                <div key={forecast.id} className="flex justify-between items-center py-2 border-b border-gray-100">
                  <span className="text-sm text-gray-600">
                    {format(new Date(forecast.forecast_date), 'MMM dd, yyyy')}
                  </span>
                  <span className="text-sm font-semibold text-gray-900">
                    ${parseFloat(forecast.predicted_price).toLocaleString()}
                  </span>
                </div>
              ))
            ) : (
              <p className="text-sm text-gray-500 text-center py-4">No LSTM forecasts available</p>
            )}
          </div>
        </div>
      </div>

      <div className="bg-gray-50 rounded-lg p-6">
        <h4 className="text-md font-semibold text-gray-900 mb-3">About Forecasting Models</h4>
        <div className="space-y-2 text-sm text-gray-700">
          <p><strong>ARIMA:</strong> Statistical time series model that captures trends and patterns in historical data.</p>
          <p><strong>LSTM:</strong> Deep learning neural network that learns complex temporal dependencies in price movements.</p>
          <p className="text-xs text-gray-500 mt-4">Note: These forecasts are for educational purposes only and should not be used as financial advice.</p>
        </div>
      </div>
    </div>
  );
}
