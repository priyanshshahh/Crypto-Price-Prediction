import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { format } from 'date-fns';

export default function PriceChart({ crypto, priceData }) {
  if (!priceData || priceData.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">No price data available</p>
      </div>
    );
  }

  const chartData = priceData.map(item => ({
    date: format(new Date(item.date), 'MMM dd'),
    fullDate: format(new Date(item.date), 'yyyy-MM-dd'),
    close: parseFloat(item.close),
    high: parseFloat(item.high),
    low: parseFloat(item.low),
    volume: parseFloat(item.volume)
  }));

  const getColor = (symbol) => {
    const colors = {
      'BTC': '#F7931A',
      'ETH': '#627EEA',
      'DOGE': '#C2A633'
    };
    return colors[symbol] || '#3B82F6';
  };

  const color = getColor(crypto.symbol);

  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          {crypto.name} Price History (90 Days)
        </h3>
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id={`color${crypto.symbol}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={color} stopOpacity={0.8}/>
                <stop offset="95%" stopColor={color} stopOpacity={0}/>
              </linearGradient>
            </defs>
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
              formatter={(value) => [`$${parseFloat(value).toLocaleString()}`, 'Price']}
              labelFormatter={(label, payload) => {
                if (payload && payload[0]) {
                  return payload[0].payload.fullDate;
                }
                return label;
              }}
            />
            <Area
              type="monotone"
              dataKey="close"
              stroke={color}
              fillOpacity={1}
              fill={`url(#color${crypto.symbol})`}
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          High-Low Range
        </h3>
        <ResponsiveContainer width="100%" height={300}>
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
              formatter={(value) => `$${parseFloat(value).toLocaleString()}`}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="high"
              stroke="#10b981"
              name="High"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="low"
              stroke="#ef4444"
              name="Low"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="close"
              stroke={color}
              name="Close"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {priceData.length > 0 && (() => {
          const latest = priceData[priceData.length - 1];
          const oldest = priceData[0];
          const allPrices = priceData.map(p => parseFloat(p.close));
          const maxPrice = Math.max(...allPrices);
          const minPrice = Math.min(...allPrices);
          const avgVolume = priceData.reduce((sum, p) => sum + parseFloat(p.volume), 0) / priceData.length;

          return (
            <>
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-sm text-gray-600 mb-1">Current Price</p>
                <p className="text-2xl font-bold text-gray-900">
                  ${parseFloat(latest.close).toLocaleString()}
                </p>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-sm text-gray-600 mb-1">90-Day High</p>
                <p className="text-2xl font-bold text-green-600">
                  ${maxPrice.toLocaleString()}
                </p>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-sm text-gray-600 mb-1">90-Day Low</p>
                <p className="text-2xl font-bold text-red-600">
                  ${minPrice.toLocaleString()}
                </p>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-sm text-gray-600 mb-1">Avg Volume</p>
                <p className="text-2xl font-bold text-gray-900">
                  {avgVolume.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </p>
              </div>
            </>
          );
        })()}
      </div>
    </div>
  );
}
