import { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { Award, TrendingUp } from 'lucide-react';

export default function ModelPerformance({ cryptoId, cryptoName }) {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModels();
  }, [cryptoId]);

  const fetchModels = async () => {
    setLoading(true);
    try {
      const { data, error } = await supabase
        .from('regression_models')
        .select('*')
        .eq('crypto_id', cryptoId)
        .order('r2_score', { ascending: false });

      if (error) throw error;
      setModels(data || []);
    } catch (error) {
      console.error('Error fetching models:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="text-center py-8">Loading model performance...</div>;
  }

  if (models.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500 mb-4">No model data available yet</p>
        <p className="text-sm text-gray-400">Run the regression analysis to populate model performance metrics</p>
      </div>
    );
  }

  const bestModel = models[0];

  const chartData = models.map(model => ({
    name: model.model_name,
    r2: parseFloat(model.r2_score),
    rmse: parseFloat(model.rmse),
    mae: parseFloat(model.mae)
  }));

  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#6366F1'];

  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Model Performance for {cryptoName}
        </h3>

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-6">
          <div className="flex items-start space-x-4">
            <Award className="w-8 h-8 text-blue-600 mt-1" />
            <div className="flex-1">
              <h4 className="text-lg font-semibold text-blue-900 mb-2">Best Model: {bestModel.model_name}</h4>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-blue-700">R² Score</p>
                  <p className="text-2xl font-bold text-blue-900">{(bestModel.r2_score * 100).toFixed(2)}%</p>
                </div>
                <div>
                  <p className="text-sm text-blue-700">RMSE</p>
                  <p className="text-2xl font-bold text-blue-900">{parseFloat(bestModel.rmse).toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-sm text-blue-700">MAE</p>
                  <p className="text-2xl font-bold text-blue-900">{parseFloat(bestModel.mae).toFixed(2)}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div>
        <h4 className="text-md font-semibold text-gray-900 mb-4">R² Score Comparison</h4>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="name" stroke="#6b7280" style={{ fontSize: '12px' }} />
            <YAxis stroke="#6b7280" style={{ fontSize: '12px' }} domain={[0, 1]} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
              }}
              formatter={(value) => [(value * 100).toFixed(2) + '%', 'R² Score']}
            />
            <Bar dataKey="r2" radius={[8, 8, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div>
        <h4 className="text-md font-semibold text-gray-900 mb-4">Error Metrics (RMSE)</h4>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="name" stroke="#6b7280" style={{ fontSize: '12px' }} />
            <YAxis stroke="#6b7280" style={{ fontSize: '12px' }} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
              }}
              formatter={(value) => [value.toFixed(2), 'RMSE']}
            />
            <Bar dataKey="rmse" fill="#EF4444" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Model
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                R² Score
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                RMSE
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                MAE
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Status
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {models.map((model, index) => (
              <tr key={model.id} className={index === 0 ? 'bg-blue-50' : ''}>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    {index === 0 && <Award className="w-4 h-4 text-blue-600 mr-2" />}
                    <span className={`text-sm font-medium ${index === 0 ? 'text-blue-900' : 'text-gray-900'}`}>
                      {model.model_name}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {(model.r2_score * 100).toFixed(4)}%
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {parseFloat(model.rmse).toFixed(2)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {parseFloat(model.mae).toFixed(2)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {index === 0 ? (
                    <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                      Best
                    </span>
                  ) : (
                    <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-gray-100 text-gray-800">
                      Alternative
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="bg-gray-50 rounded-lg p-6">
        <h4 className="text-md font-semibold text-gray-900 mb-3">About Model Metrics</h4>
        <div className="space-y-2 text-sm text-gray-700">
          <p><strong>R² Score:</strong> Measures how well the model explains price variance. Higher is better (max 1.0 = 100%).</p>
          <p><strong>RMSE (Root Mean Square Error):</strong> Average prediction error in price units. Lower is better.</p>
          <p><strong>MAE (Mean Absolute Error):</strong> Average absolute prediction error. Lower is better.</p>
        </div>
      </div>
    </div>
  );
}
