import { useState, useEffect } from 'react';
import { getClient } from '../lib/supabase';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';
import { Award } from 'lucide-react';

export default function ModelPerformance({ cryptoId, cryptoName }) {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModels();
  }, [cryptoId]);

  const fetchModels = async () => {
    setLoading(true);
    try {
      const db = await getClient();
      const { data, error } = await db
        .from('regression_models')
        .select('*')
        .eq('crypto_id', cryptoId)
        .order('directional_accuracy', { ascending: false });

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
        <p className="text-sm text-gray-400">Run `python pipeline.py` to populate model performance metrics</p>
      </div>
    );
  }

  const baseline = models.find((m) => m.model_name === 'Persistence (baseline)');
  const realModels = models.filter((m) => m.model_name !== 'Persistence (baseline)');
  const bestModel = realModels[0];

  const chartData = realModels.map((model) => ({
    name: model.model_name,
    dirAcc: parseFloat(model.directional_accuracy),
    rmse: parseFloat(model.rmse),
  }));

  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'];

  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Model Performance for {cryptoName} — next-day <em>return</em> prediction
        </h3>

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-6">
          <div className="flex items-start space-x-4">
            <Award className="w-8 h-8 text-blue-600 mt-1" />
            <div className="flex-1">
              <h4 className="text-lg font-semibold text-blue-900 mb-2">Best Model: {bestModel.model_name}</h4>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-blue-700">Directional Accuracy</p>
                  <p className="text-2xl font-bold text-blue-900">{(bestModel.directional_accuracy * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-sm text-blue-700">Return RMSE</p>
                  <p className="text-2xl font-bold text-blue-900">{parseFloat(bestModel.rmse).toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-sm text-blue-700">Return R²</p>
                  <p className="text-2xl font-bold text-blue-900">{parseFloat(bestModel.r2_score).toFixed(4)}</p>
                </div>
              </div>
              {baseline && (
                <p className="text-sm text-blue-800 mt-3">
                  Naive persistence baseline: return RMSE {parseFloat(baseline.rmse).toFixed(4)}, up-day share{' '}
                  {(baseline.directional_accuracy * 100).toFixed(1)}%. A model only adds value if it beats this.
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      <div>
        <h4 className="text-md font-semibold text-gray-900 mb-4">Directional Accuracy (share of test days with correct up/down call)</h4>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="name" stroke="#6b7280" style={{ fontSize: '12px' }} />
            <YAxis stroke="#6b7280" style={{ fontSize: '12px' }} domain={[0, 1]} />
            <Tooltip
              contentStyle={{ backgroundColor: 'white', border: '1px solid #e5e7eb', borderRadius: '8px' }}
              formatter={(value) => [(value * 100).toFixed(1) + '%', 'Directional Accuracy']}
            />
            <ReferenceLine y={0.5} stroke="#9ca3af" strokeDasharray="4 4" label={{ value: 'coin flip', fontSize: 11, fill: '#6b7280' }} />
            <Bar dataKey="dirAcc" radius={[8, 8, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {['Model', 'Dir. Accuracy', 'Return R²', 'Return RMSE', 'Return MAE', 'Status'].map((h) => (
                <th key={h} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {models.map((model) => {
              const isBaseline = model.model_name === 'Persistence (baseline)';
              const isBest = !isBaseline && model.id === bestModel.id;
              return (
                <tr key={model.id} className={isBest ? 'bg-blue-50' : isBaseline ? 'bg-gray-50' : ''}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      {isBest && <Award className="w-4 h-4 text-blue-600 mr-2" />}
                      <span className={`text-sm font-medium ${isBest ? 'text-blue-900' : 'text-gray-900'}`}>
                        {model.model_name}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {(model.directional_accuracy * 100).toFixed(1)}%{isBaseline ? ' *' : ''}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{parseFloat(model.r2_score).toFixed(4)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{parseFloat(model.rmse).toFixed(4)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{parseFloat(model.mae).toFixed(4)}</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {isBest ? (
                      <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">Best</span>
                    ) : isBaseline ? (
                      <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-yellow-100 text-yellow-800">Baseline</span>
                    ) : (
                      <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-gray-100 text-gray-800">Alternative</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="bg-gray-50 rounded-lg p-6">
        <h4 className="text-md font-semibold text-gray-900 mb-3">How to read these numbers (honestly)</h4>
        <div className="space-y-2 text-sm text-gray-700">
          <p><strong>Targets are next-day returns, not prices.</strong> Price-level R² on next-day close is trivially high (&gt;0.9) for any model — including "predict yesterday's price" — because prices are strongly autocorrelated. Returns-based metrics are the honest view.</p>
          <p><strong>Directional Accuracy:</strong> share of test days where the model called the up/down move correctly. ~50% = coin flip. * For the persistence baseline this column shows the share of up days (an always-up strawman).</p>
          <p><strong>Return R² near or below 0 is expected:</strong> daily crypto returns are close to unpredictable from technical indicators alone. Models add value only where they beat the persistence baseline.</p>
          <p>Evaluation: chronological 80/20 holdout, scalers fit on train only, no shuffling. All numbers are from a real pipeline run (see results/metrics.json for provenance).</p>
        </div>
      </div>
    </div>
  );
}
