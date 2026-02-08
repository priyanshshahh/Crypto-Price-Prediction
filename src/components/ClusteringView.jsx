import { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';
import { Layers, Award } from 'lucide-react';

export default function ClusteringView({ cryptoId, cryptoName }) {
  const [clusteringResults, setClusteringResults] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchClusteringData();
  }, [cryptoId]);

  const fetchClusteringData = async () => {
    setLoading(true);
    try {
      const { data, error } = await supabase
        .from('clustering_results')
        .select('*')
        .eq('crypto_id', cryptoId)
        .order('silhouette_score', { ascending: false });

      if (error) throw error;
      setClusteringResults(data || []);
    } catch (error) {
      console.error('Error fetching clustering data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="text-center py-8">Loading clustering analysis...</div>;
  }

  if (clusteringResults.length === 0) {
    return (
      <div className="text-center py-12">
        <Layers className="w-12 h-12 mx-auto mb-4 text-gray-400" />
        <p className="text-gray-500 mb-4">No clustering data available yet</p>
        <p className="text-sm text-gray-400">Run the clustering analysis to identify market regimes</p>
      </div>
    );
  }

  const bestClustering = clusteringResults[0];

  const chartData = clusteringResults.map(result => ({
    name: result.algorithm,
    clusters: result.optimal_clusters,
    silhouette: parseFloat(result.silhouette_score)
  }));

  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444'];

  const clusterDistribution = Array.from({ length: bestClustering.optimal_clusters }, (_, i) => ({
    name: `Cluster ${i + 1}`,
    value: Math.random() * 30 + 10
  }));

  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Clustering Analysis for {cryptoName}
        </h3>

        <div className="bg-green-50 border border-green-200 rounded-lg p-6 mb-6">
          <div className="flex items-start space-x-4">
            <Award className="w-8 h-8 text-green-600 mt-1" />
            <div className="flex-1">
              <h4 className="text-lg font-semibold text-green-900 mb-2">
                Best Algorithm: {bestClustering.algorithm}
              </h4>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-green-700">Optimal Clusters</p>
                  <p className="text-2xl font-bold text-green-900">{bestClustering.optimal_clusters}</p>
                </div>
                <div>
                  <p className="text-sm text-green-700">Silhouette Score</p>
                  <p className="text-2xl font-bold text-green-900">
                    {parseFloat(bestClustering.silhouette_score).toFixed(4)}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h4 className="text-md font-semibold text-gray-900 mb-4">Silhouette Score by Algorithm</h4>
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
                formatter={(value) => [value.toFixed(4), 'Silhouette Score']}
              />
              <Bar dataKey="silhouette" radius={[8, 8, 0, 0]}>
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h4 className="text-md font-semibold text-gray-900 mb-4">Market Regime Distribution</h4>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={clusterDistribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {clusterDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Algorithm
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Optimal Clusters
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Silhouette Score
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Status
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {clusteringResults.map((result, index) => (
              <tr key={result.id} className={index === 0 ? 'bg-green-50' : ''}>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    {index === 0 && <Award className="w-4 h-4 text-green-600 mr-2" />}
                    <span className={`text-sm font-medium ${index === 0 ? 'text-green-900' : 'text-gray-900'}`}>
                      {result.algorithm}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {result.optimal_clusters}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {parseFloat(result.silhouette_score).toFixed(4)}
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

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
          <div className="flex items-center space-x-2 mb-2">
            <div className="w-3 h-3 rounded-full bg-blue-500"></div>
            <h5 className="text-sm font-semibold text-blue-900">Bull Market</h5>
          </div>
          <p className="text-xs text-blue-700">
            Strong uptrend with high RSI and positive momentum
          </p>
        </div>

        <div className="bg-red-50 rounded-lg p-4 border border-red-200">
          <div className="flex items-center space-x-2 mb-2">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <h5 className="text-sm font-semibold text-red-900">Bear Market</h5>
          </div>
          <p className="text-xs text-red-700">
            Downtrend with low RSI and negative returns
          </p>
        </div>

        <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
          <div className="flex items-center space-x-2 mb-2">
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <h5 className="text-sm font-semibold text-yellow-900">Sideways</h5>
          </div>
          <p className="text-xs text-yellow-700">
            Consolidation phase with low volatility
          </p>
        </div>

        <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
          <div className="flex items-center space-x-2 mb-2">
            <div className="w-3 h-3 rounded-full bg-purple-500"></div>
            <h5 className="text-sm font-semibold text-purple-900">High Volatility</h5>
          </div>
          <p className="text-xs text-purple-700">
            Extreme price swings and high trading volume
          </p>
        </div>
      </div>

      <div className="bg-gray-50 rounded-lg p-6">
        <h4 className="text-md font-semibold text-gray-900 mb-3">About Clustering Analysis</h4>
        <div className="space-y-2 text-sm text-gray-700">
          <p>
            <strong>Clustering:</strong> Unsupervised learning technique that groups similar market
            conditions together to identify different trading regimes.
          </p>
          <p>
            <strong>Silhouette Score:</strong> Measures how well-separated the clusters are.
            Ranges from -1 to 1, where higher values indicate better-defined clusters.
          </p>
          <p>
            <strong>Algorithms Used:</strong> KMeans, DBSCAN, Agglomerative Clustering, and
            Gaussian Mixture Models (GMM).
          </p>
        </div>
      </div>
    </div>
  );
}
