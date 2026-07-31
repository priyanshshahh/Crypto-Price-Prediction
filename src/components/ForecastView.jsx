import { useState, useEffect } from 'react';
import { getClient } from '../lib/supabase';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, ComposedChart } from 'recharts';
import { format } from 'date-fns';
import { Calendar, TrendingUp } from 'lucide-react';

function fmtUsd(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return '—';
  if (v >= 1000) return `$${v.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
  return `$${v.toLocaleString(undefined, { maximumFractionDigits: 6 })}`;
}

export default function ForecastView({ cryptoId, cryptoName, cryptoSymbol }) {
  const [forecasts, setForecasts] = useState([]);
  const [horizonRows, setHorizonRows] = useState([]);
  const [historicalPrices, setHistoricalPrices] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState('ARIMA');
  const [paperUsd, setPaperUsd] = useState(1000);
  const [paperHorizon, setPaperHorizon] = useState('1D');
  const [journalSummary, setJournalSummary] = useState(null);
  const [journalResolved, setJournalResolved] = useState([]);

  useEffect(() => {
    fetchData();
  }, [cryptoId, cryptoSymbol]);

  useEffect(() => {
    const available = ['ARIMA', 'LSTM', 'GRU'].filter((m) =>
      forecasts.some((f) => f.model_type === m)
    );
    if (available.length > 0 && !available.includes(selectedModel)) {
      setSelectedModel(available[0]);
    }
  }, [forecasts, selectedModel]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const db = await getClient();

      const { data: forecastData, error: forecastError } = await db
        .from('forecasts')
        .select('*')
        .eq('crypto_id', cryptoId)
        .order('forecast_date', { ascending: true });

      if (forecastError) throw forecastError;

      const { data: priceData, error: priceError } = await db
        .from('price_history')
        .select('*')
        .eq('crypto_id', cryptoId)
        .order('date', { ascending: true })
        .limit(30);

      if (priceError) throw priceError;

      const { data: hzData } = await db
        .from('horizon_forecasts')
        .select('*')
        .eq('crypto_id', cryptoId)
        .order('horizon_days', { ascending: true });

      setForecasts(forecastData || []);
      setHistoricalPrices(priceData || []);
      setHorizonRows(hzData || []);

      const { data: jSum } = await db.from('prediction_journal_summary').select('*');
      const { data: jRes } = await db.from('prediction_journal_resolved').select('*');
      setJournalSummary((jSum && jSum[0]) || null);
      const sym = cryptoSymbol || null;
      setJournalResolved((jRes || []).filter((r) => !sym || r.symbol === sym));
    } catch (error) {
      console.error('Error fetching forecast data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="text-center py-8">Loading forecast data...</div>;
  }

  const hasClassic = forecasts.length > 0;
  const hasHorizons = horizonRows.length > 0;

  if (!hasClassic && !hasHorizons) {
    return (
      <div className="text-center py-12">
        <Calendar className="w-12 h-12 mx-auto mb-4 text-gray-400" />
        <p className="text-gray-500 mb-4">No forecast data available yet</p>
        <p className="text-sm text-gray-400">
          Run <code className="bg-gray-100 px-1 rounded">pipeline.py --mode production</code> to generate 1D/1W/1M forecasts
        </p>
      </div>
    );
  }

  const arimaForecasts = forecasts.filter(f => f.model_type === 'ARIMA');
  const lstmForecasts = forecasts.filter(f => f.model_type === 'LSTM');
  const gruForecasts = forecasts.filter(f => f.model_type === 'GRU');
  const pathModels = ['ARIMA', 'LSTM', 'GRU'].filter((m) =>
    forecasts.some((f) => f.model_type === m)
  );

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
        forecastLo: forecast.predicted_price_lo != null ? parseFloat(forecast.predicted_price_lo) : undefined,
        forecastHi: forecast.predicted_price_hi != null ? parseFloat(forecast.predicted_price_hi) : undefined,
        type: 'forecast'
      }))
  ];

  const allForecasts = forecasts.filter(f => f.model_type === selectedModel);
  const avgForecast = allForecasts.length > 0
    ? allForecasts.reduce((sum, f) => sum + parseFloat(f.predicted_price), 0) / allForecasts.length
    : 0;

  const lastHistorical = historicalPrices.length > 0
    ? parseFloat(historicalPrices[historicalPrices.length - 1].close)
    : (horizonRows[0]?.current_price || 0);

  const forecastChange = lastHistorical > 0
    ? ((avgForecast - lastHistorical) / lastHistorical) * 100
    : 0;

  const overviewRow = horizonRows[0] || null;
  const selectedHz = horizonRows.find((r) => r.horizon_label === paperHorizon) || horizonRows[0];
  const paperPnl = selectedHz && Number.isFinite(Number(selectedHz.predicted_return_pct))
    ? paperUsd * (Number(selectedHz.predicted_return_pct) / 100)
    : null;
  const paperEnd = paperPnl != null ? paperUsd + paperPnl : null;

  return (
    <div className="space-y-8">
      <div className="border border-gray-200 rounded-lg p-4 bg-gray-50">
        <h3 className="text-sm font-semibold text-gray-900 mb-2">Forecast stack</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs text-gray-700">
          <div><p className="font-medium text-gray-900">Fundamentals</p><p>TVL, staking APY (ETH), scenario weights</p></div>
          <div><p className="font-medium text-gray-900">Technicals</p><p>MA / RSI / MACD / Bollinger + bias scale</p></div>
          <div><p className="font-medium text-gray-900">On-chain</p><p>Funding, hashrate, tx, gas</p></div>
          <div><p className="font-medium text-gray-900">Sentiment</p><p>Fear &amp; Greed + Google Trends</p></div>
          <div><p className="font-medium text-gray-900">Machine learning</p><p>LightGBM / XGBoost + LSTM / GRU</p></div>
          <div><p className="font-medium text-gray-900">Prediction markets</p><p>Crowd odds vs model P(up)</p></div>
          <div><p className="font-medium text-gray-900">Quant risk</p><p>EWMA volatility + Monte Carlo bands</p></div>
          <div><p className="font-medium text-gray-900">Discipline</p><p>Trust gates + walk-forward journal</p></div>
        </div>
        <p className="text-[11px] text-gray-500 mt-2">Probabilities and ranges — not certainty. Paper trading only.</p>
      </div>

      {hasHorizons && (
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Multi-horizon price forecast — {cryptoName}
          </h3>
          <p className="text-sm text-gray-500 mb-4">
            1D / 1W / 1M targets with residual p10–p90 bands.
            Trustworthy only when the model beats persistence and walk-forward ≥55%.
          </p>
          <div className="overflow-x-auto border border-gray-200 rounded-lg">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50 text-left text-gray-600">
                <tr>
                  <th className="px-4 py-3 font-medium">Horizon</th>
                  <th className="px-4 py-3 font-medium">Current</th>
                  <th className="px-4 py-3 font-medium">Predicted</th>
                  <th className="px-4 py-3 font-medium">Range (p10–p90)</th>
                  <th className="px-4 py-3 font-medium">Return</th>
                  <th className="px-4 py-3 font-medium">Model</th>
                  <th className="px-4 py-3 font-medium">P(up)</th>
                  <th className="px-4 py-3 font-medium">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {horizonRows.map((row) => {
                  const ret = Number(row.predicted_return_pct);
                  return (
                    <tr key={row.id || row.horizon_label} className="border-t border-gray-100">
                      <td className="px-4 py-3 font-semibold text-gray-900">{row.horizon_label}</td>
                      <td className="px-4 py-3">{fmtUsd(row.current_price)}</td>
                      <td className="px-4 py-3 font-semibold">{fmtUsd(row.predicted_price)}</td>
                      <td className="px-4 py-3 text-gray-600">
                        {fmtUsd(row.predicted_price_p10)} – {fmtUsd(row.predicted_price_p90)}
                      </td>
                      <td className={`px-4 py-3 font-medium ${ret >= 0 ? 'text-green-700' : 'text-red-700'}`}>
                        {Number.isFinite(ret) ? `${ret >= 0 ? '+' : ''}${ret.toFixed(2)}%` : '—'}
                      </td>
                      <td className="px-4 py-3 text-gray-600">{row.model}</td>
                      <td className="px-4 py-3 text-gray-600">
                        {Number.isFinite(Number(row.direction_prob_up))
                          ? `${(Number(row.direction_prob_up) * 100).toFixed(0)}%`
                          : '—'}
                      </td>
                      <td className="px-4 py-3">
                        {row.trustworthy ? (
                          <span className="inline-flex px-2 py-0.5 rounded text-xs font-medium bg-emerald-50 text-emerald-800 border border-emerald-200">
                            Trustworthy
                          </span>
                        ) : (
                          <span className="inline-flex px-2 py-0.5 rounded text-xs font-medium bg-amber-50 text-amber-800 border border-amber-200">
                            Low confidence
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            As of {horizonRows[0]?.as_of || '—'}. Paper only. Public AI tools ~55–65% direction;
            Trustworthy badge only when persistence + walk-forward + direction gates pass.
            Educational / paper trading only — not financial advice.
          </p>

          {overviewRow && (
            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="border border-gray-200 rounded-lg p-4 bg-white">
                <h4 className="text-sm font-semibold text-gray-900 mb-1">Technical overview</h4>
                <p className="text-xs text-gray-500 mb-3">Indicator vote: RSI, MACD, MA cross, Fear &amp; Greed</p>
                <p className="text-lg font-semibold capitalize text-gray-900">
                  {overviewRow.technical_bias || '—'}
                  {overviewRow.technical_scale != null ? (
                    <span className="text-sm font-normal text-gray-600"> ({overviewRow.technical_scale}/100)</span>
                  ) : null}
                </p>
                <div className="mt-2 h-2 bg-gray-100 rounded overflow-hidden">
                  <div
                    className="h-full bg-blue-600"
                    style={{ width: `${Math.min(100, Math.max(0, Number(overviewRow.technical_scale) || 50))}%` }}
                  />
                </div>
                <ul className="mt-3 space-y-1 text-xs text-gray-600">
                  {(overviewRow.technical_signals || []).slice(0, 5).map((s) => (
                    <li key={s.name || s.note}>{s.note || s.name}</li>
                  ))}
                </ul>
              </div>

              <div className="border border-gray-200 rounded-lg p-4 bg-white">
                <h4 className="text-sm font-semibold text-gray-900 mb-1">Risk &amp; scorecard</h4>
                <p className="text-xs text-gray-500 mb-3">Volatility / ATR risk · holdout hit rate by horizon</p>
                <p className="text-sm text-gray-800">
                  Risk: <span className="font-semibold capitalize">{overviewRow.risk_level || '—'}</span>
                  {overviewRow.atr_pct != null ? ` · ATR ${Number(overviewRow.atr_pct).toFixed(2)}%` : ''}
                </p>
                <div className="mt-3 space-y-2 text-xs text-gray-700">
                  {horizonRows.map((r) => {
                    const sc = r.scorecard || {};
                    const hit = sc.holdout_hit_rate != null
                      ? sc.holdout_hit_rate
                      : r.holdout_directional_accuracy;
                    return (
                      <div key={r.horizon_label} className="flex justify-between border-b border-gray-50 py-1">
                        <span className="font-medium">{r.horizon_label}</span>
                        <span>
                          Hit {hit != null ? `${(Number(hit) * 100).toFixed(0)}%` : '—'}
                          {sc.grade ? ` · ${sc.grade}` : ''}
                        </span>
                      </div>
                    );
                  })}
                </div>
                <p className="text-[11px] text-gray-500 mt-2">
                  Hit rate = chronological holdout direction — not a live Correct/Incorrect label until the horizon resolves.
                </p>
              </div>

              <div className="border border-gray-200 rounded-lg p-4 bg-white">
                <h4 className="text-sm font-semibold text-gray-900 mb-1">Paper profit calculator</h4>
                <p className="text-xs text-gray-500 mb-3">Educational only — applies model predicted return to a notional</p>
                <label className="block text-xs text-gray-600 mb-1">Notional (USD)</label>
                <input
                  type="number"
                  min={1}
                  value={paperUsd}
                  onChange={(e) => setPaperUsd(Number(e.target.value) || 0)}
                  className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mb-2"
                />
                <label className="block text-xs text-gray-600 mb-1">Horizon</label>
                <select
                  value={selectedHz?.horizon_label || paperHorizon}
                  onChange={(e) => setPaperHorizon(e.target.value)}
                  className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mb-3"
                >
                  {horizonRows.map((r) => (
                    <option key={r.horizon_label} value={r.horizon_label}>{r.horizon_label}</option>
                  ))}
                </select>
                <p className="text-sm text-gray-800">
                  Predicted: {selectedHz && Number.isFinite(Number(selectedHz.predicted_return_pct))
                    ? `${Number(selectedHz.predicted_return_pct) >= 0 ? '+' : ''}${Number(selectedHz.predicted_return_pct).toFixed(2)}%`
                    : '—'}
                </p>
                <p className={`text-lg font-semibold mt-1 ${paperPnl != null && paperPnl >= 0 ? 'text-green-700' : 'text-red-700'}`}>
                  {paperEnd != null && paperPnl != null
                    ? `${fmtUsd(paperEnd)} (${paperPnl >= 0 ? '+' : '−'}${fmtUsd(Math.abs(paperPnl))})`
                    : '—'}
                </p>
                <p className="text-[11px] text-gray-500 mt-2">Not financial advice. Ignores fees/slippage here (paper book uses costs separately).</p>
              </div>
            </div>
          )}

          {selectedHz?.reasoning?.length > 0 && (
            <div className="mt-4 border border-gray-200 rounded-lg p-4 bg-gray-50">
              <h4 className="text-sm font-semibold text-gray-900 mb-2">
                Evidence — {selectedHz.horizon_label}
              </h4>
              <ul className="list-disc list-inside space-y-1 text-xs text-gray-700">
                {selectedHz.reasoning.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="border border-gray-200 rounded-lg p-4 bg-white">
              <h4 className="text-sm font-semibold text-gray-900 mb-1">Crowd odds vs model</h4>
              <p className="text-xs text-gray-500 mb-2">Prediction-market Yes probability vs calibrated P(up)</p>
              <p className="text-sm text-gray-800">
                Crowd Yes: {selectedHz?.polymarket_yes != null ? `${(Number(selectedHz.polymarket_yes) * 100).toFixed(0)}%` : '—'}
              </p>
              <p className="text-sm text-gray-800">
                Model P(up): {selectedHz?.direction_prob_up != null ? `${(Number(selectedHz.direction_prob_up) * 100).toFixed(0)}%` : '—'}
              </p>
              <p className="text-xs text-gray-600 mt-1">
                Gap: {selectedHz?.polymarket_model_gap != null
                  ? `${(Number(selectedHz.polymarket_model_gap) * 100).toFixed(1)} pp`
                  : '—'}
              </p>
            </div>
            <div className="border border-gray-200 rounded-lg p-4 bg-white">
              <h4 className="text-sm font-semibold text-gray-900 mb-1">Monte Carlo scenarios</h4>
              <p className="text-xs text-gray-500 mb-2">EWMA vol · quant paths (not promises)</p>
              {selectedHz?.mc_price_p50 != null ? (
                <div className="text-sm text-gray-800 space-y-1">
                  <p>p10 {fmtUsd(selectedHz.mc_price_p10)}</p>
                  <p className="font-semibold">p50 {fmtUsd(selectedHz.mc_price_p50)}</p>
                  <p>p90 {fmtUsd(selectedHz.mc_price_p90)}</p>
                  <p className="text-xs text-gray-600">MC P(up) {(Number(selectedHz.mc_prob_up) * 100).toFixed(0)}%</p>
                </div>
              ) : (
                <p className="text-sm text-gray-500">Run production pipeline to refresh MC bands.</p>
              )}
            </div>
            <div className="border border-gray-200 rounded-lg p-4 bg-white">
              <h4 className="text-sm font-semibold text-gray-900 mb-1">Fundamentals</h4>
              <p className="text-xs text-gray-500 mb-2">TVL / staking / scenario tree</p>
              {selectedHz?.fundamentals ? (
                <div className="text-xs text-gray-700 space-y-1">
                  <p>{selectedHz.fundamentals.summary}</p>
                  {selectedHz.fundamentals.staking_apy != null && (
                    <p>Staking APY ~{Number(selectedHz.fundamentals.staking_apy).toFixed(2)}%</p>
                  )}
                  {(selectedHz.fundamentals.scenario_tree || []).map((s) => (
                    <p key={s.name}>{s.name}: {(Number(s.weight) * 100).toFixed(0)}%</p>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-gray-500">Thin free fundamentals for this asset.</p>
              )}
            </div>
          </div>

          <div className="mt-4 border border-gray-200 rounded-lg p-4 bg-white">
            <h4 className="text-sm font-semibold text-gray-900 mb-1">Prediction journal (walk-forward ML)</h4>
            <p className="text-xs text-gray-500 mb-2">
              Leak-free expanding-window LightGBM (purge=horizon) — not MA seeds.
              {journalSummary?.hit_rate != null
                ? ` · hit rate ${(Number(journalSummary.hit_rate) * 100).toFixed(0)}% (${journalSummary.n_correct}/${journalSummary.n_resolved})`
                : ' · no resolved rows yet'}
              {journalSummary?.n_pending != null ? ` · ${journalSummary.n_pending} pending live` : ''}
            </p>
            {journalResolved.length > 0 ? (
              <div className="max-h-40 overflow-y-auto text-xs">
                {journalResolved
                  .filter((r) => r.source === 'walkforward_ml' || !r.source)
                  .slice(-12)
                  .reverse()
                  .map((r, i) => (
                  <div key={`${r.as_of}-${r.horizon_days}-${i}`} className="flex justify-between py-1 border-b border-gray-50">
                    <span>{r.as_of} · {r.horizon_label || `${r.horizon_days}d`} · {(r.model || 'WF').replace('WalkForward-', '')}</span>
                    <span className={r.verdict === 'Correct' ? 'text-green-700 font-medium' : r.verdict === 'Incorrect' ? 'text-red-700 font-medium' : 'text-gray-500'}>
                      {r.verdict}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-gray-500">Run production without --skip-wf-ledger to build the skill ledger.</p>
            )}
          </div>
        </div>
      )}

      {hasClassic && (
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Path forecast (ARIMA / LSTM / GRU) for {cryptoName}
        </h3>

        <div className="flex flex-wrap gap-3 mb-6">
          {pathModels.map((model) => (
            <button
              key={model}
              onClick={() => setSelectedModel(model)}
              className={`px-5 py-2.5 rounded-lg font-medium transition-colors ${
                selectedModel === model
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {model}
            </button>
          ))}
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
          <ComposedChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="date" stroke="#6b7280" style={{ fontSize: '12px' }} />
            <YAxis
              stroke="#6b7280"
              style={{ fontSize: '12px' }}
              domain={['auto', 'auto']}
              tickFormatter={(value) => `$${value.toLocaleString()}`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
              }}
              formatter={(value, name) => {
                if (value == null || Number.isNaN(Number(value))) return [null, null];
                const labels = {
                  historical: 'Historical',
                  forecast: `${selectedModel} path`,
                  forecastLo: '80% low',
                  forecastHi: '80% high',
                };
                return [`$${parseFloat(value).toLocaleString()}`, labels[name] || name];
              }}
              labelFormatter={(label, payload) => {
                if (payload && payload[0]) return payload[0].payload.fullDate;
                return label;
              }}
            />
            <Legend />
            <Area
              type="monotone"
              dataKey="forecastHi"
              stroke="none"
              fill="#10B981"
              fillOpacity={0.12}
              name="80% high"
              connectNulls
            />
            <Area
              type="monotone"
              dataKey="forecastLo"
              stroke="none"
              fill="#ffffff"
              fillOpacity={1}
              name="80% low"
              connectNulls
            />
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
              dot={{ fill: '#10B981', r: 3 }}
              name={`${selectedModel} path`}
              connectNulls
            />
          </ComposedChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-500 mt-2">
          ARIMA is fit on log-returns then compounded (not raw prices). Old level-ARIMA(0,1,1)
          multi-step paths were flat by math — that is fixed. Primary product: 1D/1W/1M table above.
        </p>
      </div>
      )}

      {hasClassic && (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h4 className="text-md font-semibold text-gray-900 mb-4">ARIMA (log-return path)</h4>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {arimaForecasts.length > 0 ? (
              arimaForecasts.map((forecast) => (
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
              <p className="text-sm text-gray-500 text-center py-4">No ARIMA forecasts</p>
            )}
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h4 className="text-md font-semibold text-gray-900 mb-4">LSTM</h4>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {lstmForecasts.length > 0 ? (
              lstmForecasts.map((forecast) => (
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
              <p className="text-sm text-gray-500 text-center py-4">No LSTM — run without --skip-lstm</p>
            )}
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h4 className="text-md font-semibold text-gray-900 mb-4">GRU</h4>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {gruForecasts.length > 0 ? (
              gruForecasts.map((forecast) => (
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
              <p className="text-sm text-gray-500 text-center py-4">No GRU — run without --skip-lstm</p>
            )}
          </div>
        </div>
      </div>
      )}

      <div className="bg-gray-50 rounded-lg p-6">
        <h4 className="text-md font-semibold text-gray-900 mb-3">About these forecasts</h4>
        <div className="space-y-2 text-sm text-gray-700">
          <p><strong>1D / 1W / 1M:</strong> Tabular ML on returns with on-chain/macro features, persistence + walk-forward gates.</p>
          <p><strong>ARIMA:</strong> Log-return ARMA with drift, compounded path + 80% bands.</p>
          <p><strong>LSTM / GRU:</strong> Deep sequence models (research pillar); evaluate vs persistence on returns, not price R².</p>
          <p className="text-xs text-gray-500 mt-4">Educational / paper trading only. Not financial advice. See docs/METHODS.md</p>
        </div>
      </div>
    </div>
  );
}
