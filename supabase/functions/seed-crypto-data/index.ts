import { createClient } from 'npm:@supabase/supabase-js@2';

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Client-Info, Apikey",
};

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, {
      status: 200,
      headers: corsHeaders,
    });
  }

  try {
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
    const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    const { data: cryptos } = await supabase
      .from('cryptocurrencies')
      .select('*');

    if (!cryptos || cryptos.length === 0) {
      return new Response(
        JSON.stringify({ error: 'No cryptocurrencies found' }),
        {
          status: 404,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    const btc = cryptos.find(c => c.symbol === 'BTC');
    const eth = cryptos.find(c => c.symbol === 'ETH');
    const doge = cryptos.find(c => c.symbol === 'DOGE');

    const today = new Date();
    const priceHistory = [];

    for (let i = 90; i >= 0; i--) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);
      const dateStr = date.toISOString().split('T')[0];

      const btcPrice = 45000 + Math.random() * 20000 + (90 - i) * 150;
      const ethPrice = 2500 + Math.random() * 1000 + (90 - i) * 15;
      const dogePrice = 0.08 + Math.random() * 0.04 + (90 - i) * 0.0005;

      if (btc) {
        priceHistory.push({
          crypto_id: btc.id,
          date: dateStr,
          open: btcPrice * 0.98,
          high: btcPrice * 1.03,
          low: btcPrice * 0.97,
          close: btcPrice,
          volume: Math.random() * 10000000000
        });
      }

      if (eth) {
        priceHistory.push({
          crypto_id: eth.id,
          date: dateStr,
          open: ethPrice * 0.98,
          high: ethPrice * 1.03,
          low: ethPrice * 0.97,
          close: ethPrice,
          volume: Math.random() * 5000000000
        });
      }

      if (doge) {
        priceHistory.push({
          crypto_id: doge.id,
          date: dateStr,
          open: dogePrice * 0.98,
          high: dogePrice * 1.03,
          low: dogePrice * 0.97,
          close: dogePrice,
          volume: Math.random() * 1000000000
        });
      }
    }

    await supabase.from('price_history').delete().neq('id', '00000000-0000-0000-0000-000000000000');
    const { error: priceError } = await supabase
      .from('price_history')
      .insert(priceHistory);

    if (priceError) throw priceError;

    const regressionModels = [];
    const models = ['Linear', 'Ridge', 'Lasso', 'ElasticNet', 'SVR', 'RandomForest', 'GradientBoosting'];

    [btc, eth, doge].forEach(crypto => {
      if (!crypto) return;
      models.forEach((model, idx) => {
        const baseR2 = 0.85 + Math.random() * 0.13;
        regressionModels.push({
          crypto_id: crypto.id,
          model_name: model,
          r2_score: baseR2,
          rmse: (1 - baseR2) * 1000 + Math.random() * 500,
          mae: (1 - baseR2) * 800 + Math.random() * 300,
          is_best: idx === models.length - 1
        });
      });
    });

    await supabase.from('regression_models').delete().neq('id', '00000000-0000-0000-0000-000000000000');
    const { error: modelError } = await supabase
      .from('regression_models')
      .insert(regressionModels);

    if (modelError) throw modelError;

    const forecasts = [];
    for (let i = 1; i <= 30; i++) {
      const forecastDate = new Date(today);
      forecastDate.setDate(forecastDate.getDate() + i);
      const dateStr = forecastDate.toISOString().split('T')[0];

      if (btc) {
        forecasts.push({
          crypto_id: btc.id,
          model_type: 'ARIMA',
          forecast_date: dateStr,
          predicted_price: 58000 + Math.random() * 5000 + i * 100
        });
        forecasts.push({
          crypto_id: btc.id,
          model_type: 'LSTM',
          forecast_date: dateStr,
          predicted_price: 59000 + Math.random() * 4000 + i * 120
        });
      }

      if (eth) {
        forecasts.push({
          crypto_id: eth.id,
          model_type: 'ARIMA',
          forecast_date: dateStr,
          predicted_price: 3800 + Math.random() * 300 + i * 10
        });
        forecasts.push({
          crypto_id: eth.id,
          model_type: 'LSTM',
          forecast_date: dateStr,
          predicted_price: 3850 + Math.random() * 250 + i * 12
        });
      }

      if (doge) {
        forecasts.push({
          crypto_id: doge.id,
          model_type: 'ARIMA',
          forecast_date: dateStr,
          predicted_price: 0.12 + Math.random() * 0.02 + i * 0.0002
        });
        forecasts.push({
          crypto_id: doge.id,
          model_type: 'LSTM',
          forecast_date: dateStr,
          predicted_price: 0.125 + Math.random() * 0.015 + i * 0.00025
        });
      }
    }

    await supabase.from('forecasts').delete().neq('id', '00000000-0000-0000-0000-000000000000');
    const { error: forecastError } = await supabase
      .from('forecasts')
      .insert(forecasts);

    if (forecastError) throw forecastError;

    const clusteringResults = [];
    const algorithms = ['KMeans', 'DBSCAN', 'Agglomerative', 'GMM'];

    [btc, eth, doge].forEach(crypto => {
      if (!crypto) return;
      algorithms.forEach((algorithm, idx) => {
        clusteringResults.push({
          crypto_id: crypto.id,
          algorithm,
          optimal_clusters: 3 + Math.floor(Math.random() * 3),
          silhouette_score: 0.15 + Math.random() * 0.25,
          is_best: idx === 0
        });
      });
    });

    await supabase.from('clustering_results').delete().neq('id', '00000000-0000-0000-0000-000000000000');
    const { error: clusterError } = await supabase
      .from('clustering_results')
      .insert(clusteringResults);

    if (clusterError) throw clusterError;

    return new Response(
      JSON.stringify({
        success: true,
        message: 'Database seeded successfully',
        counts: {
          priceHistory: priceHistory.length,
          models: regressionModels.length,
          forecasts: forecasts.length,
          clustering: clusteringResults.length
        }
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});
