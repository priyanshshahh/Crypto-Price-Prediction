/*
  # Cryptocurrency Prediction Database Schema

  1. New Tables
    - `cryptocurrencies`
      - `id` (uuid, primary key)
      - `symbol` (text) - BTC, ETH, DOGE
      - `name` (text) - Bitcoin, Ethereum, Dogecoin
      - `created_at` (timestamptz)
    
    - `price_history`
      - `id` (uuid, primary key)
      - `crypto_id` (uuid, foreign key)
      - `date` (date)
      - `open` (numeric)
      - `high` (numeric)
      - `low` (numeric)
      - `close` (numeric)
      - `volume` (numeric)
      - `created_at` (timestamptz)
    
    - `regression_models`
      - `id` (uuid, primary key)
      - `crypto_id` (uuid, foreign key)
      - `model_name` (text) - Linear, Ridge, Lasso, etc.
      - `r2_score` (numeric)
      - `rmse` (numeric)
      - `mae` (numeric)
      - `is_best` (boolean)
      - `created_at` (timestamptz)
    
    - `forecasts`
      - `id` (uuid, primary key)
      - `crypto_id` (uuid, foreign key)
      - `model_type` (text) - ARIMA, LSTM
      - `forecast_date` (date)
      - `predicted_price` (numeric)
      - `created_at` (timestamptz)
    
    - `clustering_results`
      - `id` (uuid, primary key)
      - `crypto_id` (uuid, foreign key)
      - `algorithm` (text) - KMeans, DBSCAN, etc.
      - `optimal_clusters` (integer)
      - `silhouette_score` (numeric)
      - `is_best` (boolean)
      - `created_at` (timestamptz)

  2. Security
    - Enable RLS on all tables
    - Add policies for public read access
*/

-- Create cryptocurrencies table
CREATE TABLE IF NOT EXISTS cryptocurrencies (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  symbol text UNIQUE NOT NULL,
  name text NOT NULL,
  created_at timestamptz DEFAULT now()
);

-- Create price_history table
CREATE TABLE IF NOT EXISTS price_history (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  crypto_id uuid REFERENCES cryptocurrencies(id) ON DELETE CASCADE,
  date date NOT NULL,
  open numeric NOT NULL DEFAULT 0,
  high numeric NOT NULL DEFAULT 0,
  low numeric NOT NULL DEFAULT 0,
  close numeric NOT NULL DEFAULT 0,
  volume numeric NOT NULL DEFAULT 0,
  created_at timestamptz DEFAULT now(),
  UNIQUE(crypto_id, date)
);

-- Create regression_models table
CREATE TABLE IF NOT EXISTS regression_models (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  crypto_id uuid REFERENCES cryptocurrencies(id) ON DELETE CASCADE,
  model_name text NOT NULL,
  r2_score numeric NOT NULL DEFAULT 0,
  rmse numeric NOT NULL DEFAULT 0,
  mae numeric NOT NULL DEFAULT 0,
  is_best boolean DEFAULT false,
  created_at timestamptz DEFAULT now()
);

-- Create forecasts table
CREATE TABLE IF NOT EXISTS forecasts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  crypto_id uuid REFERENCES cryptocurrencies(id) ON DELETE CASCADE,
  model_type text NOT NULL,
  forecast_date date NOT NULL,
  predicted_price numeric NOT NULL DEFAULT 0,
  created_at timestamptz DEFAULT now()
);

-- Create clustering_results table
CREATE TABLE IF NOT EXISTS clustering_results (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  crypto_id uuid REFERENCES cryptocurrencies(id) ON DELETE CASCADE,
  algorithm text NOT NULL,
  optimal_clusters integer NOT NULL DEFAULT 0,
  silhouette_score numeric NOT NULL DEFAULT 0,
  is_best boolean DEFAULT false,
  created_at timestamptz DEFAULT now()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_price_history_crypto_date ON price_history(crypto_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_forecasts_crypto_date ON forecasts(crypto_id, forecast_date);
CREATE INDEX IF NOT EXISTS idx_regression_models_crypto ON regression_models(crypto_id);
CREATE INDEX IF NOT EXISTS idx_clustering_results_crypto ON clustering_results(crypto_id);

-- Enable RLS
ALTER TABLE cryptocurrencies ENABLE ROW LEVEL SECURITY;
ALTER TABLE price_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE regression_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE forecasts ENABLE ROW LEVEL SECURITY;
ALTER TABLE clustering_results ENABLE ROW LEVEL SECURITY;

-- Create policies for public read access
CREATE POLICY "Allow public read access to cryptocurrencies"
  ON cryptocurrencies FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Allow public read access to price_history"
  ON price_history FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Allow public read access to regression_models"
  ON regression_models FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Allow public read access to forecasts"
  ON forecasts FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Allow public read access to clustering_results"
  ON clustering_results FOR SELECT
  TO public
  USING (true);

-- Insert initial cryptocurrency data
INSERT INTO cryptocurrencies (symbol, name) 
VALUES 
  ('BTC', 'Bitcoin'),
  ('ETH', 'Ethereum'),
  ('DOGE', 'Dogecoin')
ON CONFLICT (symbol) DO NOTHING;