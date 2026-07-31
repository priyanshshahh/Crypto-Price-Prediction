# Data sources

**Last updated:** 2026-07-31

All production features use **free or free-tier** APIs. Missing series degrade gracefully.

## Market data

| Source | Use | Key |
|--------|-----|-----|
| Coinbase Exchange public API | Spot OHLCV (primary) | None |
| Kraken public OHLC | Fallback OHLCV | None |
| Binance Vision archives | Perp funding history | None |

## Sentiment & search

| Source | Use | Key |
|--------|-----|-----|
| alternative.me | Fear & Greed index | None |
| Google Trends (`pytrends`) | Search interest | None |

## Macro

| Source | Use | Key |
|--------|-----|-----|
| FRED | Dollar, 10Y, VIX, fed funds | `FRED_API_KEY` (free) |

## On-chain & network (proxies)

| Source | Use | Key |
|--------|-----|-----|
| mempool.space | BTC hashrate | None |
| blockchain.info charts | BTC tx count | None |
| DefiLlama | ETH chain TVL; Lido APY proxy | None |
| Etherscan | ETH gas snapshot | `ETHERSCAN_API_KEY` (free) |
| CoinGecko / CMC trial | Dominance / mcap proxies | Optional |

## Prediction markets

| Source | Use | Key |
|--------|-----|-----|
| Polymarket Gamma API | Crowd Yes probability snapshot | None |

## Local files

```bash
cp .env.example .env.local
# fill FRED_API_KEY, optional COINGECKO_API_KEY, ETHERSCAN_API_KEY
```

Raw pulls cache under `data/raw/` (gitignored, ~24h freshness).

## What we do not require

Paid institutional on-chain APIs (labeled whales, exchange netflow time series, MVRV/SOPR). Those remain optional future upgrades if free proxies plateau.
