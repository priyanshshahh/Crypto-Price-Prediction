# Deploy (Vercel)

Static Vite dashboard. Data comes from `public/data/crypto_data.json` produced by
`pipeline.py`. No backend required for the default deploy.

## Setup

```bash
npm i -g vercel   # or: brew install vercel-cli
vercel login
```

## Deploy

```bash
.venv/bin/python pipeline.py --mode production   # refresh JSON (optional but recommended)
npm run build
vercel            # preview
vercel --prod     # production
```

`vercel.json` configures the Vite build output.

## Optional Supabase

Set `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` in Vercel env vars and run
migrations under `supabase/migrations/`. Without them, the app uses static JSON.

## Updating forecasts on the live site

```bash
.venv/bin/python pipeline.py --mode production
vercel --prod
```
