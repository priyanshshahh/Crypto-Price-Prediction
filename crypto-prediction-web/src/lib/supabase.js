import { createClient } from '@supabase/supabase-js';
import { getMockClient } from './mockSupabase';

const supabaseUrl     = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

// Supabase is considered "configured" only when both env vars are present and
// look like real values (not the placeholder strings from the README).
const isConfigured =
  supabaseUrl &&
  supabaseAnonKey &&
  supabaseUrl.startsWith('https://') &&
  !supabaseUrl.includes('your-project');

const _realClient = isConfigured
  ? createClient(supabaseUrl, supabaseAnonKey)
  : null;

/**
 * Returns a Supabase-compatible client.
 * Uses the real Supabase client when credentials are configured, otherwise
 * falls back to a mock client backed by public/data/crypto_data.json.
 */
export async function getClient() {
  if (_realClient) return _realClient;
  return getMockClient();
}

// Convenience re-export for components that still import `supabase` directly
// (real client only; components should use getClient() instead).
export const supabase = _realClient;
