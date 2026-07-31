import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    // Dashboard + Recharts produce a large main chunk; raise the warn floor.
    chunkSizeWarningLimit: 1000,
  },
});
