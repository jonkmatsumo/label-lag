import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    proxy: {
      '/bff': {
        target: process.env.VITE_BFF_BASE_URL || 'http://localhost:3210',
        changeOrigin: true,
      },
    },
  },
})
