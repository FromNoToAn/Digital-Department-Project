import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      "/upload": "http://127.0.0.1:8000",  // Прокси запросов на сервер FastAPI
    },
  },
})
