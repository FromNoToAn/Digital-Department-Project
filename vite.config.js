import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      "/upload_video": process.env.VITE_API_BASE_URL,  // Прокси запросов на сервер FastAPI
    },
  },
})
