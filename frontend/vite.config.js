import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // WebSocket endpoint - must come before /api to match first
      "/api/ws": {
        target: "ws://localhost:8000",
        ws: true,
        changeOrigin: true,
      },
      // REST API endpoints
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
