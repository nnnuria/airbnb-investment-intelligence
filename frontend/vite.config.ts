import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

// FastAPI runs on :8000 (CORS is open in dev). The app talks to it directly via
// VITE_API_BASE_URL; no proxy needed. Alias "@" -> ./src.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    strictPort: false,
  },
});
