import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  build: { emptyOutDir: true },
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: { "/api/gui": "http://127.0.0.1:8770" },
  },
  test: { environment: "node", include: ["tests/**/*.test.ts"] },
});
