import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const API_TARGET = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // KaTeX and the markdown stack are large and rarely change; splitting
        // them keeps the app chunk small and cacheable across deploys.
        manualChunks: {
          katex: ["katex", "rehype-katex"],
          markdown: ["react-markdown", "remark-gfm", "remark-math"],
        },
      },
    },
  },
  server: {
    port: 5173,
    // Proxying keeps the browser on one origin in development, so uploads and
    // the <img> tags that load source crops need no CORS handling at all.
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: true },
    },
  },
});
