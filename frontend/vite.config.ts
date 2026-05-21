import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite config — dev server runs on :5173 by default, which is in the
// backend's CORS allow-list out of the box.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,
  },
});
