/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Absolute API origin. Leave unset in dev; the Vite proxy handles /api. */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
