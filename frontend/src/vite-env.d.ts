/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string
  // Add other env variables here if needed
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

// Expose environment variables on process.env as well
declare global {
  interface Window {
    env: ImportMetaEnv
  }
}

export {} 