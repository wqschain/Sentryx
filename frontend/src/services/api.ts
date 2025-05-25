import axios from 'axios';

// Default to localhost:8000 if environment variable is not set
const API_URL = import.meta.env.VITE_API_URL || '/api';

// Create axios instance
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Token endpoints
export const getTokens = () => api.get('/token');
export const getToken = (symbol: string) => api.get(`/token/${symbol}`);
export const getTokenArticles = (symbol: string) => api.get(`/token/${symbol}/articles`);
export const getTokenSentiment = (symbol: string) => api.get(`/token/${symbol}/sentiment`);

// Sentiment analysis endpoint
export const analyzeSentiment = (text: string) =>
  api.post('/sentiment/analyze', { text });

export default api; 