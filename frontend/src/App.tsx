import React from 'react';
import { ChakraProvider, CSSReset } from '@chakra-ui/react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';

// Components
import Navbar from './components/layout/Navbar';
import HomePage from './pages/HomePage';
import TokenPage from './pages/TokenPage';
import SentimentPage from './pages/SentimentPage';

// Create a client for React Query
const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ChakraProvider>
        <CSSReset />
        <Router>
          <Navbar />
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/token/:symbol" element={<TokenPage />} />
            <Route path="/sentiment" element={<SentimentPage />} />
          </Routes>
        </Router>
      </ChakraProvider>
    </QueryClientProvider>
  );
}

export default App; 