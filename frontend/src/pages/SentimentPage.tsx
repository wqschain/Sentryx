import React from 'react';
import { Container } from '@chakra-ui/react';
import SentimentAnalyzer from '../components/SentimentAnalyzer';

const SentimentPage: React.FC = () => {
  return (
    <Container maxW="container.xl" py={8}>
      <SentimentAnalyzer />
    </Container>
  );
};

export default SentimentPage; 