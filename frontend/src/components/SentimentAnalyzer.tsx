import React, { useState } from 'react';
import {
  Box,
  Button,
  Text,
  Textarea,
  VStack,
  Heading,
  Flex,
  Spinner,
  Card,
  CardBody,
  useToast,
} from '@chakra-ui/react';
import { analyzeSentiment } from '../services/api';

interface SentimentResult {
  sentiment: string;
  confidence: number;
}

const SentimentAnalyzer: React.FC = () => {
  const [text, setText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const toast = useToast();

  const handleAnalyzeSentiment = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const response = await analyzeSentiment(text.trim());
      setResult(response.data);
      toast({
        title: 'Analysis Complete',
        description: 'Sentiment analysis has been completed successfully.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(errorMessage);
      toast({
        title: 'Error',
        description: errorMessage,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case 'positive':
        return 'green.500';
      case 'negative':
        return 'red.500';
      case 'neutral':
        return 'orange.500';
      default:
        return 'gray.500';
    }
  };

  return (
    <Box maxW="800px" mx="auto" p={6}>
      <VStack spacing={6} align="stretch">
        <Heading as="h1" size="lg">
          Crypto Sentiment Analyzer
        </Heading>
        
        <Text>
          Enter any crypto-related text to analyze its sentiment. The analyzer will determine if the sentiment is positive, negative, or neutral.
        </Text>

        <Textarea
          placeholder="Enter your text here (e.g., 'Bitcoin's recent price movement shows strong potential for growth')"
          value={text}
          onChange={(e) => setText(e.target.value)}
          isDisabled={isAnalyzing}
          minH="200px"
          p={4}
          resize="vertical"
        />

        {error && (
          <Text color="red.500" fontSize="sm">
            Error: {error}
          </Text>
        )}

        <Flex justify="flex-end">
          <Button
            colorScheme="blue"
            onClick={handleAnalyzeSentiment}
            isLoading={isAnalyzing}
            loadingText="Analyzing"
            minW="120px"
            isDisabled={!text.trim()}
          >
            Analyze
          </Button>
        </Flex>

        {result && (
          <Card variant="elevated">
            <CardBody>
              <VStack spacing={4} align="stretch">
                <Heading size="md">
                  Analysis Results
                </Heading>
                
                <Flex align="center">
                  <Text mr={2}>Sentiment:</Text>
                  <Text
                    fontWeight="bold"
                    color={getSentimentColor(result.sentiment)}
                  >
                    {result.sentiment.toUpperCase()}
                  </Text>
                </Flex>

                <Flex align="center">
                  <Text mr={2}>Confidence:</Text>
                  <Text fontWeight="semibold">
                    {(result.confidence * 100).toFixed(1)}%
                  </Text>
                </Flex>
              </VStack>
            </CardBody>
          </Card>
        )}
      </VStack>
    </Box>
  );
};

export default SentimentAnalyzer; 