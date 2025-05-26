import React, { useState } from 'react';
import {
  Box,
  Button,
  Text,
  Textarea,
  VStack,
  Heading,
  Flex,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Card,
  CardBody,
  Badge,
  useToast,
  Divider,
  Progress,
  SimpleGrid,
  Icon,
  HStack,
  Tooltip,
} from '@chakra-ui/react';
import { FaThumbsUp, FaThumbsDown, FaMinus } from 'react-icons/fa';
import { analyzeSentiment } from '../services/api';
import { SentimentResult } from '../types/sentiment';

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
      
      if (!response.data.is_relevant && response.data.feedback) {
        toast({
          title: 'Content Not Relevant',
          description: response.data.feedback,
          status: 'warning',
          duration: 5000,
          isClosable: true,
        });
      } else {
        toast({
          title: 'Analysis Complete',
          description: 'Sentiment analysis has been completed successfully.',
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      const errorResponse = err.response?.data;
      setError(errorMessage);
      
      toast({
        title: 'Error',
        description: errorResponse?.feedback || errorMessage,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      
      if (errorResponse?.feedback) {
        setResult({
          sentiment: 'neutral',
          confidence: 0,
          is_relevant: false,
          relevance_score: 0,
          relevance_explanation: errorResponse.feedback,
          matched_terms: {},
          feedback: errorResponse.feedback
        });
      }
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

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case 'positive':
        return FaThumbsUp;
      case 'negative':
        return FaThumbsDown;
      default:
        return FaMinus;
    }
  };

  const getRelevanceColor = (score: number) => {
    if (score >= 0.8) return 'green';
    if (score >= 0.6) return 'yellow';
    if (score >= 0.4) return 'orange';
    return 'red';
  };

  const getCategoryLabel = (category: string) => {
    switch (category) {
      case 'crypto':
        return 'Cryptocurrency Terms';
      case 'context':
        return 'Context Terms';
      case 'analysis':
        return 'Analysis Terms';
      default:
        return category.charAt(0).toUpperCase() + category.slice(1);
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'crypto':
        return 'blue';
      case 'context':
        return 'purple';
      case 'analysis':
        return 'teal';
      default:
        return 'gray';
    }
  };

  return (
    <Box maxW="700px" mx="auto" p={4}>
      <VStack spacing={4} align="stretch">
        <Flex align="center" mb={1}>
          <Heading 
            as="h1" 
            size="md" 
            bgGradient="linear(to-r, blue.500, cyan.500)"
            bgClip="text"
          >
            SentryxAI
          </Heading>
          <Text fontSize="xs" color="gray.500" ml={2} mt={1}>
            Crypto Sentiment Analyzer
          </Text>
        </Flex>
        
        <Text fontSize="sm">
          Enter any crypto-related text to analyze its sentiment and relevance. The analyzer will determine if the content is relevant to cryptocurrency and analyze its sentiment.
        </Text>

        <Textarea
          placeholder="Enter your text here (e.g., 'Bitcoin's recent price movement shows strong potential for growth')"
          value={text}
          onChange={(e) => setText(e.target.value)}
          isDisabled={isAnalyzing}
          minH="150px"
          p={3}
          resize="vertical"
          borderRadius="md"
          fontSize="sm"
          _focus={{
            borderColor: 'blue.400',
            boxShadow: '0 0 0 1px var(--chakra-colors-blue-400)',
          }}
        />

        {error && (
          <Alert status="error" borderRadius="md" size="sm">
            <AlertIcon />
            <AlertDescription fontSize="sm">{error}</AlertDescription>
          </Alert>
        )}

        <Flex justify="flex-end">
          <Button
            colorScheme="blue"
            onClick={handleAnalyzeSentiment}
            isLoading={isAnalyzing}
            loadingText="Analyzing"
            size="md"
            minW="100px"
            isDisabled={!text.trim()}
            _hover={{
              transform: 'translateY(-1px)',
              boxShadow: 'sm',
            }}
            transition="all 0.2s"
          >
            Analyze
          </Button>
        </Flex>

        {result && (
          <Card variant="elevated" borderRadius="lg" size="sm">
            <CardBody py={3} px={4}>
              <VStack spacing={4} align="stretch">
                <Flex justify="space-between" align="center">
                  <Heading size="sm">Analysis Results</Heading>
                  <Badge 
                    colorScheme={result.is_relevant ? 'green' : 'red'}
                    fontSize="xs"
                    px={2}
                    py={0.5}
                    borderRadius="full"
                  >
                    {result.is_relevant ? 'RELEVANT' : 'NOT RELEVANT'}
                  </Badge>
                </Flex>

                <Divider />

                {!result.is_relevant && result.feedback && (
                  <Alert status="warning" variant="left-accent" borderRadius="md" py={2} px={3}>
                    <AlertIcon boxSize="16px" />
                    <Box>
                      <AlertTitle fontWeight="bold" fontSize="sm">Feedback</AlertTitle>
                      <AlertDescription fontSize="sm">{result.feedback}</AlertDescription>
                    </Box>
                  </Alert>
                )}

                {result.is_relevant && (
                  <>
                    <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                      <Box p={3} borderRadius="md" borderWidth="1px">
                        <VStack align="stretch" spacing={3}>
                          <Flex align="center" justify="space-between">
                            <Text fontWeight="semibold" fontSize="sm">Sentiment</Text>
                            <HStack>
                              <Icon 
                                as={getSentimentIcon(result.sentiment)}
                                color={getSentimentColor(result.sentiment)}
                                boxSize={4}
                              />
                              <Text
                                fontWeight="bold"
                                fontSize="sm"
                                color={getSentimentColor(result.sentiment)}
                              >
                                {result.sentiment.toUpperCase()}
                              </Text>
                            </HStack>
                          </Flex>
                          <Box>
                            <Text mb={1} fontSize="xs">Confidence</Text>
                            <Tooltip label={`${(result.confidence * 100).toFixed(1)}%`}>
                              <Progress
                                value={result.confidence * 100}
                                colorScheme={result.confidence > 0.7 ? 'green' : 'orange'}
                                borderRadius="full"
                                size="xs"
                              />
                            </Tooltip>
                          </Box>
                        </VStack>
                      </Box>

                      <Box p={3} borderRadius="md" borderWidth="1px">
                        <VStack align="stretch" spacing={3}>
                          <Text fontWeight="semibold" fontSize="sm">Relevance Score</Text>
                          <Box>
                            <Tooltip label={`${(result.relevance_score * 100).toFixed(1)}%`}>
                              <Progress
                                value={result.relevance_score * 100}
                                colorScheme={getRelevanceColor(result.relevance_score)}
                                borderRadius="full"
                                size="xs"
                              />
                            </Tooltip>
                          </Box>
                        </VStack>
                      </Box>
                    </SimpleGrid>

                    {result.matched_terms && Object.entries(result.matched_terms).length > 0 && (
                      <Box>
                        <Text fontWeight="semibold" fontSize="sm" mb={2}>Detected Terms</Text>
                        <SimpleGrid columns={{ base: 1, md: 2 }} spacing={3}>
                          {Object.entries(result.matched_terms)
                            .filter(([category, terms]) => terms.length > 0 && category !== 'manipulation')
                            .map(([category, terms]) => (
                              <Box 
                                key={category}
                                p={2}
                                borderRadius="md"
                                borderWidth="1px"
                                bg={`${getCategoryColor(category)}.50`}
                              >
                                <Text fontWeight="semibold" fontSize="xs" mb={1.5} color={`${getCategoryColor(category)}.700`}>
                                  {getCategoryLabel(category)}
                                </Text>
                                <Flex wrap="wrap" gap={1.5}>
                                  {terms.map((term, i) => (
                                    <Badge
                                      key={`${category}-${i}`}
                                      colorScheme={getCategoryColor(category)}
                                      borderRadius="full"
                                      px={2}
                                      py={0.5}
                                      fontSize="xs"
                                    >
                                      {term}
                                    </Badge>
                                  ))}
                                </Flex>
                              </Box>
                          ))}
                        </SimpleGrid>
                      </Box>
                    )}

                    {result.relevance_explanation && (
                      <Box>
                        <Text fontSize="xs" color="gray.600">
                          {result.relevance_explanation}
                        </Text>
                      </Box>
                    )}
                  </>
                )}
              </VStack>
            </CardBody>
          </Card>
        )}
      </VStack>
    </Box>
  );
};

export default SentimentAnalyzer; 