import React from 'react';
import {
  Box,
  Container,
  Heading,
  Text,
  VStack,
  Grid,
  GridItem,
  Stat,
  StatLabel,
  StatNumber,
  useColorModeValue,
  Spinner,
  HStack,
  Image,
} from '@chakra-ui/react';
import { useParams } from 'react-router-dom';
import { useQuery } from 'react-query';
import {
  getToken,
  getTokenArticles,
  getTokenSentiment,
} from '../services/api';
import { ArticleList, SentimentChart } from '../components';
import { TOKEN_LOGOS } from '../constants/tokenLogos';

const TokenPage = () => {
  const { symbol } = useParams<{ symbol: string }>();
  const bgColor = useColorModeValue('white', 'gray.800');
  const upperSymbol = symbol?.toUpperCase() || '';

  const { data: tokenData, isLoading: isLoadingToken } = useQuery(
    ['token', symbol],
    () => getToken(symbol!)
  );

  const { data: articles } = useQuery(['articles', symbol], () =>
    getTokenArticles(symbol!)
  );

  const { data: sentiment } = useQuery(['sentiment', symbol], () =>
    getTokenSentiment(symbol!)
  );

  if (isLoadingToken) {
    return (
      <Container maxW="container.xl" centerContent py={10}>
        <Spinner size="xl" />
      </Container>
    );
  }

  return (
    <Container maxW="container.xl" py={10}>
      <VStack spacing={8} align="stretch">
        <Box>
          <HStack spacing={3} mb={2}>
            <Image
              src={TOKEN_LOGOS[upperSymbol]}
              alt={`${upperSymbol} logo`}
              boxSize="40px"
              fallbackSrc="https://via.placeholder.com/40"
            />
            <Heading size="2xl">
              {upperSymbol}
            </Heading>
          </HStack>
          <Text color="gray.600" fontSize="lg">
            Market Data and Sentiment Analysis
          </Text>
        </Box>

        <Grid templateColumns={{ base: '1fr', lg: '2fr 1fr' }} gap={8}>
          <GridItem>
            <VStack spacing={8} align="stretch">
              {/* Market Stats */}
              <Box p={5} shadow="md" borderWidth="1px" borderRadius="lg" bg={bgColor}>
                <Grid templateColumns="repeat(3, 1fr)" gap={4}>
                  <Stat>
                    <StatLabel>Price</StatLabel>
                    <StatNumber>
                      ${tokenData?.data.price.toLocaleString()}
                    </StatNumber>
                  </Stat>
                  <Stat>
                    <StatLabel>Market Cap</StatLabel>
                    <StatNumber>
                      ${tokenData?.data.market_cap.toLocaleString()}
                    </StatNumber>
                  </Stat>
                  <Stat>
                    <StatLabel>24h Volume</StatLabel>
                    <StatNumber>
                      ${tokenData?.data.volume.toLocaleString()}
                    </StatNumber>
                  </Stat>
                </Grid>
              </Box>

              {/* Sentiment Chart */}
              <Box p={5} shadow="md" borderWidth="1px" borderRadius="lg" bg={bgColor}>
                <Heading size="md" mb={4}>Sentiment Analysis</Heading>
                <SentimentChart data={sentiment?.data || []} />
              </Box>

              {/* Recent Articles */}
              <Box p={5} shadow="md" borderWidth="1px" borderRadius="lg" bg={bgColor}>
                <Heading size="md" mb={4}>Recent Articles</Heading>
                <ArticleList articles={articles?.data || []} />
              </Box>
            </VStack>
          </GridItem>
        </Grid>
      </VStack>
    </Container>
  );
};

export default TokenPage; 