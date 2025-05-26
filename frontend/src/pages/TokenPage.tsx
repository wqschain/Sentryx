import React from 'react';
import {
  Box,
  Container,
  Heading,
  Text,
  VStack,
  Grid,
  GridItem,
  useColorModeValue,
  Spinner,
  HStack,
  Image,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
} from '@chakra-ui/react';
import { useParams } from 'react-router-dom';
import { useQuery } from 'react-query';
import {
  getToken,
  getTokenArticles,
  getTokenSentiment,
  getTokenPriceHistory,
  getTokenVolumeHistory,
} from '../services/api';
import { ArticleList } from '../components';
import { TOKEN_LOGOS } from '../constants/tokenLogos';
import PriceChart from '../components/token/PriceChart';
import VolumeChart from '../components/token/VolumeChart';
import MarketStats from '../components/token/MarketStats';
import SentimentChart from '../components/token/SentimentChart';

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

  const { data: priceHistory } = useQuery(['priceHistory', symbol], () =>
    getTokenPriceHistory(symbol!)
  );

  const { data: volumeHistory } = useQuery(['volumeHistory', symbol], () =>
    getTokenVolumeHistory(symbol!)
  );

  if (isLoadingToken) {
    return (
      <Container maxW="container.xl" centerContent py={10}>
        <Spinner size="xl" />
      </Container>
    );
  }

  return (
    <Container maxW="container.md" py={4}>
      <VStack spacing={4} align="stretch">
        <Box>
          <HStack spacing={2} mb={1}>
            <Image
              src={TOKEN_LOGOS[upperSymbol]}
              alt={`${upperSymbol} logo`}
              boxSize="24px"
              fallbackSrc="https://via.placeholder.com/24"
            />
            <Heading size="lg">
              {upperSymbol}
            </Heading>
          </HStack>
          <Text color="gray.600" fontSize="sm">
            Market Data and Sentiment Analysis
          </Text>
        </Box>

        {/* Market Stats */}
        <Box>
          <MarketStats
            price={tokenData?.data.price || 0}
            priceChange24h={tokenData?.data.price_change_24h || 0}
            marketCap={tokenData?.data.market_cap || 0}
            volume={tokenData?.data.volume || 0}
            high24h={tokenData?.data.high_24h || 0}
            low24h={tokenData?.data.low_24h || 0}
            supply={tokenData?.data.circulating_supply || 0}
            maxSupply={tokenData?.data.max_supply}
          />
        </Box>

        {/* Charts and Analysis */}
        <Grid templateColumns={{ base: '1fr', lg: '3fr 1fr' }} gap={4}>
          <GridItem>
            <VStack spacing={4} align="stretch">
              <Box p={3} shadow="sm" borderWidth="1px" borderRadius="md" bg={bgColor}>
                <Tabs>
                  <TabList>
                    <Tab fontSize="sm">Price</Tab>
                    <Tab fontSize="sm">Volume</Tab>
                  </TabList>

                  <TabPanels>
                    <TabPanel px={0}>
                      <PriceChart data={priceHistory?.data || []} />
                    </TabPanel>
                    <TabPanel px={0}>
                      <VolumeChart data={volumeHistory?.data || []} />
                    </TabPanel>
                  </TabPanels>
                </Tabs>
              </Box>

              {/* Sentiment Chart */}
              <Box p={3} shadow="sm" borderWidth="1px" borderRadius="md" bg={bgColor}>
                <Heading size="xs" mb={2}>Sentiment Analysis</Heading>
                <SentimentChart data={sentiment?.data || []} />
              </Box>

              {/* Recent Articles */}
              <Box p={3} shadow="sm" borderWidth="1px" borderRadius="md" bg={bgColor}>
                <Heading size="xs" mb={2}>Recent Articles</Heading>
                <ArticleList articles={articles?.data || []} />
              </Box>
            </VStack>
          </GridItem>

          {/* Additional Information */}
          <GridItem>
            <VStack spacing={3} align="stretch">
              <Box p={3} shadow="sm" borderWidth="1px" borderRadius="md" bg={bgColor}>
                <Heading size="xs" mb={2}>Market Overview</Heading>
                <VStack align="stretch" spacing={2}>
                  <HStack justify="space-between">
                    <Text color="gray.600" fontSize="xs">Market Rank</Text>
                    <Text fontWeight="bold" fontSize="sm">#{tokenData?.data.market_rank || 'N/A'}</Text>
                  </HStack>
                  <HStack justify="space-between">
                    <Text color="gray.600" fontSize="xs">All-Time High</Text>
                    <Text fontWeight="bold" fontSize="sm">${tokenData?.data.ath?.toLocaleString()}</Text>
                  </HStack>
                </VStack>
              </Box>
            </VStack>
          </GridItem>
        </Grid>
      </VStack>
    </Container>
  );
};

export default TokenPage; 