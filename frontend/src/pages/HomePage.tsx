import React from 'react';
import {
  Box,
  Container,
  Heading,
  SimpleGrid,
  Text,
  VStack,
  HStack,
  Badge,
  useColorModeValue,
  Spinner,
  Image,
} from '@chakra-ui/react';
import { Link as RouterLink } from 'react-router-dom';
import { useQuery } from 'react-query';
import { getTokens } from '../services/api';
import { TOKEN_LOGOS } from '../constants/tokenLogos';

interface TokenData {
  symbol: string;
  price: number;
  market_cap: number;
  volume: number;
}

const TokenCard: React.FC<{ token: TokenData }> = ({ token }) => {
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  return (
    <Box
      as={RouterLink}
      to={`/token/${token.symbol.toLowerCase()}`}
      p={5}
      shadow="md"
      borderWidth="1px"
      borderRadius="lg"
      bg={bgColor}
      borderColor={borderColor}
      _hover={{ transform: 'translateY(-2px)', transition: 'all 0.2s' }}
    >
      <VStack align="start" spacing={3}>
        <HStack justify="space-between" width="100%">
          <HStack spacing={2}>
            <Image
              src={TOKEN_LOGOS[token.symbol]}
              alt={`${token.symbol} logo`}
              boxSize="24px"
              fallbackSrc="https://via.placeholder.com/24"
            />
            <Heading size="md">{token.symbol}</Heading>
          </HStack>
          <Badge colorScheme="green" fontSize="sm">
            ${token.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </Badge>
        </HStack>
        <Text color="gray.500" fontSize="sm">
          Market Cap: ${token.market_cap.toLocaleString()}
        </Text>
        <Text color="gray.500" fontSize="sm">
          24h Volume: ${token.volume.toLocaleString()}
        </Text>
      </VStack>
    </Box>
  );
};

const HomePage = () => {
  const { data: tokens, isLoading, error } = useQuery('tokens', getTokens);

  if (isLoading) {
    return (
      <Container maxW="container.xl" centerContent py={10}>
        <Spinner size="xl" />
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxW="container.xl" centerContent py={10}>
        <Text color="red.500">Error loading tokens</Text>
      </Container>
    );
  }

  return (
    <Container maxW="container.xl" py={10}>
      <VStack spacing={8} align="stretch">
        <Box textAlign="center">
          <Heading size="2xl" mb={2}>
            Sentryx – Crypto Sentiment Dashboard
          </Heading>
          <Text color="gray.600" fontSize="lg">
            Track sentiment and market data for major cryptocurrencies
          </Text>
        </Box>

        <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
          {tokens?.data.map((token: TokenData) => (
            <TokenCard key={token.symbol} token={token} />
          ))}
        </SimpleGrid>
      </VStack>
    </Container>
  );
};

export default HomePage; 