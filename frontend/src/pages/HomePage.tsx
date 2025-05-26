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
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
} from '@chakra-ui/react';
import { Link as RouterLink } from 'react-router-dom';
import { useQuery } from 'react-query';
import { getTokens } from '../services/api';
import { TOKEN_LOGOS } from '../constants/tokenLogos';
import { keyframes } from '@emotion/react';

const gradientAnimation = keyframes`
  0% { background-position: 0% center; }
  100% { background-position: -200% center; }
`;

interface TokenData {
  symbol: string;
  price: number;
  market_cap: number;
  volume: number;
}

const TokenCard: React.FC<{ token: TokenData }> = ({ token }) => {
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const statBg = useColorModeValue('gray.50', 'gray.700');

  return (
    <Box
      as={RouterLink}
      to={`/token/${token.symbol.toLowerCase()}`}
      p={3}
      shadow="sm"
      borderWidth="1px"
      borderRadius="md"
      bg={bgColor}
      borderColor={borderColor}
      _hover={{
        transform: 'translateY(-1px)',
        shadow: 'md',
        borderColor: useColorModeValue('blue.200', 'blue.700'),
        transition: 'all 0.2s'
      }}
    >
      <VStack align="stretch" spacing={2}>
        <HStack justify="space-between" align="center">
          <HStack spacing={2}>
            <Image
              src={TOKEN_LOGOS[token.symbol]}
              alt={`${token.symbol} logo`}
              boxSize="20px"
              fallbackSrc="https://via.placeholder.com/20"
            />
            <Text fontWeight="bold" fontSize="sm">{token.symbol}</Text>
          </HStack>
          <Badge 
            colorScheme="green" 
            fontSize="xs" 
            px={2} 
            py={0.5} 
            borderRadius="full"
          >
            ${token.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </Badge>
        </HStack>
        <SimpleGrid columns={2} spacing={2} pt={1}>
          <Box bg={statBg} p={2} borderRadius="md">
            <Text fontSize="xs" color="gray.500">Market Cap</Text>
            <Text fontSize="sm" fontWeight="medium">
              ${(token.market_cap / 1e9).toFixed(2)}B
            </Text>
          </Box>
          <Box bg={statBg} p={2} borderRadius="md">
            <Text fontSize="xs" color="gray.500">24h Volume</Text>
            <Text fontSize="sm" fontWeight="medium">
              ${(token.volume / 1e6).toFixed(1)}M
            </Text>
          </Box>
        </SimpleGrid>
      </VStack>
    </Box>
  );
};

const HomePage = () => {
  const { data: tokens, isLoading, error } = useQuery('tokens', getTokens);

  if (isLoading) {
    return (
      <Container maxW="container.lg" centerContent py={6}>
        <Spinner size="lg" />
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxW="container.lg" centerContent py={6}>
        <Text color="red.500">Error loading tokens</Text>
      </Container>
    );
  }

  return (
    <Container maxW="container.lg" py={6}>
      <VStack spacing={6} align="stretch">
        <Box textAlign="center">
          <Heading 
            size="lg" 
            mb={2}
          >
            <Box
              as="span"
              bgGradient="linear(to-r, blue.400, cyan.400, blue.400, cyan.400, blue.400, cyan.400)"
              bgClip="text"
              bgSize="200% auto"
              animation={`${gradientAnimation} 3s linear infinite`}
              display="inline-block"
            >
              Sentryx
            </Box>
            {' Dashboard'}
          </Heading>
          <Text color="gray.500" fontSize="sm">
            Track sentiment and market data for major cryptocurrencies
          </Text>
        </Box>

        <SimpleGrid columns={{ base: 1, sm: 2, md: 3 }} spacing={4}>
          {tokens?.data.map((token: TokenData) => (
            <TokenCard key={token.symbol} token={token} />
          ))}
        </SimpleGrid>
      </VStack>
    </Container>
  );
};

export default HomePage; 