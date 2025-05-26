import React from 'react';
import {
  SimpleGrid,
  Box,
  Stat,
  StatLabel,
  StatNumber,
  StatArrow,
  StatGroup,
  useColorModeValue,
  Text,
} from '@chakra-ui/react';

interface MarketStatsProps {
  price: number;
  priceChange24h: number;
  marketCap: number;
  volume: number;
  high24h: number;
  low24h: number;
  supply: number;
  maxSupply?: number;
}

const MarketStats: React.FC<MarketStatsProps> = ({
  price,
  priceChange24h,
  marketCap,
  volume,
  high24h,
  low24h,
  supply,
  maxSupply,
}) => {
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  const formatNumber = (value: number) => {
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
    return `$${value.toLocaleString()}`;
  };

  const formatSupply = (value: number) => {
    if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
    return value.toLocaleString();
  };

  return (
    <SimpleGrid columns={{ base: 2, md: 4 }} spacing={2}>
      <Box p={2} shadow="sm" borderWidth="1px" borderRadius="sm" bg={bgColor} borderColor={borderColor}>
        <Stat>
          <StatLabel fontSize="xs">Current Price</StatLabel>
          <StatNumber fontSize="sm">${price.toLocaleString()}</StatNumber>
          <StatArrow type={priceChange24h >= 0 ? 'increase' : 'decrease'} />
          <Text fontSize="xs">{Math.abs(priceChange24h).toFixed(2)}%</Text>
        </Stat>
      </Box>

      <Box p={2} shadow="sm" borderWidth="1px" borderRadius="sm" bg={bgColor} borderColor={borderColor}>
        <Stat>
          <StatLabel fontSize="xs">Market Cap</StatLabel>
          <StatNumber fontSize="sm">{formatNumber(marketCap)}</StatNumber>
        </Stat>
      </Box>

      <Box p={2} shadow="sm" borderWidth="1px" borderRadius="sm" bg={bgColor} borderColor={borderColor}>
        <StatGroup>
          <Stat>
            <StatLabel fontSize="xs">24h High</StatLabel>
            <StatNumber fontSize="sm">${high24h.toLocaleString()}</StatNumber>
          </Stat>
        </StatGroup>
      </Box>

      <Box p={2} shadow="sm" borderWidth="1px" borderRadius="sm" bg={bgColor} borderColor={borderColor}>
        <StatGroup>
          <Stat>
            <StatLabel fontSize="xs">24h Low</StatLabel>
            <StatNumber fontSize="sm">${low24h.toLocaleString()}</StatNumber>
          </Stat>
        </StatGroup>
      </Box>

      <Box p={2} shadow="sm" borderWidth="1px" borderRadius="sm" bg={bgColor} borderColor={borderColor}>
        <Stat>
          <StatLabel fontSize="xs">24h Volume</StatLabel>
          <StatNumber fontSize="sm">{formatNumber(volume)}</StatNumber>
        </Stat>
      </Box>

      <Box p={2} shadow="sm" borderWidth="1px" borderRadius="sm" bg={bgColor} borderColor={borderColor}>
        <Stat>
          <StatLabel fontSize="xs">Circulating Supply</StatLabel>
          <StatNumber fontSize="sm">{formatSupply(supply)}</StatNumber>
        </Stat>
      </Box>

      {maxSupply && (
        <Box p={2} shadow="sm" borderWidth="1px" borderRadius="sm" bg={bgColor} borderColor={borderColor}>
          <Stat>
            <StatLabel fontSize="xs">Max Supply</StatLabel>
            <StatNumber fontSize="sm">{formatSupply(maxSupply)}</StatNumber>
          </Stat>
        </Box>
      )}
    </SimpleGrid>
  );
};

export default MarketStats; 