import React from 'react';
import {
  Box,
  Flex,
  Link,
  useColorModeValue,
  Container,
  Button,
} from '@chakra-ui/react';
import { Link as RouterLink } from 'react-router-dom';

const Navbar = () => {
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  return (
    <Box bg={bgColor} borderBottom="1px" borderColor={borderColor}>
      <Container maxW="container.xl">
        <Flex h={16} alignItems="center" justifyContent="space-between">
          <Flex alignItems="center">
            <Link as={RouterLink} to="/" fontSize="xl" fontWeight="bold" mr={8}>
              Sentryx
            </Link>
            <Link as={RouterLink} to="/" mr={4}>
              Tokens
            </Link>
            <Link as={RouterLink} to="/sentiment">
              Sentiment Analysis
            </Link>
          </Flex>
        </Flex>
      </Container>
    </Box>
  );
};

export default Navbar; 