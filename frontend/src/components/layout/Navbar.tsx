import React from 'react';
import {
  Box,
  Flex,
  Link,
  useColorModeValue,
  Container,
  Text,
} from '@chakra-ui/react';
import { Link as RouterLink } from 'react-router-dom';
import { keyframes } from '@emotion/react';

const gradientAnimation = keyframes`
  0% { background-position: 0% center; }
  100% { background-position: -200% center; }
`;

const NavLink = ({ to, children }: { to: string; children: React.ReactNode }) => {
  const gradientBg = useColorModeValue(
    'linear-gradient(to right, rgba(49, 130, 206, 0.08), rgba(49, 206, 206, 0.08))',
    'linear-gradient(to right, rgba(49, 130, 206, 0.15), rgba(49, 206, 206, 0.15))'
  );
  
  return (
    <Link
      as={RouterLink}
      to={to}
      px={2}
      py={1}
      rounded="md"
      position="relative"
      fontSize="sm"
      fontWeight="medium"
      _hover={{
        textDecoration: 'none',
        background: gradientBg,
        transform: 'translateY(-1px)',
        _after: {
          width: '100%',
          opacity: 1,
        }
      }}
      _after={{
        content: '""',
        position: 'absolute',
        bottom: '-1px',
        left: '0',
        width: '0%',
        height: '2px',
        bgGradient: 'linear(to-r, blue.400, cyan.400)',
        opacity: 0,
        transition: 'all 0.3s ease-in-out',
      }}
      transition="all 0.2s"
    >
      {children}
    </Link>
  );
};

const Navbar = () => {
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  return (
    <Box bg={bgColor} borderBottom="1px" borderColor={borderColor}>
      <Container maxW="container.xl">
        <Flex h={14} alignItems="center" justifyContent="space-between">
          <Flex alignItems="center" gap={8}>
            <Link
              as={RouterLink}
              to="/"
              fontSize="xl"
              fontWeight="bold"
              bgGradient="linear(to-r, blue.500, cyan.500)"
              bgClip="text"
              _hover={{
                textDecoration: 'none',
                bgGradient: 'linear(to-r, blue.400, cyan.400)',
                transform: 'translateY(-1px)',
              }}
              transition="all 0.2s"
            >
              Sentryx
            </Link>
            <Flex gap={4}>
              <NavLink to="/">Tokens</NavLink>
              <NavLink to="/sentiment">SentryxAI</NavLink>
            </Flex>
          </Flex>
        </Flex>
      </Container>
    </Box>
  );
};

export default Navbar; 