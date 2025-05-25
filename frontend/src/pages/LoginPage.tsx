import React, { useState } from 'react';
import {
  Box,
  Button,
  Container,
  FormControl,
  FormLabel,
  Heading,
  Input,
  Stack,
  Text,
  Link as ChakraLink,
  useToast,
} from '@chakra-ui/react';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import { useMutation } from 'react-query';
import { login } from '../services/api';
import { useAuth } from '../context/AuthContext';

const LoginPage = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { login: authLogin } = useAuth();
  const navigate = useNavigate();
  const toast = useToast();

  const mutation = useMutation(
    (credentials: { email: string; password: string }) =>
      login(credentials.email, credentials.password),
    {
      onSuccess: (data) => {
        authLogin(data.data.access_token);
        toast({
          title: 'Login successful',
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
        navigate('/');
      },
      onError: () => {
        toast({
          title: 'Login failed',
          description: 'Invalid email or password',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
      },
    }
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    mutation.mutate({ email, password });
  };

  return (
    <Container maxW="container.sm" py={10}>
      <Box
        p={8}
        borderWidth={1}
        borderRadius="lg"
        boxShadow="lg"
      >
        <Stack spacing={4}>
          <Heading size="lg" textAlign="center">
            Login to Sentryx
          </Heading>
          <Box as="form" onSubmit={handleSubmit}>
            <Stack spacing={4}>
              <FormControl isRequired>
                <FormLabel>Email</FormLabel>
                <Input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </FormControl>
              <FormControl isRequired>
                <FormLabel>Password</FormLabel>
                <Input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </FormControl>
              <Button
                type="submit"
                colorScheme="blue"
                size="lg"
                fontSize="md"
                isLoading={mutation.isLoading}
              >
                Sign in
              </Button>
            </Stack>
          </Box>
          <Text textAlign="center" mt={4}>
            Don't have an account?{' '}
            <ChakraLink as={RouterLink} to="/register" color="blue.500">
              Register
            </ChakraLink>
          </Text>
        </Stack>
      </Box>
    </Container>
  );
};

export default LoginPage; 