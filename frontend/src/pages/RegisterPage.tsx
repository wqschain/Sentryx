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
import { register } from '../services/api';

const RegisterPage = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const navigate = useNavigate();
  const toast = useToast();

  const mutation = useMutation(
    (credentials: { email: string; password: string }) =>
      register(credentials.email, credentials.password),
    {
      onSuccess: () => {
        toast({
          title: 'Registration successful',
          description: 'Please login with your credentials',
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
        navigate('/login');
      },
      onError: () => {
        toast({
          title: 'Registration failed',
          description: 'Email might be already registered',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
      },
    }
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (password !== confirmPassword) {
      toast({
        title: 'Error',
        description: 'Passwords do not match',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

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
            Create an Account
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
              <FormControl isRequired>
                <FormLabel>Confirm Password</FormLabel>
                <Input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                />
              </FormControl>
              <Button
                type="submit"
                colorScheme="blue"
                size="lg"
                fontSize="md"
                isLoading={mutation.isLoading}
              >
                Register
              </Button>
            </Stack>
          </Box>
          <Text textAlign="center" mt={4}>
            Already have an account?{' '}
            <ChakraLink as={RouterLink} to="/login" color="blue.500">
              Login
            </ChakraLink>
          </Text>
        </Stack>
      </Box>
    </Container>
  );
};

export default RegisterPage; 