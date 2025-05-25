import React from 'react';
import {
  VStack,
  Box,
  Heading,
  Text,
  Link,
  Badge,
  useColorModeValue,
} from '@chakra-ui/react';

interface Article {
  id: number;
  title: string;
  url: string;
  source: string;
  sentiment: string;
  score: number;
  timestamp: string;
}

interface ArticleListProps {
  articles: Article[];
}

const ArticleList: React.FC<ArticleListProps> = ({ articles }) => {
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  
  const getSentimentColor = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case 'positive':
        return 'green';
      case 'negative':
        return 'red';
      default:
        return 'gray';
    }
  };

  return (
    <VStack spacing={4} align="stretch">
      {articles.map((article) => (
        <Box
          key={article.id}
          p={4}
          borderWidth="1px"
          borderRadius="md"
          borderColor={borderColor}
        >
          <Heading size="sm" mb={2}>
            <Link href={article.url} isExternal color="blue.500">
              {article.title}
            </Link>
          </Heading>
          <Text fontSize="sm" color="gray.500" mb={2}>
            Source: {article.source} | {new Date(article.timestamp).toLocaleDateString()}
          </Text>
          <Badge
            colorScheme={getSentimentColor(article.sentiment)}
            variant="subtle"
          >
            {article.sentiment} ({(article.score * 100).toFixed(1)}%)
          </Badge>
        </Box>
      ))}
      {articles.length === 0 && (
        <Text color="gray.500" textAlign="center">
          No articles available
        </Text>
      )}
    </VStack>
  );
};

export default ArticleList; 