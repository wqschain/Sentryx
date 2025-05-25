import React from 'react';
import {
  Box,
  Text,
  useColorModeValue,
} from '@chakra-ui/react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface SentimentData {
  timestamp: string;
  average_score: number;
}

interface SentimentChartProps {
  data: SentimentData[];
}

const SentimentChart: React.FC<SentimentChartProps> = ({ data }) => {
  const lineColor = useColorModeValue('blue.500', 'blue.300');
  const gridColor = useColorModeValue('gray.200', 'gray.700');

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
    });
  };

  const formatScore = (score: number) => {
    return `${(score * 100).toFixed(1)}%`;
  };

  if (data.length === 0) {
    return (
      <Box textAlign="center" py={10}>
        <Text color="gray.500">No sentiment data available</Text>
      </Box>
    );
  }

  const chartData = data.map(item => ({
    date: formatDate(item.timestamp),
    sentiment: item.average_score,
  }));

  return (
    <Box h="300px" w="100%">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
          <XAxis
            dataKey="date"
            tick={{ fill: useColorModeValue('gray.800', 'gray.200') }}
          />
          <YAxis
            tickFormatter={formatScore}
            domain={[-1, 1]}
            tick={{ fill: useColorModeValue('gray.800', 'gray.200') }}
          />
          <Tooltip
            formatter={(value: number) => formatScore(value)}
            labelStyle={{ color: useColorModeValue('gray.800', 'gray.200') }}
          />
          <Line
            type="monotone"
            dataKey="sentiment"
            stroke={lineColor}
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default SentimentChart; 