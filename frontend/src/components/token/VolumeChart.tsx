import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { Box, useColorModeValue } from '@chakra-ui/react';

interface VolumeData {
  timestamp: string;
  volume: number;
}

interface VolumeChartProps {
  data: VolumeData[];
}

const VolumeChart: React.FC<VolumeChartProps> = ({ data }) => {
  const barColor = useColorModeValue('cyan.500', 'cyan.200');
  const gridColor = useColorModeValue('gray.200', 'gray.700');

  return (
    <Box h="175px" w="100%">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{
            top: 3,
            right: 20,
            left: 15,
            bottom: 3,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
          <XAxis
            dataKey="timestamp"
            tickFormatter={(timestamp) => new Date(timestamp).toLocaleDateString()}
          />
          <YAxis
            tickFormatter={(value) => {
              if (value >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
              if (value >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
              return `$${value.toLocaleString()}`;
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: useColorModeValue('white', 'gray.800'),
              border: '1px solid',
              borderColor: gridColor,
            }}
            formatter={(value: number) => {
              if (value >= 1e9) return [`$${(value / 1e9).toFixed(2)}B`, 'Volume'];
              if (value >= 1e6) return [`$${(value / 1e6).toFixed(2)}M`, 'Volume'];
              return [`$${value.toLocaleString()}`, 'Volume'];
            }}
            labelFormatter={(label) => new Date(label).toLocaleString()}
          />
          <Legend />
          <Bar
            dataKey="volume"
            fill={barColor}
            name="Trading Volume"
          />
        </BarChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default VolumeChart; 