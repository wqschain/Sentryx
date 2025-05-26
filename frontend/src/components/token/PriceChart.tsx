import React from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { Box, useColorModeValue } from '@chakra-ui/react';

interface PriceData {
  timestamp: string;
  price: number;
}

interface PriceChartProps {
  data: PriceData[];
}

const PriceChart: React.FC<PriceChartProps> = ({ data }) => {
  const areaColor = useColorModeValue('blue.500', 'blue.200');
  const gridColor = useColorModeValue('gray.200', 'gray.700');

  return (
    <Box h="175px" w="100%">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
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
            tickFormatter={(value) => `$${value.toLocaleString()}`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: useColorModeValue('white', 'gray.800'),
              border: '1px solid',
              borderColor: gridColor,
            }}
            formatter={(value: number) => [`$${value.toLocaleString()}`, 'Price']}
            labelFormatter={(label) => new Date(label).toLocaleString()}
          />
          <Legend />
          <Area
            type="monotone"
            dataKey="price"
            stroke={areaColor}
            fill={areaColor}
            fillOpacity={0.1}
            name="Price"
          />
        </AreaChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default PriceChart; 