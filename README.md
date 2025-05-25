# Sentryx (v2.1)

> **Development Status**: This is a base version (v2.1) and is under constant development. Features and functionality are being actively added and improved. Please expect frequent updates and potential breaking changes.

A powerful cryptocurrency analysis platform that combines real-time price tracking with advanced sentiment analysis. The platform uses a fine-tuned CryptoBERT model to analyze market sentiment from various sources.

## Features

- Real-time cryptocurrency price tracking for BTC, ETH, and SOL
- Advanced sentiment analysis using our custom fine-tuned CryptoBERT model
- Beautiful and modern UI built with React and TailwindCSS
- High-performance FastAPI backend with async support
- Automated data collection and sentiment analysis
- Real-time sentiment scoring for custom text input
- Historical sentiment tracking and analysis
- Performance monitoring and metrics collection

## Technology Stack

### Frontend
- React 18 with TypeScript
- Vite for fast development and hot module replacement
- TailwindCSS for modern, responsive styling
- Axios for API communication
- Real-time data updates and websocket support

### Backend
- FastAPI (Python) with async support
- SQLite database with SQLAlchemy ORM
- Custom fine-tuned CryptoBERT model for sentiment analysis
- APScheduler for automated tasks
- Comprehensive API monitoring and rate limiting
- Playwright for reliable web scraping

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd sentryx_v2
   ```

2. Set up environment variables:
   ```bash
   # Create .env files in both frontend and backend directories
   # Frontend (.env)
   VITE_API_URL=http://localhost:8000/api

   # Backend (.env)
   COINGECKO_API_KEY=your_api_key  # Optional
   ```

## Setup Instructions

### Backend Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend server:
   ```bash
   cd backend
   python -m uvicorn app.main:app --reload
   ```

### Frontend Setup

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:5173` (or next available port).

## Project Structure

```
sentryx_v2/
├── frontend/                # React frontend application
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/      # API services
│   │   └── styles/        # TailwindCSS styles
│   └── package.json
├── backend/
│   ├── app/
│   │   ├── api/           # API endpoints
│   │   ├── core/          # Core functionality
│   │   ├── models/        # Database models
│   │   ├── services/      # Business logic
│   │   └── token/         # Token-specific logic
│   └── models/            # ML models
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## API Documentation

The API documentation is available at `http://localhost:8000/docs` when the backend server is running. This provides interactive Swagger documentation for all available endpoints.

## Features in Detail

### Sentiment Analysis
- Custom-trained CryptoBERT model for crypto-specific sentiment analysis
- Real-time sentiment scoring for any crypto-related text
- Historical sentiment tracking and analysis
- Performance metrics and model monitoring

### Price Tracking
- Real-time price updates for supported tokens
- Historical price data
- Market cap and volume tracking
- Automated data collection

### Monitoring
- API usage tracking
- Model performance monitoring
- Sentiment distribution analysis
- Automated report generation

## API Keys & External Services

### CoinGecko API (Recommended)
This project uses CoinGecko's API for real-time cryptocurrency data. While the free tier works, we recommend getting an API key for better rate limits and reliability.

1. Visit [CoinGecko Pro](https://www.coingecko.com/en/api/pricing)
2. Sign up for an account (free tier available)
3. Get your API key
4. Add it to your `.env` file:
   ```bash
   COINGECKO_API_KEY=your_api_key_here
   ```

> **Note**: The application can run without a CoinGecko API key, but you'll have limited requests per minute.

## License

MIT License

## Security & Environment Setup

⚠️ **Important: API Keys and Sensitive Data**
- The repository includes a `.env.example` file as a template
- Never commit your actual `.env` files to the repository
- The `.gitignore` file is configured to exclude `.env` files
- Keep your API keys and sensitive data secure

1. Create your environment files:
   ```bash
   # Create .env files from examples
   cp .env.example backend/.env
   cp .env.example frontend/.env
   ```

2. Configure your environment variables:
   ```bash
   # Backend (.env)
   DATABASE_URL=sqlite:///./app.db
   COINGECKO_API_KEY=your_api_key
   MODEL_PATH=./models/crypto_sentiment
   LOG_LEVEL=INFO

   # Frontend (.env)
   VITE_API_URL=http://localhost:8000/api
   ```

3. Add any additional API keys you need:
   - CoinGecko API key (optional, but recommended)
   - News API keys (if using news features)
   - Any other third-party service keys

⚠️ **Security Best Practices**:
- Keep your `.env` files local and never share them
- Use different API keys for development and production
- Regularly rotate your API keys
- Monitor your API usage for unauthorized access
- Consider using a secrets management service in production 