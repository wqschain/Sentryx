# Sentryx (v2.2)

> **Development Status**: This is a base version (v2.2) and is under constant development. Features and functionality are being actively added and improved. Please expect frequent updates and potential breaking changes.

A powerful cryptocurrency analysis platform that combines real-time price tracking with advanced sentiment analysis. The platform uses SentryxAI, our fine-tuned sentiment analysis model, to analyze market sentiment from various sources.

## Version 2.2 Updates

### New Features
- Implemented text relevancy analysis:
  - Pre-analysis filtering to determine if text is crypto-relevant
  - Smart decision making to only analyze relevant content
  - UI indicators showing relevancy scores
  - Improved accuracy by filtering out non-crypto content
- Enhanced token data display:
  - More comprehensive market metrics
  - Detailed historical data visualization
  - Better data organization and presentation

### UI Improvements
- Reduced overall UI scaling by 30% for better space utilization
- Optimized TokenPage, MarketStats, PriceChart, and VolumeChart components
- Enhanced visual hierarchy with improved color contrast and typography
- Added subtle animations for better user feedback
- Improved component spacing and layout
- Enriched color palette for better data visualization:
  - Price increases: Vibrant green (#00C853)
  - Price decreases: Warm red (#FF3D00)
  - Volume bars: Rich blue (#2962FF)
  - Chart backgrounds: Gradient overlays for depth
- Enhanced navigation experience:
  - Smoother transitions between pages
  - Better hover states for interactive elements
  - Improved scrolling behavior in data tables
  - Consistent spacing and alignment across views
- Added relevancy indicators and feedback

### Data Model Optimization
- Streamlined token data model by removing redundant fields (market dominance, ATL)
- Enhanced database initialization process for better reliability
- Improved case handling for token symbols to prevent duplicates
- Added support for relevancy scoring and tracking

### Backend Enhancements
- Implemented text relevancy analysis system
- Improved error handling in CoinGecko service integration
- Enhanced transaction management in database operations
- Added better rate limiting for external API calls
- Optimized data processing pipelines

### Code Quality
- Implemented consistent case handling for token symbols
- Enhanced error logging and monitoring
- Improved startup sequence reliability
- Better handling of database cleanup operations
- Refactored for better code organization and maintainability
- Added comprehensive error handling for API integrations

### Known Issues
- Initial Token Page Loading:
  - Token pages may take several minutes to load after backend startup
  - This is due to the initial data population and caching process
  - BTC data typically loads faster than other tokens
  - Subsequent loads are much faster once data is cached

- Price/Volume Chart Updates:
  - Chart updates can be inconsistent or spotty
  - Updates may not reflect in real-time as expected
  - This is related to rate limiting and data synchronization
  - Working on optimizing the update frequency and reliability

- Database Synchronization:
  - Occasional timezone comparison issues between stored and new data
  - Some database operations may fail due to unique constraint conflicts
  - These issues typically resolve themselves after a few minutes
  - Working on improving data consistency and error handling

## Features

- Real-time cryptocurrency price tracking for BTC, ETH, SOL, XRP, and DOGE
- Advanced sentiment analysis using SentryxAI (based on CryptoBERT), our custom-tuned model
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
- SentryxAI: Our custom sentiment analysis model
- APScheduler for automated tasks
- Comprehensive API monitoring and rate limiting
- Playwright for reliable web scraping

## SentryxAI Model

SentryxAI is our custom-tuned sentiment analysis model, specifically designed for cryptocurrency market analysis. It is built upon the foundation of CryptoBERT (created by ElKulako), which we've fine-tuned with extensive cryptocurrency-specific data and enhanced for better performance in crypto sentiment analysis.

### Model Architecture
- Base Model: CryptoBERT by ElKulako (https://huggingface.co/ElKulako/cryptobert)
- Fine-tuning: Custom training on crypto-specific sentiment data
- Improvements: Enhanced accuracy for crypto terminology and market sentiment

### Key Features
- Specialized in cryptocurrency market sentiment
- Three-way classification (Positive/Neutral/Negative)
- Optimized for real-time analysis
- Continuous performance monitoring and updates

### Credits
Special thanks to ElKulako for creating and open-sourcing CryptoBERT, which serves as the foundation for SentryxAI. The original CryptoBERT model can be found at https://huggingface.co/ElKulako/cryptobert.

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
- SentryxAI: Our custom-tuned model for crypto-specific sentiment analysis
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

## Contact

For questions, support, or collaboration:
- X (Twitter): [@wqschain](https://twitter.com/wqschain)

## Security & Environment Setup

 **Important: API Keys and Sensitive Data**
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
   MODEL_PATH=./models/SentryxAI
   LOG_LEVEL=INFO

   # Frontend (.env)
   VITE_API_URL=http://localhost:8000/api
   ```

3. Add any additional API keys you need:
   - CoinGecko API key (optional, but recommended)
   - News API keys (if using news features)
   - Any other third-party service keys

 **Security Best Practices**:
- Keep your `.env` files local and never share them
- Use different API keys for development and production
- Regularly rotate your API keys
- Monitor your API usage for unauthorized access
- Consider using a secrets management service in production 
