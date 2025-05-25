from pydantic_settings import BaseSettings
from typing import List, Dict
import secrets
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Sentryx"
    API_V1_STR: str = "/api"
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Database
    SQLITE_URL: str = "sqlite:///./sentryx.db"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",  # Vite's default development port
        "http://127.0.0.1:5173",
        "http://localhost:5174",  # Alternative Vite port
        "http://127.0.0.1:5174"
    ]
    
    # Supported tokens
    SUPPORTED_TOKENS: List[str] = ["BTC", "ETH", "SOL"]
    
    # Model Configuration
    MODEL_BASE_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    SENTIMENT_MODEL_PATH: str = os.path.join(MODEL_BASE_PATH, "crypto_sentiment")
    MODEL_VERSION: str = "cryptobert-v2"
    
    # Sentiment Label Mapping
    SENTIMENT_LABEL_MAP: Dict[str, str] = {
        "Bearish": "negative",
        "Neutral": "neutral",
        "Bullish": "positive"
    }
    
    # CoinGecko API
    COINGECKO_API_KEY: str = ""
    COINGECKO_API_URL: str = "https://api.coingecko.com/api/v3"
    COINGECKO_RATE_LIMIT: int = 50
    COINGECKO_MONTHLY_LIMIT: int = 10000
    COINGECKO_MAPPING: Dict[str, str] = {
        "btc": "bitcoin",
        "eth": "ethereum",
        "sol": "solana"
    }
    
    # Update intervals
    PRICE_UPDATE_INTERVAL_MINUTES: int = 10  # Update prices every 10 minutes
    SCRAPING_INTERVAL_HOURS: int = 6  # News scraping interval

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings() 