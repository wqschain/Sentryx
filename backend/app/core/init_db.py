import asyncio
import logging
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import inspect
from app.core.config import settings
from app.models.base import Base
from app.models.models import TokenData, User, Article, SentimentHistory, APIUsage, SentimentPrediction, ModelMetrics
from datetime import datetime

logger = logging.getLogger(__name__)

async def init_db():
    """Initialize database and create tables if they don't exist"""
    # Create async engine
    engine = create_async_engine(
        settings.SQLITE_URL.replace("sqlite:///", "sqlite+aiosqlite:///"),
        echo=True
    )
    
    async with engine.begin() as conn:
        # Create tables that don't exist
        await conn.run_sync(Base.metadata.create_all)
        
        # Check if TokenData table is empty
        result = await conn.execute(TokenData.__table__.select())
        if not result.fetchone():
            logger.info("Initializing supported tokens...")
            current_time = datetime.utcnow()
            
            # Initialize supported tokens with realistic initial data
            initial_data = {
                "BTC": {"price": 65000.0, "market_cap": 1.2e12, "volume": 45.0e9},
                "ETH": {"price": 3500.0, "market_cap": 420.0e9, "volume": 15.0e9},
                "SOL": {"price": 110.0, "market_cap": 48.0e9, "volume": 2.5e9}
            }
            
            for symbol in settings.SUPPORTED_TOKENS:
                data = initial_data[symbol]
                token_data = TokenData(
                    symbol=symbol,
                    price=data["price"],
                    market_cap=data["market_cap"],
                    volume=data["volume"],
                    last_updated=current_time
                )
                await conn.execute(token_data.__table__.insert().values(
                    symbol=token_data.symbol,
                    price=token_data.price,
                    market_cap=token_data.market_cap,
                    volume=token_data.volume,
                    last_updated=token_data.last_updated
                ))
            logger.info("Token initialization complete!")
        else:
            logger.info("TokenData table already initialized.")

if __name__ == "__main__":
    asyncio.run(init_db()) 