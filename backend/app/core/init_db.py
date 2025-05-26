import asyncio
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import inspect, delete
from app.core.config import settings
from app.models.base import Base
from app.models.models import TokenData
from datetime import datetime, timezone
from sqlalchemy.sql import select, insert

logger = logging.getLogger(__name__)

async def init_db():
    """Initialize database and create tables if they don't exist"""
    # Create async engine
    engine = create_async_engine(
        settings.SQLITE_URL.replace("sqlite:///", "sqlite+aiosqlite:///"),
        echo=True
    )
    
    # Create tables that don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Check if TokenData table is already initialized
    async with engine.begin() as conn:
        result = await conn.execute(select(TokenData))
        if result.first() is not None:
            logger.info("TokenData table already initialized.")
            return
    
    # Initialize supported tokens with realistic initial data
    initial_data = {
        "BTC": {
            "price": 65000.0,
            "market_cap": 1.2e12,
            "volume": 45.0e9,
            "market_rank": 1
        },
        "ETH": {
            "price": 3500.0,
            "market_cap": 420.0e9,
            "volume": 15.0e9,
            "market_rank": 2
        },
        "SOL": {
            "price": 110.0,
            "market_cap": 48.0e9,
            "volume": 2.5e9,
            "market_rank": 3
        },
        "XRP": {
            "price": 0.75,
            "market_cap": 35.0e9,
            "volume": 1.2e9,
            "market_rank": 4
        },
        "DOGE": {
            "price": 0.15,
            "market_cap": 18.0e9,
            "volume": 800.0e6,
            "market_rank": 5
        }
    }
    
    # Insert initial token data in a single transaction
    async with engine.begin() as conn:
        for symbol, data in initial_data.items():
            await conn.execute(
                insert(TokenData).values(
                    symbol=symbol.upper(),  # Always store uppercase symbols
                    price=data["price"],
                    market_cap=data["market_cap"],
                    volume=data["volume"],
                    price_change_24h=0.0,
                    high_24h=data["price"],
                    low_24h=data["price"],
                    circulating_supply=0.0,
                    max_supply=0.0,
                    market_rank=data["market_rank"],  # Use the provided market rank
                    ath=data["price"],
                    last_updated=datetime.now(timezone.utc)
                )
            )
        logger.info("Token data initialized successfully")

if __name__ == "__main__":
    asyncio.run(init_db()) 