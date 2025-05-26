import aiohttp
from app.core.config import settings
from app.models.models import TokenData, TokenPriceHistory, TokenVolumeHistory
from datetime import datetime
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from .coingecko_service import coingecko_service
import asyncio

logger = logging.getLogger(__name__)

# CoinGecko ID mapping for our supported tokens
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "DOGE": "dogecoin"
}

async def fetch_token_data() -> dict:
    """Fetch current price data for all supported tokens from CoinGecko"""
    try:
        data = {}
        for symbol in coingecko_service.token_id_map.keys():
            token_data = await coingecko_service.get_token_data(symbol)
            if token_data:
                data[symbol.upper()] = token_data
            await asyncio.sleep(1)  # Rate limiting
        return data
    except Exception as e:
        logger.error(f"Error fetching token data: {e}")
        return {}

async def update_token_prices(db: AsyncSession):
    """Update price data for all supported tokens"""
    try:
        # Get current token data
        for symbol in coingecko_service.token_id_map.keys():
            token_data = await coingecko_service.get_token_data(symbol)
            if not token_data:
                continue

            # Update or create token data
            result = await db.execute(
                select(TokenData).where(TokenData.symbol == symbol.upper())
            )
            token = result.scalar_one_or_none()

            if token:
                for key, value in token_data.items():
                    setattr(token, key, value)
            else:
                token = TokenData(**token_data)
                db.add(token)

            # Get and store price history
            price_history = await coingecko_service.get_price_history(symbol)
            for price_data in price_history:
                price_record = TokenPriceHistory(**price_data)
                db.add(price_record)

            # Get and store volume history
            volume_history = await coingecko_service.get_volume_history(symbol)
            for volume_data in volume_history:
                volume_record = TokenVolumeHistory(**volume_data)
                db.add(volume_record)

        # Commit all changes
        await db.commit()

    except Exception as e:
        logger.error(f"Error in update_token_prices: {e}")
        await db.rollback()
        raise 