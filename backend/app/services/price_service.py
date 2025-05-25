import aiohttp
from app.core.config import settings
from app.models.models import TokenData
from datetime import datetime
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

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
    ids = ','.join(COINGECKO_IDS.values())
    url = f"{settings.COINGECKO_API_URL}/simple/price"
    params = {
        "ids": ids,
        "vs_currencies": "usd",
        "include_market_cap": "true",
        "include_24hr_vol": "true",
        "include_24hr_change": "true"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logging.error(f"CoinGecko API error: {response.status}")
                    return {}
    except Exception as e:
        logging.error(f"Error fetching token data: {e}")
        return {}

async def update_token_prices(db: AsyncSession):
    """Update price data for all supported tokens"""
    try:
        price_data = await fetch_token_data()
        
        for symbol, coingecko_id in COINGECKO_IDS.items():
            if coingecko_id in price_data:
                data = price_data[coingecko_id]
                
                # Check if token exists
                result = await db.execute(
                    select(TokenData).where(TokenData.symbol == symbol)
                )
                token = result.scalar_one_or_none()
                
                if token:
                    # Update existing token
                    token.price = data.get('usd', 0.0)
                    token.market_cap = data.get('usd_market_cap', 0.0)
                    token.volume = data.get('usd_24h_vol', 0.0)
                    token.last_updated = datetime.utcnow()
                else:
                    # Create new token
                    token = TokenData(
                        symbol=symbol,
                        price=data.get('usd', 0.0),
                        market_cap=data.get('usd_market_cap', 0.0),
                        volume=data.get('usd_24h_vol', 0.0),
                        last_updated=datetime.utcnow()
                    )
                    db.add(token)
        
        # Commit all changes
        await db.commit()
                
    except Exception as e:
        logging.error(f"Error in update_token_prices: {e}")
        await db.rollback()
        raise 