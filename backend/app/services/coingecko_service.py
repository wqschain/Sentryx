from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import asyncio
import logging
from ..core.config import settings
import aiohttp
import time
from functools import wraps
from cachetools import TTLCache
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..models.models import (
    TokenData as TokenDataModel,
    Article,
    SentimentHistory,
    TokenPriceHistory,
    TokenVolumeHistory
)
from sqlalchemy.ext.asyncio import create_async_engine
import random

logger = logging.getLogger(__name__)

# Global caches
token_cache = TTLCache(maxsize=100, ttl=43200)  # 12 hours TTL
details_cache = TTLCache(maxsize=100, ttl=43200)  # 12 hours TTL
price_history_cache = TTLCache(maxsize=100, ttl=43200)  # 12 hours TTL
volume_history_cache = TTLCache(maxsize=100, ttl=43200)  # 12 hours TTL

def get_nested(data: Dict[str, Any], *keys: str) -> Any:
    """Helper function to safely get nested dictionary values"""
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data

class RateLimiter:
    def __init__(self, max_calls_per_minute: int = 5):  # Conservative: 5 calls per minute
        self.max_calls = max_calls_per_minute
        self.calls = []
        self.lock = asyncio.Lock()
        self.min_delay = 12  # 12 seconds between calls for free API
        self.last_request_time = 0

    async def acquire(self):
        """Wait for rate limit and acquire permission to make a request"""
        async with self.lock:
            now = time.time()
            
            # First enforce minimum delay between calls
            time_since_last = now - self.last_request_time
            if time_since_last < self.min_delay:
                delay = self.min_delay - time_since_last
                logger.info(f"Enforcing minimum delay: waiting {delay:.1f} seconds")
                await asyncio.sleep(delay)
            
            # Then check rate limits
            self.calls = [t for t in self.calls if now - t < 60]  # Keep only last minute
            if len(self.calls) >= self.max_calls:
                wait_time = 60 - (now - self.calls[0])
                if wait_time > 0:
                    logger.warning(f"Rate limiting: waiting {wait_time:.1f} seconds")
                    await asyncio.sleep(wait_time)
                    self.calls = []  # Reset after waiting
            
            self.calls.append(now)
            self.last_request_time = now

rate_limiter = RateLimiter()

async def rate_limited_request(session: aiohttp.ClientSession, url: str, params: dict = None) -> Optional[dict]:
    """Make a rate-limited request with proper error handling and retries"""
    max_retries = 3
    retry_delay = 30  # Start with 30 seconds
    
    for attempt in range(max_retries):
        try:
            await rate_limiter.acquire()
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Rate limit hit
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry")
                        await asyncio.sleep(wait_time)
                        continue
                    logger.warning("Rate limit hit, using cached data if available")
                    return None
                else:
                    logger.error(f"Request failed with status {response.status}")
                    return None
        except asyncio.TimeoutError:
            logger.error("Request timeout")
            if attempt < max_retries - 1:
                continue
            return None
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            if attempt < max_retries - 1:
                continue
            return None
    
    return None

class CoinGeckoService:
    def __init__(self):
        self.cg = CoinGeckoAPI()
        self.api_key = settings.COINGECKO_API_KEY
        
        # Always use regular API URL for demo keys
        self.api_url = "https://api.coingecko.com/api/v3"
        self.headers = {}
        self.rate_limit = settings.COINGECKO_RATE_LIMIT
        self.token_id_map = {k.upper(): v for k, v in settings.COINGECKO_MAPPING.items()}
        self._session = None
        
        # Initialize caches as instance variables
        self.token_cache = token_cache
        self.details_cache = details_cache
        self.price_history_cache = price_history_cache
        self.volume_history_cache = volume_history_cache
        
        logger.info(f"Initializing CoinGeckoService with API URL: {self.api_url}")

    async def get_session(self):
        """Get or create an aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
            self._session = aiohttp.ClientSession(headers=self.headers, timeout=timeout)
        return self._session

    async def batch_get_token_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get data for multiple tokens in a single batch"""
        results = {}
        coin_ids = []
        
        # First, check cache for all tokens
        for symbol in symbols:
            symbol = symbol.upper()
            if symbol in token_cache:
                cached_data = token_cache[symbol]
                # Use cache if less than 5 minutes old
                if isinstance(cached_data.get('last_updated'), datetime) and cached_data['last_updated'] > datetime.now(timezone.utc) - timedelta(minutes=5):
                    results[symbol] = cached_data
                    continue
            
            if symbol in self.token_id_map:
                coin_ids.append(self.token_id_map[symbol])
        
        if not coin_ids:
            return results
        
        # Batch API call for remaining tokens
        url = f"{self.api_url}/coins/markets"
        params = {
            "vs_currency": "usd",
            "ids": ",".join(coin_ids),
            "order": "market_cap_desc",
            "per_page": len(coin_ids),
            "page": 1,
            "sparkline": "false"
        }
        
        session = await self.get_session()
        data = await rate_limited_request(session, url, params)
        
        if data:
            for item in data:
                symbol = next((s for s, i in self.token_id_map.items() if i == item['id']), None)
                if symbol:
                    # Convert timestamp to datetime object
                    if isinstance(item.get('last_updated'), str):
                        try:
                            last_updated = datetime.fromisoformat(item['last_updated'].replace('Z', '+00:00'))
                        except ValueError:
                            logger.warning(f"Could not parse timestamp for {symbol}, using current time")
                            last_updated = datetime.now(timezone.utc)
                    else:
                        last_updated = datetime.now(timezone.utc)
                    
                    result = {
                        "symbol": symbol,
                        "price": item.get('current_price', 0),
                        "price_change_24h": item.get('price_change_percentage_24h', 0),
                        "market_cap": item.get('market_cap', 0),
                        "volume": item.get('total_volume', 0),
                        "high_24h": item.get('high_24h', 0),
                        "low_24h": item.get('low_24h', 0),
                        "circulating_supply": item.get('circulating_supply', 0),
                        "max_supply": item.get('max_supply', 0),
                        "market_rank": item.get('market_cap_rank', 0),
                        "ath": item.get('ath', 0),
                        "last_updated": last_updated
                    }
                    results[symbol] = result
                    token_cache[symbol] = result
        else:
            # Return any available cached data for remaining tokens
            for symbol in symbols:
                if symbol not in results and symbol in token_cache:
                    results[symbol] = token_cache[symbol]
        
        return results

    async def batch_get_market_charts(self, symbols: List[str], chart_type: str = "prices", days: int = 7) -> Dict[str, List[Dict]]:
        """Get historical data for multiple tokens in batches"""
        results = {}
        cache = price_history_cache if chart_type == "prices" else volume_history_cache
        session = await self.get_session()
        
        for symbol in symbols:
            symbol = symbol.upper()
            cache_key = f"{symbol}_{chart_type}_{days}"
            
            # Check cache first
            if cache_key in cache:
                results[symbol] = cache[cache_key]
                continue
            
            if symbol not in self.token_id_map:
                continue
            
            url = f"{self.api_url}/coins/{self.token_id_map[symbol]}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": str(days),
                "interval": "daily"
            }
            
            data = await rate_limited_request(session, url, params)
            if data:
                if chart_type == "prices":
                    result = [
                        {
                            'token_symbol': symbol,
                            'price': price,
                            'timestamp': datetime.fromtimestamp(timestamp/1000, tz=timezone.utc)
                        }
                        for timestamp, price in data.get('prices', [])
                    ]
                else:  # volumes
                    result = [
                        {
                            'token_symbol': symbol,
                            'volume': volume,
                            'timestamp': datetime.fromtimestamp(timestamp/1000, tz=timezone.utc)
                        }
                        for timestamp, volume in data.get('total_volumes', [])
                    ]
                
                results[symbol] = result
                cache[cache_key] = result
            elif symbol in cache:
                results[symbol] = cache[symbol]
        
        return results

    async def update_all_token_data(self, db: AsyncSession):
        """Update all token data in optimized batches"""
        try:
            symbols = list(self.token_id_map.keys())
            
            # Step 1: Get basic token data in a single batch
            token_data = await self.batch_get_token_data(symbols)
            
            # Step 2: Update database with token data
            for symbol, data in token_data.items():
                stmt = select(TokenDataModel).where(TokenDataModel.symbol == symbol)
                result = await db.execute(stmt)
                token_model = result.scalar_one_or_none()
                
                if token_model:
                    for key, value in data.items():
                        if hasattr(token_model, key):
                            setattr(token_model, key, value)
                else:
                    token_model = TokenDataModel(**data)
                    db.add(token_model)
            
            await db.commit()
            
            # Step 3: Get historical data in batches
            price_history = await self.batch_get_market_charts(symbols, "prices")
            volume_history = await self.batch_get_market_charts(symbols, "volumes")
            
            # Step 4: Update historical data
            for symbol in symbols:
                # Update price history
                if symbol in price_history:
                    for price_data in price_history[symbol]:
                        price_record = TokenPriceHistory(**price_data)
                        db.add(price_record)
                
                # Update volume history
                if symbol in volume_history:
                    for volume_data in volume_history[symbol]:
                        volume_record = TokenVolumeHistory(**volume_data)
                        db.add(volume_record)
            
            await db.commit()
            logger.info("Successfully updated all token data in batches")
            
        except Exception as e:
            logger.error(f"Error in update_all_token_data: {str(e)}")
            await db.rollback()
            raise

    async def get_token_details(self, symbol: str, db: AsyncSession) -> Optional[Dict[str, Any]]:
        """Get detailed token information including price and volume history"""
        try:
            symbol = symbol.upper()
            
            # Check cache first
            cache_key = f"{symbol}_details"
            if cache_key in details_cache:
                return details_cache[cache_key]
            
            # Get base token data
            result = await db.execute(select(TokenDataModel).where(TokenDataModel.symbol == symbol))
            token = result.scalar_one_or_none()
            if not token:
                return None
            
            # Initialize response with base data
            details = {
                "id": token.id,  # Include the database ID
                "symbol": token.symbol,
                "price": token.price,
                "price_change_24h": token.price_change_24h,
                "market_cap": token.market_cap,
                "volume": token.volume,
                "high_24h": token.high_24h,
                "low_24h": token.low_24h,
                "circulating_supply": token.circulating_supply,
                "max_supply": token.max_supply,
                "market_rank": token.market_rank,
                "ath": token.ath,
                "last_updated": token.last_updated,
                "price_history": [],
                "volume_history": [],
                "recent_articles": [],
                "sentiment_history": [],
                "current_sentiment": 0.0
            }
            
            # Cache initial response immediately
            details_cache[cache_key] = details
            
            # Fetch additional data in parallel
            try:
                price_history = (await self.batch_get_market_charts([symbol], "prices")).get(symbol, [])
                volume_history = (await self.batch_get_market_charts([symbol], "volumes")).get(symbol, [])
                
                sentiment_result = await db.execute(
                    select(SentimentHistory)
                    .where(SentimentHistory.token == symbol)
                    .order_by(SentimentHistory.timestamp.desc())
                    .limit(24)
                )
                
                articles_result = await db.execute(
                    select(Article)
                    .where(Article.token == symbol)
                    .order_by(Article.timestamp.desc())
                    .limit(5)
                )
                
                # Update details with results
                details["price_history"] = price_history
                details["volume_history"] = volume_history
                
                sentiment_history = list(sentiment_result.scalars().all())
                details["sentiment_history"] = sentiment_history
                if sentiment_history:
                    details["current_sentiment"] = sum(sh.average_score for sh in sentiment_history) / len(sentiment_history)
                
                details["recent_articles"] = list(articles_result.scalars().all())
                
                # Update cache with full data
                details_cache[cache_key] = details
                
            except Exception as e:
                logger.error(f"Error getting additional data for {symbol}: {str(e)}")
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting token details for {symbol}: {str(e)}")
            return None

    async def close(self):
        """Close the aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_price_history(self, symbol: str, days: int = 7) -> List[Dict]:
        """Get historical price data for a token"""
        try:
            token_id = self.token_id_map.get(symbol)
            if not token_id:
                logger.error(f"Token ID not found for symbol {symbol}")
                return []
            
            # Generate mock data for now
            data = []
            now = datetime.now(timezone.utc)
            for i in range(days):
                timestamp = now - timedelta(days=days-i-1)
                price = random.uniform(100000, 120000) if symbol == "BTC" else random.uniform(2000, 3000)
                data.append({
                    "token_symbol": symbol,
                    "price": price,
                    "timestamp": timestamp
                })
            # Add current price point
            data.append({
                "token_symbol": symbol,
                "price": self.token_cache.get(symbol, {}).get('price', 0),
                "timestamp": now
            })
            return data
        except Exception as e:
            logger.error(f"Error getting price history for {symbol}: {e}")
            return []

    async def get_volume_history(self, symbol: str, days: int = 7) -> List[Dict]:
        """Get historical volume data for a token"""
        try:
            token_id = self.token_id_map.get(symbol)
            if not token_id:
                logger.error(f"Token ID not found for symbol {symbol}")
                return []
            
            # Generate mock data for now
            data = []
            now = datetime.now(timezone.utc)
            for i in range(days):
                timestamp = now - timedelta(days=days-i-1)
                volume = random.uniform(20e9, 60e9) if symbol == "BTC" else random.uniform(5e9, 15e9)
                data.append({
                    "token_symbol": symbol,
                    "volume": volume,
                    "timestamp": timestamp
                })
            # Add current volume point
            data.append({
                "token_symbol": symbol,
                "volume": self.token_cache.get(symbol, {}).get('volume', 0),
                "timestamp": now
            })
            return data
        except Exception as e:
            logger.error(f"Error getting volume history for {symbol}: {e}")
            return []

coingecko_service = CoinGeckoService() 