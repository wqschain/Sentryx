import asyncio
from datetime import datetime, timezone
from sqlalchemy import select, delete
from sqlalchemy.dialects.sqlite import insert
from typing import List, Dict
from ..db.session import async_session
from ..models.models import TokenData, TokenPriceHistory, TokenVolumeHistory
from ..services.coingecko_service import coingecko_service
import logging

logger = logging.getLogger(__name__)

async def update_token_data():
    """Update token data from CoinGecko"""
    async with async_session() as db:
        try:
            # Get all tokens in a single batch
            symbols = list(coingecko_service.token_id_map.keys())
            token_data = await coingecko_service.batch_get_token_data(symbols)
            
            if not token_data:
                logger.error("No token data received from batch request")
                return
            
            async with db.begin():
                # First, delete any tokens that are no longer supported
                current_tokens = list(token_data.keys())
                if current_tokens:
                    await db.execute(
                        delete(TokenData).where(TokenData.symbol.notin_(current_tokens))
                    )
                
                # Then update all tokens in a single transaction
                for symbol, data in token_data.items():
                    # Ensure last_updated is a datetime object
                    if isinstance(data.get('last_updated'), str):
                        data['last_updated'] = datetime.fromisoformat(data['last_updated'].replace('Z', '+00:00'))
                    elif not isinstance(data.get('last_updated'), datetime):
                        data['last_updated'] = datetime.now(timezone.utc)
                    
                    # Remove id from data if present
                    data.pop('id', None)
                    
                    stmt = insert(TokenData).values(**data)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=['symbol'],
                        set_=data
                    )
                    await db.execute(stmt)
                
            await db.commit()
                
        except Exception as e:
            logger.error(f"Error updating token data: {e}")
            await db.rollback()

async def update_token_history():
    """Update token price and volume history"""
    async with async_session() as db:
        try:
            symbols = list(coingecko_service.token_id_map.keys())
            
            # Get price and volume history in batches
            price_history = await coingecko_service.batch_get_market_charts(symbols, "prices", days=7)
            volume_history = await coingecko_service.batch_get_market_charts(symbols, "volumes", days=7)
            
            async with db.begin():
                for symbol in symbols:
                    # Update price history
                    if symbol in price_history:
                        for entry in price_history[symbol]:
                            # Ensure timestamp is a datetime object
                            if isinstance(entry.get('timestamp'), str):
                                entry['timestamp'] = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                            elif not isinstance(entry.get('timestamp'), datetime):
                                continue
                            
                            stmt = insert(TokenPriceHistory).values(
                                token_symbol=symbol,
                                price=entry["price"],
                                timestamp=entry["timestamp"]
                            )
                            await db.execute(stmt)

                    # Update volume history
                    if symbol in volume_history:
                        for entry in volume_history[symbol]:
                            # Ensure timestamp is a datetime object
                            if isinstance(entry.get('timestamp'), str):
                                entry['timestamp'] = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                            elif not isinstance(entry.get('timestamp'), datetime):
                                continue
                            
                            stmt = insert(TokenVolumeHistory).values(
                                token_symbol=symbol,
                                volume=entry["volume"],
                                timestamp=entry["timestamp"]
                            )
                            await db.execute(stmt)

                await db.commit()
                
        except Exception as e:
            logger.error(f"Error updating token history: {e}")
            await db.rollback()

async def start_background_tasks():
    """Start background tasks"""
    while True:
        try:
            # Update token data every 4 hours
            await update_token_data()
            await asyncio.sleep(14400)  # 4 hours

            # Update history data every 4 hours, offset by 2 hours from token data
            await asyncio.sleep(7200)  # 2 hour offset
            await update_token_history()
            await asyncio.sleep(7200)  # Remaining 2 hours
        except Exception as e:
            logger.error(f"Error in background task: {str(e)}")
            await asyncio.sleep(300)  # Wait 5 minutes before retrying 