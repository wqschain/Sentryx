from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta, timezone
from typing import List, Dict
import asyncio
import logging

logger = logging.getLogger(__name__)

from app.core.database import get_db
from app.core.security import get_current_user
from app.auth.schemas import User
from app.schemas.token import TokenData, TokenDetailResponse, Article, SentimentHistory
from app.services.data_collection.api_monitor import api_monitor
from app.models.models import TokenData as TokenDataModel, Article as ArticleModel, SentimentHistory as SentimentHistoryModel
from app.services.coingecko_service import coingecko_service
from app.core.config import settings

router = APIRouter()

async def update_token_data_background(db: AsyncSession):
    """Background task to update token data"""
    try:
        await coingecko_service.update_all_token_data(db)
    except Exception as e:
        logger.error(f"Error in background update: {str(e)}")

@router.get("/usage", response_model=Dict)
async def get_api_usage(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current API usage statistics (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return await api_monitor.get_usage_stats(db)

@router.get("/", response_model=List[TokenData])
async def get_tokens(db: AsyncSession = Depends(get_db)):
    """Get list of all supported tokens"""
    try:
        # Get all tokens from database first
        result = await db.execute(
            select(TokenDataModel)
            .order_by(TokenDataModel.market_rank)  # Order by market rank
        )
        tokens = result.scalars().all()
        
        # If no tokens in database or data is stale, update from CoinGecko
        now = datetime.now(timezone.utc)
        if not tokens or any(
            token.last_updated.replace(tzinfo=timezone.utc) < now - timedelta(minutes=5)
            for token in tokens
            if token.last_updated is not None
        ):
            # Use the new batch update method
            await coingecko_service.update_all_token_data(db)
            # Refresh tokens from database with ordering
            result = await db.execute(
                select(TokenDataModel)
                .order_by(TokenDataModel.market_rank)  # Order by market rank
            )
            tokens = result.scalars().all()
        
        return tokens
    except Exception as e:
        logger.error(f"Error getting tokens: {e}")
        raise HTTPException(status_code=500, detail="Error fetching token data")

@router.get("/{symbol}", response_model=TokenDetailResponse)
async def get_token(
    symbol: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Get detailed data for a specific token"""
    try:
        symbol = symbol.upper()
        
        # First try to get from cache
        cache_key = f"{symbol}_details"
        if cache_key in coingecko_service.details_cache:
            details = coingecko_service.details_cache[cache_key]
            # Schedule background update if data is older than 5 minutes
            last_updated = details.get('last_updated')
            if isinstance(last_updated, str):
                last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            if isinstance(last_updated, datetime) and not last_updated.tzinfo:
                last_updated = last_updated.replace(tzinfo=timezone.utc)
            
            if last_updated < datetime.now(timezone.utc) - timedelta(minutes=5):
                background_tasks.add_task(coingecko_service.update_all_token_data, db)
            return details
        
        # Then check database
        result = await db.execute(
            select(TokenDataModel).where(TokenDataModel.symbol == symbol)
        )
        token = result.scalar_one_or_none()
        
        if not token or (
            token.last_updated and 
            token.last_updated.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc) - timedelta(minutes=5)
        ):
            # Use batch update for efficiency
            await coingecko_service.update_all_token_data(db)
            result = await db.execute(
                select(TokenDataModel).where(TokenDataModel.symbol == symbol)
            )
            token = result.scalar_one_or_none()
        
        if not token:
            raise HTTPException(status_code=404, detail=f"Token {symbol} not found")
        
        # Get additional data
        details = await coingecko_service.get_token_details(symbol, db)
        if not details:
            raise HTTPException(status_code=404, detail=f"Token {symbol} details not found")
        
        # Ensure last_updated is a string in ISO format
        if isinstance(details['last_updated'], datetime):
            details['last_updated'] = details['last_updated'].isoformat()
        
        return details
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting token {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching token data")

@router.get("/{symbol}/articles", response_model=List[Article])
async def get_token_articles(
    symbol: str,
    db: AsyncSession = Depends(get_db)
):
    """Get recent articles for a specific token"""
    result = await db.execute(
        select(ArticleModel)
        .where(ArticleModel.token == symbol.upper())
        .order_by(ArticleModel.timestamp.desc())
        .limit(5)
    )
    return result.scalars().all()

@router.get("/{symbol}/sentiment", response_model=List[SentimentHistory])
async def get_token_sentiment(
    symbol: str,
    db: AsyncSession = Depends(get_db)
):
    """Get sentiment history for a specific token"""
    result = await db.execute(
        select(SentimentHistoryModel)
        .where(SentimentHistoryModel.token == symbol.upper())
        .order_by(SentimentHistoryModel.timestamp.desc())
        .limit(24)
    )
    return result.scalars().all()

@router.get("/{symbol}/price-history", response_model=List[Dict])
async def get_token_price_history(
    symbol: str,
    days: int = 7,
    db: AsyncSession = Depends(get_db)
):
    """Get price history for a specific token"""
    try:
        price_history = await coingecko_service.get_price_history(symbol, days)
        if not price_history:
            # Return empty list instead of 404 to maintain consistency
            return []
        return price_history
    except Exception as e:
        logger.error(f"Error getting price history for {symbol}: {str(e)}")
        return []

@router.get("/{symbol}/volume-history", response_model=List[Dict])
async def get_token_volume_history(
    symbol: str,
    days: int = 7,
    db: AsyncSession = Depends(get_db)
):
    """Get volume history for a specific token"""
    try:
        volume_history = await coingecko_service.get_volume_history(symbol, days)
        if not volume_history:
            # Return empty list instead of 404 to maintain consistency
            return []
        return volume_history
    except Exception as e:
        logger.error(f"Error getting volume history for {symbol}: {str(e)}")
        return []

# Admin endpoints (requires authentication)
@router.post("/update", status_code=204)
async def trigger_update(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Manually trigger token data update (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    await coingecko_service.update_token_data(db)
    return {"message": "Update triggered successfully"}

@router.get("/test-health", response_model=dict)
async def test_health():
    return {"status": "ok"} 