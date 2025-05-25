from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict
from sqlalchemy import select

from app.core.database import get_db
from app.core.security import get_current_user
from app.auth.schemas import User
from . import service, schemas
from .api_monitor import api_monitor
from .models import TokenData, Article, SentimentHistory

router = APIRouter()

@router.get("/usage", response_model=Dict)
async def get_api_usage(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current API usage statistics (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return await api_monitor.get_usage_stats(db)

@router.get("/", response_model=List[schemas.TokenData])
async def get_tokens(db: AsyncSession = Depends(get_db)):
    """Get all supported tokens with their current data"""
    result = await db.execute(select(TokenData))
    return result.scalars().all()

@router.get("/{symbol}", response_model=schemas.TokenDetailResponse)
async def get_token(
    symbol: str,
    db: AsyncSession = Depends(get_db)
):
    """Get detailed data for a specific token"""
    details = await service.get_token_details(symbol, db)
    if not details["token_data"]:
        raise HTTPException(status_code=404, detail="Token not found")
    return details

@router.get("/{symbol}/articles", response_model=List[schemas.Article])
async def get_token_articles(
    symbol: str,
    db: AsyncSession = Depends(get_db)
):
    """Get recent articles for a specific token"""
    result = await db.execute(
        select(Article)
        .where(Article.token == symbol.upper())
        .order_by(Article.timestamp.desc())
        .limit(5)
    )
    return result.scalars().all()

@router.get("/{symbol}/sentiment", response_model=List[schemas.SentimentHistory])
async def get_token_sentiment(
    symbol: str,
    db: AsyncSession = Depends(get_db)
):
    """Get sentiment history for a specific token"""
    result = await db.execute(
        select(SentimentHistory)
        .where(SentimentHistory.token == symbol.upper())
        .order_by(SentimentHistory.timestamp.desc())
        .limit(24)
    )
    return result.scalars().all()

# Admin endpoints (requires authentication)
@router.post("/update", status_code=204)
async def trigger_update(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Manually trigger token data update (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    await service.update_token_data(db)
    return {"message": "Update triggered successfully"} 