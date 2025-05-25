from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.core.database import get_db
from app.schemas.schemas import TokenData, Article, SentimentHistory
from app.models.models import TokenData as TokenDataModel
from app.models.models import Article as ArticleModel
from app.models.models import SentimentHistory as SentimentHistoryModel
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from datetime import datetime

router = APIRouter()

@router.get("/", response_model=List[TokenData])
async def get_tokens(db: AsyncSession = Depends(get_db)):
    """Get all supported tokens with their current data"""
    try:
        result = await db.execute(select(TokenDataModel))
        tokens = result.scalars().all()
        # Ensure last_updated is not None
        current_time = datetime.utcnow()
        for token in tokens:
            if token.last_updated is None:
                token.last_updated = current_time
        return tokens
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{symbol}", response_model=TokenData)
async def get_token(symbol: str, db: AsyncSession = Depends(get_db)):
    """Get detailed data for a specific token"""
    try:
        result = await db.execute(
            select(TokenDataModel).where(TokenDataModel.symbol == symbol.upper())
        )
        token = result.scalar_one_or_none()
        if not token:
            raise HTTPException(status_code=404, detail="Token not found")
        # Ensure last_updated is not None
        if token.last_updated is None:
            token.last_updated = datetime.utcnow()
        return token
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{symbol}/articles", response_model=List[Article])
async def get_token_articles(symbol: str, db: AsyncSession = Depends(get_db)):
    """Get recent articles for a specific token"""
    try:
        result = await db.execute(
            select(ArticleModel)
            .where(ArticleModel.token == symbol.upper())
            .order_by(ArticleModel.timestamp.desc())
            .limit(3)
        )
        articles = result.scalars().all()
        return articles
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{symbol}/sentiment", response_model=List[SentimentHistory])
async def get_token_sentiment(symbol: str, db: AsyncSession = Depends(get_db)):
    """Get sentiment history for a specific token"""
    try:
        result = await db.execute(
            select(SentimentHistoryModel)
            .where(SentimentHistoryModel.token == symbol.upper())
            .order_by(SentimentHistoryModel.timestamp.desc())
            .limit(24)  # Last 24 periods (6-hour intervals = 6 days)
        )
        sentiment_history = result.scalars().all()
        return sentiment_history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 