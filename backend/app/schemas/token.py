from pydantic import BaseModel
from datetime import datetime

class TokenDataBase(BaseModel):
    symbol: str
    price: float
    market_cap: float
    volume: float

class TokenData(TokenDataBase):
    id: int
    last_updated: datetime

    class Config:
        from_attributes = True

class ArticleBase(BaseModel):
    token: str
    title: str
    url: str
    content: str
    source: str

class Article(ArticleBase):
    id: int
    sentiment: str
    score: float
    timestamp: datetime

    class Config:
        from_attributes = True

class SentimentHistoryBase(BaseModel):
    token: str
    average_score: float

class SentimentHistory(SentimentHistoryBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True

class TokenPriceHistory(BaseModel):
    id: int
    token_symbol: str
    price: float
    timestamp: datetime

    class Config:
        from_attributes = True

class TokenVolumeHistory(BaseModel):
    id: int
    token_symbol: str
    volume: float
    timestamp: datetime

    class Config:
        from_attributes = True

# Response models for aggregated data
class TokenDetailResponse(BaseModel):
    id: int
    symbol: str
    price: float
    price_change_24h: float
    market_cap: float
    volume: float
    high_24h: float
    low_24h: float
    circulating_supply: float
    max_supply: float | None = None
    market_rank: int | None = None
    ath: float | None = None
    last_updated: str
    price_history: list[dict]
    volume_history: list[dict]
    recent_articles: list[Article]
    sentiment_history: list[SentimentHistory]
    current_sentiment: float  # Average of recent sentiments 