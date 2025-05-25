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

# Response models for aggregated data
class TokenDetailResponse(BaseModel):
    token_data: TokenData
    recent_articles: list[Article]
    sentiment_history: list[SentimentHistory]
    current_sentiment: float  # Average of recent sentiments 