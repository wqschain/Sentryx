from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# User schemas
class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserLogin(UserBase):
    password: str

class User(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Token schemas
class Token(BaseModel):
    access_token: str
    token_type: str

# Article schemas
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

# Token data schemas
class TokenDataBase(BaseModel):
    symbol: str
    price: float
    price_change_24h: float = 0
    market_cap: float
    volume: float
    high_24h: float
    low_24h: float
    circulating_supply: float
    max_supply: Optional[float] = None
    market_rank: int
    ath: float
    last_updated: datetime

    class Config:
        from_attributes = True

class TokenDataCreate(TokenDataBase):
    pass

class TokenData(TokenDataBase):
    id: int
    last_updated: datetime
    
    class Config:
        from_attributes = True

# Sentiment history schemas
class SentimentHistory(BaseModel):
    token: str
    average_score: float
    timestamp: datetime
    
    class Config:
        from_attributes = True

class TokenPriceHistoryBase(BaseModel):
    token_symbol: str
    price: float
    timestamp: datetime

    class Config:
        from_attributes = True

class TokenVolumeHistoryBase(BaseModel):
    token_symbol: str
    volume: float
    timestamp: datetime

    class Config:
        from_attributes = True

class TokenPriceHistoryCreate(TokenPriceHistoryBase):
    pass

class TokenVolumeHistoryCreate(TokenVolumeHistoryBase):
    pass

class TokenPriceHistory(TokenPriceHistoryBase):
    id: int

class TokenVolumeHistory(TokenVolumeHistoryBase):
    id: int

class TokenPriceHistoryResponse(BaseModel):
    data: List[TokenPriceHistory]

class TokenVolumeHistoryResponse(BaseModel):
    data: List[TokenVolumeHistory]

class TokenBase(BaseModel):
    symbol: str
    price: float
    price_change_24h: float
    market_cap: float
    volume: float
    high_24h: float
    low_24h: float
    circulating_supply: float
    max_supply: Optional[float] = None
    market_rank: int
    ath: float
    last_updated: datetime

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    symbol: str
    price: float
    price_change_24h: float
    market_cap: float
    volume: float
    high_24h: float
    low_24h: float
    circulating_supply: float
    max_supply: Optional[float]
    market_rank: int
    ath: float
    last_updated: datetime

    class Config:
        from_attributes = True

class TokenPriceHistory(BaseModel):
    token_symbol: str
    price: float
    timestamp: datetime

    class Config:
        from_attributes = True

class TokenVolumeHistory(BaseModel):
    token_symbol: str
    volume: float
    timestamp: datetime

    class Config:
        from_attributes = True

class TokenPriceHistoryResponse(BaseModel):
    data: List[TokenPriceHistory]

class TokenVolumeHistoryResponse(BaseModel):
    data: List[TokenVolumeHistory] 