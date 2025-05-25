from pydantic import BaseModel, EmailStr
from typing import Optional
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
    market_cap: float
    volume: float

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