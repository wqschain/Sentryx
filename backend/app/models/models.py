from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.sql import func
from datetime import datetime
from .base import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Article(Base):
    __tablename__ = "articles"
    
    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, index=True)
    title = Column(String)
    url = Column(String, unique=True)
    content = Column(Text)
    source = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    sentiment = Column(String)  # positive, negative, neutral
    score = Column(Float)

class SentimentHistory(Base):
    __tablename__ = "sentiment_history"
    
    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    average_score = Column(Float)

class TokenData(Base):
    __tablename__ = "token_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True)
    price = Column(Float)
    market_cap = Column(Float)
    volume = Column(Float)
    last_updated = Column(DateTime(timezone=True))

class APIUsage(Base):
    __tablename__ = "api_usage"
    
    id = Column(Integer, primary_key=True, index=True)
    month = Column(String, index=True)  # Format: YYYY-MM
    total_calls = Column(Integer, default=0)
    calls_by_endpoint = Column(JSON, default=dict)  # Store calls per endpoint
    last_updated = Column(DateTime, default=datetime.utcnow)
    minute_usage = Column(JSON, default=dict)  # Store minute-by-minute usage

class SentimentPrediction(Base):
    __tablename__ = 'sentiment_predictions'
    
    id = Column(Integer, primary_key=True)
    text = Column(String)
    predicted_sentiment = Column(String)
    actual_sentiment = Column(String, nullable=True)
    confidence = Column(Float)
    category = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String)
    prediction_scores = Column(JSON)  # Store all class probabilities
    processing_time = Column(Float)  # Track inference time

class ModelMetrics(Base):
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String)
    accuracy = Column(Float)
    confidence_mean = Column(Float)
    confidence_std = Column(Float)
    sentiment_distribution = Column(JSON)
    error_rate = Column(Float)
    performance_metrics = Column(JSON)  # Store detailed performance metrics 