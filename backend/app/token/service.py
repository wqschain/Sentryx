import aiohttp
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from bs4 import BeautifulSoup
import asyncio
from time import time
import calendar
import logging
import os
from pathlib import Path
import random
import json
import shutil
from playwright.async_api import async_playwright

from app.core.config import settings
from app.models.models import TokenData, Article, SentimentHistory, APIUsage
from .api_monitor import api_monitor
from .sentiment_monitor import monitor
import time
from app.core.database import AsyncSessionLocal

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelVersion:
    def __init__(self, version: str, path: str, timestamp: str, metrics: dict):
        self.version = version
        self.path = path
        self.timestamp = timestamp
        self.metrics = metrics

class ModelManager:
    def __init__(self):
        self.models_dir = Path("models")
        self.versions_file = self.models_dir / "versions.json"
        self.current_version = None
        self._load_versions()
    
    def _load_versions(self):
        """Load model versions from versions.json"""
        if not self.versions_file.exists():
            self.versions = {}
            return
        
        with open(self.versions_file, 'r') as f:
            self.versions = json.load(f)
    
    def _save_versions(self):
        """Save model versions to versions.json"""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def save_model(self, model_path: str, metrics: dict, version: str = None):
        """Save a new model version"""
        if not version:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create version directory
        version_path = self.models_dir / f"v{version}"
        version_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        shutil.copytree(model_path, version_path / "model", dirs_exist_ok=True)
        
        # Save version info
        version_info = {
            "version": version,
            "path": str(version_path / "model"),
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        self.versions[version] = version_info
        self._save_versions()
        self.current_version = version
        
        logger.info(f"Saved model version {version}")
        return version_info
    
    def load_model(self, version: str = None):
        """Load a specific model version or the latest version"""
        if not version:
            if not self.versions:
                raise ValueError("No model versions available")
            version = max(self.versions.keys())
        
        if version not in self.versions:
            raise ValueError(f"Model version {version} not found")
        
        version_info = self.versions[version]
        model_path = version_info["path"]
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model files for version {version} not found")
        
        return model_path, version_info
    
    def get_version_info(self, version: str = None):
        """Get information about a specific version or the current version"""
        if not version:
            version = self.current_version
        return self.versions.get(version)
    
    def list_versions(self):
        """List all available model versions"""
        return self.versions

# Initialize model manager
if not os.path.exists(Path("models")):
    os.makedirs(Path("models"))
model_manager = ModelManager()

# Global model variables
tokenizer = None
model = None
device = None

def load_sentiment_model():
    """Load the sentiment analysis model with proper error handling and versioning"""
    global tokenizer, model, device
    
    try:
        # Try to load the improved model
        if os.path.exists(settings.SENTIMENT_MODEL_PATH):
            logger.info(f"Loading improved model from {settings.SENTIMENT_MODEL_PATH}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(settings.SENTIMENT_MODEL_PATH)
                model = AutoModelForSequenceClassification.from_pretrained(
                    settings.SENTIMENT_MODEL_PATH,
                    num_labels=3
                )
                logger.info("Successfully loaded improved model")
            except Exception as e:
                logger.error(f"Error loading improved model: {e}")
                logger.info("Falling back to base CryptoBERT...")
                tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
                model = AutoModelForSequenceClassification.from_pretrained("ElKulako/cryptobert", num_labels=3)
        else:
            logger.warning("Improved model not found, using base CryptoBERT")
            tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
            model = AutoModelForSequenceClassification.from_pretrained("ElKulako/cryptobert", num_labels=3)
        
        # Move model to appropriate device
        device = torch.device("cpu")  # Force CPU for now
        logger.info(f"Using device: {device}")
        model = model.to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        return tokenizer, model, device
    
    except Exception as e:
        logger.error(f"Critical error in model loading: {e}")
        raise

# Initialize model on module load
tokenizer, model, device = load_sentiment_model()

# Run model validation
async def validate_model():
    """Validate the sentiment model with test cases"""
    test_cases = [
        ("Bitcoin price surges to new all-time high", "positive"),
        ("Market crash wipes out billions in crypto value", "negative"),
        ("Bitcoin trading sideways in recent market activity", "neutral")
    ]
    
    logger.info("Validating sentiment model...")
    try:
        for text, expected in test_cases:
            sentiment, confidence = await analyze_sentiment(text)
            logger.info(f"Test case: {text}")
            logger.info(f"Expected: {expected}, Got: {sentiment} (confidence: {confidence:.2f})")
            
            if sentiment != expected:
                logger.warning(f"Validation failed for test case: {text}")
                logger.warning(f"Expected {expected} but got {sentiment}")
                return False
        
        logger.info("Model validation successful")
        return True
    except Exception as e:
        logger.error(f"Error during model validation: {e}")
        return False

# Don't run validation immediately
async def init_validation():
    return await validate_model()

async def update_token_data(db: AsyncSession) -> None:
    """Update price data for all supported tokens"""
    headers = {"X-CG-API-Key": settings.COINGECKO_API_KEY}
    
    async with aiohttp.ClientSession(headers=headers) as session:
        for symbol in settings.SUPPORTED_TOKENS:
            try:
                # Get current usage stats
                stats = await api_monitor.get_usage_stats(db)
                logger.info(f"Current API usage: {stats['total_calls']}/{settings.COINGECKO_MONTHLY_LIMIT} calls this month")
                
                if stats['current_minute_rate'] >= settings.COINGECKO_RATE_LIMIT:
                    wait_time = 60 - datetime.now(timezone.utc).second
                    logger.warning(f"Rate limit approaching, waiting {wait_time} seconds")
                    await asyncio.sleep(wait_time)
                
                coingecko_id = settings.COINGECKO_MAPPING.get(symbol.lower())
                if not coingecko_id:
                    continue

                url = f"{settings.COINGECKO_API_URL}/simple/price"
                params = {
                    "ids": coingecko_id,
                    "vs_currencies": "usd",
                    "include_market_cap": "true",
                    "include_24hr_vol": "true"
                }

                # Track the API request
                await api_monitor.track_request(db, "simple/price")
                
                async with session.get(url, params=params) as response:
                    if response.status == 429:  # Rate limit exceeded
                        logger.error(f"Rate limit exceeded for {symbol}")
                        wait_time = 60
                        if "Retry-After" in response.headers:
                            wait_time = int(response.headers["Retry-After"])
                        await asyncio.sleep(wait_time)
                        continue
                    elif response.status == 401:  # Invalid API key
                        logger.error("Invalid API key. Please check your CoinGecko API key.")
                        return
                    elif response.status == 200:
                        data = await response.json()
                        token_data = data[coingecko_id]
                        
                        # Update database
                        result = await db.execute(
                            select(TokenData).where(TokenData.symbol == symbol)
                        )
                        db_token = result.scalar_one_or_none()
                        
                        if db_token:
                            db_token.price = token_data["usd"]
                            db_token.market_cap = token_data["usd_market_cap"]
                            db_token.volume = token_data["usd_24h_vol"]
                            db_token.last_updated = datetime.utcnow()
                        else:
                            db_token = TokenData(
                                symbol=symbol,
                                price=token_data["usd"],
                                market_cap=token_data["usd_market_cap"],
                                volume=token_data["usd_24h_vol"]
                            )
                            db.add(db_token)
                        
                        logger.info(f"Successfully updated {symbol} data")
                
                await db.commit()
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")

async def analyze_sentiment(text: str) -> tuple[str, float]:
    """Analyze sentiment of text using improved CryptoBERT model"""
    global tokenizer, model, device
    
    try:
        # Input validation
        if not text or not isinstance(text, str):
            logger.warning("Invalid input text for sentiment analysis")
            return "neutral", 0.5
            
        # Truncate very long texts to prevent memory issues
        if len(text) > 10000:
            logger.warning("Text too long, truncating to 10000 characters")
            text = text[:10000]
        
        # Track processing time
        start_time = time.time()
        
        # Ensure model is in eval mode
        model.eval()
        
        # Tokenize and move to device
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            try:
                outputs = model(**inputs)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error("GPU out of memory, falling back to CPU")
                    model.to("cpu")
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    outputs = model(**inputs)
                    model.to(device)
                else:
                    raise
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_scores = probabilities[0].cpu().tolist()
        
        # Map model output to sentiment using settings
        predicted_class = torch.argmax(probabilities[0]).item()
        model_sentiment = list(settings.SENTIMENT_LABEL_MAP.keys())[predicted_class]
        mapped_sentiment = settings.SENTIMENT_LABEL_MAP[model_sentiment]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log prediction with monitoring
        prediction_scores = {
            settings.SENTIMENT_LABEL_MAP[label]: score 
            for label, score in zip(settings.SENTIMENT_LABEL_MAP.keys(), sentiment_scores)
        }
        
        monitor.log_prediction(
            text=text[:1000],  # Truncate text for storage
            predicted_sentiment=mapped_sentiment,
            confidence=sentiment_scores[predicted_class],
            category="general",
            prediction_scores=prediction_scores,
            processing_time=processing_time,
            model_version=settings.MODEL_VERSION
        )
        
        # Update model metrics periodically
        if random.random() < 0.1:  # 10% chance to update metrics
            asyncio.create_task(monitor.update_model_metrics())
        
        return mapped_sentiment, sentiment_scores[predicted_class]
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return "neutral", 0.5

async def update_model_metrics():
    """Update model metrics in the background"""
    try:
        monitor.update_model_metrics()
    except Exception as e:
        logger.error(f"Error updating model metrics: {e}")

# Schedule periodic model performance report
async def generate_performance_report():
    """Generate daily model performance report"""
    while True:
        try:
            async with AsyncSessionLocal() as db:
                report = await monitor.generate_report(db=db, days=1, save_path="sentiment_reports")
                logger.info("Generated daily model performance report")
                
                # Check for critical issues
                if report["model_performance"]["accuracy"] < 0.6:
                    logger.error("Critical: Model accuracy below 60%")
                
                if len(report["alerts"]) > 0:
                    logger.warning("Model alerts detected:")
                    for alert in report["alerts"]:
                        logger.warning(f"- {alert}")
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
        
        # Wait for 24 hours
        await asyncio.sleep(24 * 60 * 60)

# Start the performance monitoring
asyncio.create_task(generate_performance_report())

async def update_news_and_sentiment(db: AsyncSession) -> None:
    """Update news and sentiment data for all supported tokens"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        )
        
        try:
            page = await context.new_page()
            
            for token in settings.SUPPORTED_TOKENS:
                # Process both news and research sections
                for section, url in {
                    'news': settings.FORBES_NEWS_URL,
                    'research': settings.FORBES_RESEARCH_URL
                }.items():
                    
                    await page.goto(url, wait_until='networkidle')
                    await page.wait_for_selector('article.stream-item', timeout=10000)
                    
                    # Get article links
                    article_cards = await page.query_selector_all('article.stream-item')
                    for card in article_cards[:5]:  # Get top 5 articles
                        try:
                            title_elem = await card.query_selector('a.stream-item__title')
                            if title_elem:
                                title = await title_elem.text_content()
                                url = await title_elem.get_attribute('href')
                                
                                if not url.startswith('http'):
                                    url = f"https://www.forbes.com{url}"
                                
                                # Check if article exists
                                result = await db.execute(
                                    select(Article).where(Article.url == url)
                                )
                                if result.scalar_one_or_none():
                                    continue
                                
                                # Get article content
                                await page.goto(url, wait_until='networkidle')
                                content_elem = await page.query_selector('.article-body')
                                if content_elem:
                                    content = await content_elem.text_content()
                                    
                                    # Analyze sentiment
                                    sentiment, score = await analyze_sentiment(content)
                                    
                                    # Save article
                                    db_article = Article(
                                        token=token,
                                        title=title.strip(),
                                        url=url,
                                        content=content.strip(),
                                        source=f'Forbes {section.capitalize()}',
                                        sentiment=sentiment,
                                        score=score
                                    )
                                    db.add(db_article)
                        
                        except Exception as e:
                            logger.error(f"Error processing article: {e}")
                            continue
                
                # Calculate and save average sentiment
                result = await db.execute(
                    select(func.avg(Article.score))
                    .where(Article.token == token)
                    .where(Article.timestamp >= datetime.utcnow().replace(hour=0, minute=0))
                )
                avg_score = result.scalar_one_or_none() or 0.0
                
                sentiment_history = SentimentHistory(
                    token=token,
                    average_score=avg_score
                )
                db.add(sentiment_history)
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"Error in update_news_and_sentiment: {e}")
            await db.rollback()
            raise
        
        finally:
            await browser.close()

async def get_token_details(symbol: str, db: AsyncSession) -> dict:
    """Get detailed token data including price, articles, and sentiment"""
    # Get token data
    token_result = await db.execute(
        select(TokenData).where(TokenData.symbol == symbol.upper())
    )
    token_data = token_result.scalar_one_or_none()
    
    # Get recent articles
    articles_result = await db.execute(
        select(Article)
        .where(Article.token == symbol.upper())
        .order_by(Article.timestamp.desc())
        .limit(5)
    )
    articles = articles_result.scalars().all()
    
    # Get sentiment history
    sentiment_result = await db.execute(
        select(SentimentHistory)
        .where(SentimentHistory.token == symbol.upper())
        .order_by(SentimentHistory.timestamp.desc())
        .limit(24)  # Last 24 periods
    )
    sentiment_history = sentiment_result.scalars().all()
    
    # Calculate current sentiment (average of recent sentiments)
    current_sentiment = sum(s.average_score for s in sentiment_history[:3]) / 3 if sentiment_history else 0
    
    return {
        "token_data": token_data,
        "recent_articles": articles,
        "sentiment_history": sentiment_history,
        "current_sentiment": current_sentiment
    } 