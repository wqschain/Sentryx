"""
Sentiment analysis services package.
This package contains functionality for sentiment analysis and relevance detection.
"""

from .relevance_detection import check_relevance, get_relevance_explanation
from .blocked_content_detector import is_content_blocked, BlockedContentDetector
from .sentiment_monitor import monitor_sentiment, get_sentiment_metrics
from .sentiment_rules import apply_sentiment_rules, get_sentiment_score
from .service import load_sentiment_model
import torch
from app.schemas.sentiment import SentimentResponse
import logging
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

async def analyze_sentiment(text: str, db: AsyncSession = None) -> tuple[str, float]:
    """
    Analyze the sentiment of the given text using the loaded model.
    First checks if the text is relevant to cryptocurrency.
    """
    try:
        # Load model components
        tokenizer, model, device = load_sentiment_model()
        
        # Transform input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            
        # Map prediction to sentiment
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = sentiment_map[prediction.item()]
        confidence_value = float(confidence.item())
        
        # Monitor sentiment if db is provided
        if db:
            await monitor_sentiment(
                db=db,
                text=text,
                predicted_sentiment=sentiment,
                confidence=confidence_value,
                category="crypto",
                prediction_scores=get_sentiment_score(text),
                processing_time=0.0  # TODO: Add actual processing time
            )
        
        return sentiment, confidence_value
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return "neutral", 0.5

__all__ = [
    'check_relevance',
    'get_relevance_explanation',
    'is_content_blocked',
    'monitor_sentiment',
    'get_sentiment_metrics',
    'apply_sentiment_rules',
    'get_sentiment_score',
    'analyze_sentiment'
] 