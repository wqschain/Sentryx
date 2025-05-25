from app.token.service import load_sentiment_model
from app.schemas.sentiment import SentimentResponse
import torch
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

async def analyze_sentiment(text: str) -> SentimentResponse:
    """
    Analyze the sentiment of the given text using the loaded model.
    """
    try:
        tokenizer, model, device = load_sentiment_model()
        
        # Tokenize and prepare input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities).item()
        
        # Map prediction to sentiment
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = sentiment_map[prediction.item()]
        
        return SentimentResponse(
            sentiment=sentiment,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise 