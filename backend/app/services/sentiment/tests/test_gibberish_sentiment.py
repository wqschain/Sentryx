"""Tests for gibberish sentiment detection."""
import pytest
import logging
from typing import List, Tuple

from app.services.sentiment.relevance_detection import check_relevance, get_relevance_explanation
from app.services.sentiment.sentiment_rules import adjust_sentiment
import sys
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_sentiment_direct(text: str, tokenizer, model) -> tuple[str, float]:
    """Direct sentiment analysis without service layer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        confidence = torch.max(probabilities).item()
    
    # Map prediction to sentiment (0: negative, 1: neutral, 2: positive)
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    sentiment = sentiment_map[prediction.item()]
    
    return sentiment, confidence

def test_gibberish_with_sentiment():
    """Test how gibberish text with crypto terms is handled by both systems."""
    
    # Load our fine-tuned model
    logger.info("Loading fine-tuned sentiment model...")
    model_path = "/Users/waqasrana/sentryx_V2/backend/models/SentryxAI"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fine-tuned model not found at {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
    model.eval()
    
    test_cases = [
        "afjusahjfaof bitcoin afjusahjfaof",  # Gibberish with bitcoin
        "bitcoin",  # Just bitcoin for comparison
        "afjusahjfaof",  # Pure gibberish
        "bitcoin is going to the moon afjusahjfaof",  # Gibberish with sentiment
        "afjusahjfaof bitcoin crashed today afjusahjfaof",  # Gibberish with negative sentiment
    ]
    
    logger.info("\nTesting Gibberish Text Analysis:")
    
    for text in test_cases:
        logger.info("\n" + "="*50)
        logger.info(f"Testing text: '{text}'")
        
        # Step 1: Check relevance
        is_relevant, relevance_score, matched_terms = check_relevance(text)
        relevance_explanation = get_relevance_explanation(matched_terms)
        
        logger.info("\nRelevance Results:")
        logger.info(f"Is Relevant: {is_relevant}")
        logger.info(f"Relevance Score: {relevance_score:.2f}")
        logger.info(f"Explanation: {relevance_explanation}")
        
        # Step 2: If relevant, check sentiment
        if is_relevant:
            logger.info("\nSentiment Results:")
            sentiment, confidence = analyze_sentiment_direct(text, tokenizer, model)
            logger.info(f"Sentiment: {sentiment}")
            logger.info(f"Confidence: {confidence:.2f}")
        else:
            logger.info("\nSkipping sentiment analysis (not relevant)")
        
        logger.info("="*50)

if __name__ == "__main__":
    test_gibberish_with_sentiment() 