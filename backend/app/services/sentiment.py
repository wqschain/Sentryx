from app.services.sentiment.service import load_sentiment_model
from app.schemas.sentiment import SentimentResponse
from app.services.sentiment.relevance_detection import check_relevance, get_relevance_explanation
import torch
from typing import Tuple, Dict, List
import logging
from app.services.sentiment.blocked_content_detector import BlockedContentDetector

logger = logging.getLogger(__name__)

async def analyze_sentiment(text: str) -> SentimentResponse:
    """
    Analyze the sentiment of the given text using the loaded model.
    First checks if the text is relevant to cryptocurrency.
    """
    try:
        # Check relevance first
        is_relevant, relevance_score, matched_terms = check_relevance(text)
        relevance_explanation = get_relevance_explanation(matched_terms)
        
        # Transform matched_terms to expected format
        transformed_terms: Dict[str, List[str]] = {
            "crypto": matched_terms.get("crypto_terms", []),
            "context": matched_terms.get("context_terms", []),
            "manipulation": [str(matched_terms.get("manipulation", {}).get("score", 0))],
            "analysis": [term["term"] for term in matched_terms.get("analysis", {}).get("term_details", [])]
        }
        
        # If not relevant, return neutral sentiment with explanation
        if not is_relevant:
            detector = BlockedContentDetector()
            result = detector.is_blocked(text)
            return SentimentResponse(
                sentiment="neutral",
                confidence=0.0,
                is_relevant=False,
                relevance_score=relevance_score,
                relevance_explanation=relevance_explanation,
                matched_terms=transformed_terms,
                feedback=detector.get_feedback_message(result)
            )
        
        # If relevant, proceed with sentiment analysis
        tokenizer, model, device = load_sentiment_model()
        
        # Tokenize and prepare input
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
        
        return SentimentResponse(
            sentiment=sentiment,
            confidence=float(confidence.item()),
            is_relevant=True,
            relevance_score=relevance_score,
            relevance_explanation=relevance_explanation,
            matched_terms=transformed_terms,
            feedback=None
        )
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return SentimentResponse(
            sentiment="neutral",
            confidence=0.0,
            is_relevant=False,
            relevance_score=0.0,
            relevance_explanation="Error during analysis",
            matched_terms={},
            feedback=f"Error: {str(e)}"
        ) 