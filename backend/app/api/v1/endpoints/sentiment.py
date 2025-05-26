from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.sentiment import SentimentRequest, SentimentResponse
from app.services.sentiment import analyze_sentiment
from app.services.sentiment.relevance_detection import check_relevance, get_relevance_explanation
from app.core.database import get_db
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/analyze", response_model=SentimentResponse)
async def analyze_text(request: SentimentRequest, db: AsyncSession = Depends(get_db)) -> SentimentResponse:
    """
    Analyze the sentiment of the provided text.
    """
    try:
        # Check relevance
        logger.info("Checking text relevance...")
        is_relevant, relevance_score, relevance_details = check_relevance(request.text)
        logger.info(f"Relevance check complete: is_relevant={is_relevant}, score={relevance_score}")
        
        # Transform matched terms into expected format
        matched_terms = {
            "crypto": relevance_details.get("crypto_terms", []),
            "context": relevance_details.get("context_terms", []),
            "manipulation": [str(relevance_details.get("manipulation", {}).get("score", 0))],
            "analysis": [term["term"] for term in relevance_details.get("analysis", {}).get("term_details", [])]
        }
        
        # Only analyze sentiment if relevant
        logger.info("Analyzing sentiment...")
        sentiment, confidence = await analyze_sentiment(request.text, db=db)
        logger.info(f"Sentiment analysis complete: sentiment={sentiment}, confidence={confidence}")
        
        # Generate explanation based on matched terms
        if not is_relevant:
            relevance_explanation = "Not relevant to cryptocurrency"
        else:
            crypto_terms = len(relevance_details.get("crypto_terms", []))
            context_terms = len(relevance_details.get("context_terms", []))
            relevance_explanation = f"Found {crypto_terms} crypto-specific terms and {context_terms} context-related terms"
        
        response = SentimentResponse(
            sentiment=sentiment,
            confidence=confidence,
            is_relevant=is_relevant,
            relevance_score=relevance_score,
            relevance_explanation=relevance_explanation,
            matched_terms=matched_terms,
            feedback=None
        )
        logger.info("Successfully created response")
        return response
    except Exception as e:
        logger.error(f"Error in analyze_text: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing text: {str(e)}"
        ) 