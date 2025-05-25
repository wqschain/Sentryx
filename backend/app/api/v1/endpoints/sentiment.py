from fastapi import APIRouter, HTTPException
from app.schemas.sentiment import SentimentRequest, SentimentResponse
from app.services.sentiment import analyze_sentiment

router = APIRouter()

@router.post("/analyze", response_model=SentimentResponse)
async def analyze_text(request: SentimentRequest) -> SentimentResponse:
    """
    Analyze the sentiment of the provided text.
    """
    try:
        result = await analyze_sentiment(request.text)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) 