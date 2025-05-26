from pydantic import BaseModel
from typing import Dict, Optional, List

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    is_relevant: bool
    relevance_score: float
    relevance_explanation: str
    matched_terms: Dict[str, List[str]]
    feedback: Optional[str] = None 