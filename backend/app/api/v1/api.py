from fastapi import APIRouter
from app.api.v1.endpoints import tokens, sentiment

api_router = APIRouter()

api_router.include_router(tokens.router, prefix="/token", tags=["token"])
api_router.include_router(sentiment.router, prefix="/sentiment", tags=["sentiment"]) 