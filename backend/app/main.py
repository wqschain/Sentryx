from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.api import api_router
from app.core.scheduler import setup_scheduler
from app.core.init_db import init_db
from app.services.sentiment.service import load_sentiment_model
import logging
import asyncio
from app.core.tasks import start_background_tasks

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    try:
        # Initialize database
        logger.info("Initializing database...")
        await init_db()
        
        # Load sentiment model
        logger.info("Loading sentiment model...")
        tokenizer, model, device = load_sentiment_model()
        logger.info(f"Model loaded successfully on {device}")
        
        # Set up background tasks scheduler
        logger.info("Setting up background tasks...")
        setup_scheduler()
        
        # Start the background tasks in a separate task
        asyncio.create_task(start_background_tasks())
        logger.info("Started background tasks")
        
        logger.info("Application startup complete!")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "Welcome to Sentryx API"} 