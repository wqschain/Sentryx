from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from app.token.service import update_token_data
from app.services.news_service import update_news_and_sentiment
from app.core.database import AsyncSessionLocal
import logging

logger = logging.getLogger(__name__)

async def setup_scheduler():
    """Set up the scheduler for periodic tasks"""
    scheduler = AsyncIOScheduler()
    
    # Add token data update job (every 5 minutes)
    scheduler.add_job(
        update_token_data,
        IntervalTrigger(minutes=5),
        id='update_token_data',
        name='Update token data',
        replace_existing=True
    )
    
    # Add news and sentiment update job (every 30 minutes)
    scheduler.add_job(
        update_news_and_sentiment,
        IntervalTrigger(minutes=30),
        id='update_news_and_sentiment',
        name='Update news and sentiment',
        replace_existing=True,
        kwargs={'db': AsyncSessionLocal()}
    )
    
    scheduler.start()
    logger.info("Scheduler started") 