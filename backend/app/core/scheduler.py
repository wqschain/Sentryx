from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import logging
import asyncio

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.services.news_service import update_news_and_sentiment
from app.services.price_service import update_token_prices
from app.services.sentiment.sentiment_monitor import SentimentMonitor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create monitor instance
monitor = SentimentMonitor()

class TaskScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self._setup_tasks()
    
    async def _run_with_db(self, func, *args, **kwargs):
        """Run a function with a database session"""
        async with AsyncSessionLocal() as db:
            try:
                await func(db, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in scheduled task {func.__name__}: {e}")
                await db.rollback()
            else:
                await db.commit()
    
    def _setup_tasks(self):
        """Set up scheduled tasks"""
        # Schedule news and sentiment updates
        self.scheduler.add_job(
            self._run_with_db,
            trigger=IntervalTrigger(hours=settings.SCRAPING_INTERVAL_HOURS),
            args=[update_news_and_sentiment],
            id='update_news_sentiment',
            name='Update news and sentiment data',
            replace_existing=True
        )
        
        # Schedule price updates (every minute)
        self.scheduler.add_job(
            self._run_with_db,
            trigger=IntervalTrigger(minutes=settings.PRICE_UPDATE_INTERVAL_MINUTES),
            args=[update_token_prices],
            id='update_prices',
            name='Update token prices',
            replace_existing=True
        )
        
        # Clean up old monitoring data daily at 2 AM
        self.scheduler.add_job(
            self._run_with_db,
            CronTrigger(hour=2),
            args=[monitor.cleanup_old_data],
            name="cleanup_monitoring_data",
            misfire_grace_time=3600
        )
        
        # Export monitoring data weekly on Sunday at 1 AM
        self.scheduler.add_job(
            self._export_weekly_data,
            CronTrigger(day_of_week="sun", hour=1),
            name="export_weekly_data",
            misfire_grace_time=3600
        )
    
    async def _export_weekly_data(self):
        """Export weekly monitoring data"""
        try:
            async with AsyncSessionLocal() as db:
                export_path = await monitor.export_monitoring_data(db, format="json")
                logger.info(f"Weekly monitoring data exported to {export_path}")
        except Exception as e:
            logger.error(f"Error exporting weekly monitoring data: {e}")
    
    def start(self):
        """Start the scheduler"""
        self.scheduler.start()
        logger.info("Task scheduler started")
    
    def shutdown(self):
        """Shutdown the scheduler"""
        self.scheduler.shutdown()
        logger.info("Task scheduler shutdown")

# Create scheduler instance
scheduler = TaskScheduler()

def setup_scheduler():
    """Initialize and start the scheduler"""
    scheduler.start()
    return scheduler 