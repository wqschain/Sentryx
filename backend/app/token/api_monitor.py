from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import json
from app.models.models import APIUsage

class APIMonitor:
    def __init__(self):
        self._current_minute_calls = 0
        self._current_minute = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        
    async def track_request(self, db: AsyncSession, endpoint: str):
        """Track an API request"""
        current_month = datetime.now(timezone.utc).strftime("%Y-%m")
        current_minute = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        
        # Get or create usage record for current month
        result = await db.execute(
            select(APIUsage).where(APIUsage.month == current_month)
        )
        usage = result.scalar_one_or_none()
        
        if not usage:
            usage = APIUsage(
                month=current_month,
                total_calls=0,
                calls_by_endpoint={},
                minute_usage={}
            )
            db.add(usage)
        
        # Update total calls
        usage.total_calls += 1
        
        # Update endpoint-specific calls
        calls_by_endpoint = usage.calls_by_endpoint or {}
        calls_by_endpoint[endpoint] = calls_by_endpoint.get(endpoint, 0) + 1
        usage.calls_by_endpoint = calls_by_endpoint
        
        # Update minute usage
        minute_usage = usage.minute_usage or {}
        minute_usage[current_minute] = minute_usage.get(current_minute, 0) + 1
        usage.minute_usage = minute_usage
        
        # Update timestamp
        usage.last_updated = datetime.now(timezone.utc)
        
        await db.commit()
    
    async def get_usage_stats(self, db: AsyncSession):
        """Get current API usage statistics"""
        current_month = datetime.now(timezone.utc).strftime("%Y-%m")
        result = await db.execute(
            select(APIUsage).where(APIUsage.month == current_month)
        )
        usage = result.scalar_one_or_none()
        
        if not usage:
            return {
                "month": current_month,
                "total_calls": 0,
                "remaining_calls": 10000,
                "calls_by_endpoint": {},
                "current_minute_rate": 0
            }
        
        # Calculate current minute rate
        current_minute = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        current_rate = usage.minute_usage.get(current_minute, 0) if usage.minute_usage else 0
        
        return {
            "month": usage.month,
            "total_calls": usage.total_calls,
            "remaining_calls": 10000 - usage.total_calls,
            "calls_by_endpoint": usage.calls_by_endpoint,
            "current_minute_rate": current_rate
        }

# Global instance
api_monitor = APIMonitor() 