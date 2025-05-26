from datetime import datetime, timedelta, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import json
import logging
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from app.core.config import settings
from app.models.models import SentimentPrediction, ModelMetrics, Base

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentMonitor:
    def __init__(self):
        # Create reports directory
        self.reports_dir = Path("sentiment_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # Performance thresholds
        self.accuracy_threshold = 0.7
        self.confidence_threshold = 0.6
        self.error_rate_threshold = 0.1
        
        # Data retention settings (in days)
        self.prediction_retention = 30
        self.metrics_retention = 90
        self.report_retention = 180
    
    async def log_prediction(
        self,
        db: AsyncSession,
        text: str,
        predicted_sentiment: str,
        confidence: float,
        category: str,
        prediction_scores: Dict[str, float],
        processing_time: float,
        actual_sentiment: Optional[str] = None,
        model_version: str = settings.MODEL_VERSION
    ):
        """Log a sentiment prediction with enhanced metrics"""
        try:
            prediction = SentimentPrediction(
                text=text,
                predicted_sentiment=predicted_sentiment,
                actual_sentiment=actual_sentiment,
                confidence=confidence,
                category=category,
                model_version=model_version,
                prediction_scores=prediction_scores,
                processing_time=processing_time
            )
            db.add(prediction)
            await db.commit()
            
            # Check for potential issues
            await self._check_prediction_quality(db, prediction)
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
            await db.rollback()
            raise
    
    async def _check_prediction_quality(self, db: AsyncSession, prediction: SentimentPrediction):
        """Check prediction quality and log warnings"""
        # Check confidence
        if prediction.confidence < self.confidence_threshold:
            logger.warning(f"Low confidence prediction ({prediction.confidence:.2f}) for text: {prediction.text[:100]}...")
        
        # Check for high uncertainty (similar scores for different classes)
        scores = prediction.prediction_scores
        if max(scores.values()) - min(scores.values()) < 0.3:
            logger.warning(f"High uncertainty in prediction for text: {prediction.text[:100]}...")
        
        # Check processing time
        if prediction.processing_time > 1.0:  # More than 1 second
            logger.warning(f"Slow prediction ({prediction.processing_time:.2f}s) for text: {prediction.text[:100]}...")
    
    async def update_model_metrics(self, db: AsyncSession):
        """Update and store model performance metrics"""
        try:
            # Get recent predictions
            recent_predictions = await db.execute(
                select(SentimentPrediction)
                .where(SentimentPrediction.timestamp >= datetime.utcnow() - timedelta(days=1))
            )
            recent_predictions = recent_predictions.scalars().all()
            
            if not recent_predictions:
                return
            
            # Calculate metrics
            total = len(recent_predictions)
            correct = sum(1 for p in recent_predictions 
                         if p.actual_sentiment and p.predicted_sentiment == p.actual_sentiment)
            
            confidences = [p.confidence for p in recent_predictions]
            processing_times = [p.processing_time for p in recent_predictions]
            
            metrics = ModelMetrics(
                model_version=settings.MODEL_VERSION,
                accuracy=correct/total if total > 0 else 0,
                confidence_mean=np.mean(confidences),
                confidence_std=np.std(confidences),
                sentiment_distribution={
                    "positive": sum(1 for p in recent_predictions if p.predicted_sentiment == "positive") / total,
                    "negative": sum(1 for p in recent_predictions if p.predicted_sentiment == "negative") / total,
                    "neutral": sum(1 for p in recent_predictions if p.predicted_sentiment == "neutral") / total
                },
                error_rate=sum(1 for p in recent_predictions if p.processing_time > 1.0) / total,
                performance_metrics={
                    "avg_processing_time": np.mean(processing_times),
                    "max_processing_time": max(processing_times),
                    "95th_percentile_time": np.percentile(processing_times, 95)
                }
            )
            
            db.add(metrics)
            await db.commit()
            
            # Check for model drift
            await self._check_model_drift(metrics)
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            await db.rollback()
            raise
    
    async def _check_model_drift(self, metrics: ModelMetrics):
        """Check for model drift and performance degradation"""
        if metrics.accuracy < self.accuracy_threshold:
            logger.warning(f"Model accuracy ({metrics.accuracy:.2f}) below threshold ({self.accuracy_threshold})")
        
        if metrics.error_rate > self.error_rate_threshold:
            logger.warning(f"High error rate ({metrics.error_rate:.2f}) detected")
        
        # Check for unusual sentiment distribution
        for sentiment, ratio in metrics.sentiment_distribution.items():
            if ratio > 0.6:  # More than 60% predictions in one class
                logger.warning(f"Possible bias detected: {sentiment} predictions are {ratio:.1%} of total")
        
        # Check for significant confidence changes
        if metrics.confidence_std > 0.25:  # High variance in confidence
            logger.warning("High variance in prediction confidence detected")
    
    async def generate_report(
        self,
        db: AsyncSession,
        days: int = 7,
        save_path: Optional[str] = None
    ) -> Dict:
        """Generate a comprehensive sentiment analysis report with enhanced metrics"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        try:
            # Get predictions and metrics
            predictions_result = await db.execute(
                select(SentimentPrediction)
                .where(SentimentPrediction.timestamp >= start_date)
                .where(SentimentPrediction.timestamp <= end_date)
            )
            predictions = predictions_result.scalars().all()
            
            metrics_result = await db.execute(
                select(ModelMetrics)
                .where(ModelMetrics.timestamp >= start_date)
                .where(ModelMetrics.timestamp <= end_date)
            )
            metrics = metrics_result.scalars().all()
            
            # Convert to pandas DataFrames
            predictions_df = pd.DataFrame([vars(p) for p in predictions])
            metrics_df = pd.DataFrame([vars(m) for m in metrics])
            
            # Generate visualizations
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Accuracy over time
            plt.subplot(2, 2, 1)
            plt.plot(metrics_df['timestamp'], metrics_df['accuracy'])
            plt.title('Model Accuracy Over Time')
            
            # Plot 2: Confidence distribution
            plt.subplot(2, 2, 2)
            plt.hist(predictions_df['confidence'], bins=20)
            plt.title('Prediction Confidence Distribution')
            
            # Plot 3: Processing time distribution
            plt.subplot(2, 2, 3)
            plt.hist(predictions_df['processing_time'], bins=20)
            plt.title('Processing Time Distribution')
            
            # Plot 4: Sentiment distribution
            plt.subplot(2, 2, 4)
            sentiment_counts = predictions_df['predicted_sentiment'].value_counts()
            plt.bar(sentiment_counts.index, sentiment_counts.values)
            plt.title('Sentiment Distribution')
            
            # Save plots
            if save_path:
                plot_path = Path(save_path) / f"model_performance_{end_date.strftime('%Y%m%d')}.png"
                plt.savefig(plot_path)
                plt.close()
            
            # Generate report
            report = {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "days": days
                },
                "model_performance": {
                    "accuracy": metrics_df['accuracy'].mean(),
                    "confidence": {
                        "mean": predictions_df['confidence'].mean(),
                        "std": predictions_df['confidence'].std(),
                        "min": predictions_df['confidence'].min(),
                        "max": predictions_df['confidence'].max()
                    },
                    "processing_time": {
                        "mean": predictions_df['processing_time'].mean(),
                        "95th_percentile": predictions_df['processing_time'].quantile(0.95),
                        "max": predictions_df['processing_time'].max()
                    },
                    "sentiment_distribution": predictions_df['predicted_sentiment'].value_counts().to_dict()
                },
                "alerts": await self._generate_alerts(predictions_df, metrics_df),
                "recommendations": await self._generate_recommendations(predictions_df, metrics_df)
            }
            
            # Save report
            if save_path:
                report_path = Path(save_path) / f"model_performance_report_{end_date.strftime('%Y%m%d')}.json"
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
            
            return report
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    async def cleanup_old_data(self, db: AsyncSession):
        """Clean up old monitoring data based on retention settings"""
        try:
            # Calculate cutoff dates
            prediction_cutoff = datetime.utcnow() - timedelta(days=self.prediction_retention)
            metrics_cutoff = datetime.utcnow() - timedelta(days=self.metrics_retention)
            
            # Delete old predictions
            await db.execute(
                select(SentimentPrediction)
                .where(SentimentPrediction.timestamp < prediction_cutoff)
                .delete()
            )
            
            # Delete old metrics
            await db.execute(
                select(ModelMetrics)
                .where(ModelMetrics.timestamp < metrics_cutoff)
                .delete()
            )
            
            await db.commit()
            
            # Clean up old report files
            report_cutoff = datetime.utcnow() - timedelta(days=self.report_retention)
            for report_file in self.reports_dir.glob("*"):
                try:
                    file_date = datetime.strptime(report_file.stem.split("_")[-1], "%Y%m%d")
                    if file_date.replace(tzinfo=timezone.utc) < report_cutoff:
                        report_file.unlink()
                except (ValueError, OSError) as e:
                    logger.warning(f"Error cleaning up report file {report_file}: {e}")
            
            logger.info("Successfully cleaned up old monitoring data")
        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")
            await db.rollback()
            raise

# Global monitor instance
_monitor = SentimentMonitor()

async def monitor_sentiment(
    db: AsyncSession,
    text: str,
    predicted_sentiment: str,
    confidence: float,
    category: str,
    prediction_scores: Dict[str, float],
    processing_time: float,
    actual_sentiment: Optional[str] = None,
    model_version: str = settings.MODEL_VERSION
) -> None:
    """
    Monitor a sentiment prediction.
    
    Args:
        db: Database session
        text: The analyzed text
        predicted_sentiment: The predicted sentiment
        confidence: Prediction confidence
        category: Content category
        prediction_scores: Raw prediction scores
        processing_time: Time taken for prediction
        actual_sentiment: Actual sentiment if known
        model_version: Model version used
    """
    await _monitor.log_prediction(
        db=db,
        text=text,
        predicted_sentiment=predicted_sentiment,
        confidence=confidence,
        category=category,
        prediction_scores=prediction_scores,
        processing_time=processing_time,
        actual_sentiment=actual_sentiment,
        model_version=model_version
    )
    await _monitor.update_model_metrics(db)

async def get_sentiment_metrics(
    db: AsyncSession,
    days: int = 7
) -> Dict:
    """
    Get sentiment analysis metrics for the specified period.
    
    Args:
        db: Database session
        days: Number of days to analyze
        
    Returns:
        Dict containing sentiment metrics and visualizations
    """
    return await _monitor.generate_report(db, days) 