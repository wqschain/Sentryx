import asyncio
import logging
from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd

from .service import analyze_sentiment
from .sentiment_monitor import monitor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentTester:
    def __init__(self):
        self.test_cases = {
            "basic": [
                ("Bitcoin hits new all-time high of $100,000!", "positive"),
                ("Crypto market crashes, billions lost", "negative"),
                ("Bitcoin trading sideways at $50,000", "neutral")
            ],
            "technical": [
                ("RSI indicates oversold conditions on the 4-hour chart", "positive"),
                ("MACD showing bearish divergence", "negative"),
                ("Trading volume remains consistent with 30-day average", "neutral")
            ],
            "news": [
                ("Major bank announces crypto custody service", "positive"),
                ("Regulatory crackdown on crypto exchanges", "negative"),
                ("New crypto ETF application under review", "neutral")
            ],
            "mixed": [
                ("Despite market downturn, institutional adoption grows", "positive"),
                ("Technical indicators positive but regulatory concerns remain", "mixed"),
                ("Short-term bearish but long-term bullish outlook", "neutral")
            ],
            "edge_cases": [
                ("", "neutral"),  # Empty text
                ("A" * 11000, "neutral"),  # Very long text
                ("🚀🌙💎✨", "neutral")  # Emojis
            ]
        }
        
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)

    async def run_single_test(self, text: str, expected: str) -> Dict:
        """Run a single test case"""
        start_time = datetime.now()
        sentiment, confidence = await analyze_sentiment(text)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "expected": expected,
            "predicted": sentiment,
            "confidence": confidence,
            "processing_time": processing_time,
            "correct": sentiment == expected if expected != "mixed" else True
        }
        
        return result

    async def run_category_tests(self, category: str, test_cases: List) -> Dict:
        """Run tests for a specific category"""
        logger.info(f"\nTesting {category} cases:")
        results = []
        
        for text, expected in test_cases:
            result = await self.run_single_test(text, expected)
            results.append(result)
            
            logger.info(f"\nText: {result['text']}")
            logger.info(f"Expected: {expected}")
            logger.info(f"Predicted: {result['predicted']} (confidence: {result['confidence']:.2f})")
            logger.info(f"Processing time: {result['processing_time']:.3f}s")
            logger.info(f"{'✓' if result['correct'] else '✗'}")
        
        return results

    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate metrics for test results"""
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        
        metrics = {
            "accuracy": correct / total,
            "average_confidence": sum(r['confidence'] for r in results) / total,
            "average_processing_time": sum(r['processing_time'] for r in results) / total,
            "correct_predictions": correct,
            "total_predictions": total
        }
        
        return metrics

    async def run_all_tests(self):
        """Run all test cases and generate report"""
        all_results = {}
        all_metrics = {}
        
        for category, test_cases in self.test_cases.items():
            results = await self.run_category_tests(category, test_cases)
            metrics = self.calculate_metrics(results)
            
            all_results[category] = results
            all_metrics[category] = metrics
        
        # Calculate overall metrics
        total_correct = sum(m['correct_predictions'] for m in all_metrics.values())
        total_predictions = sum(m['total_predictions'] for m in all_metrics.values())
        
        overall_metrics = {
            "accuracy": total_correct / total_predictions,
            "average_confidence": sum(m['average_confidence'] * m['total_predictions'] 
                                   for m in all_metrics.values()) / total_predictions,
            "average_processing_time": sum(m['average_processing_time'] * m['total_predictions'] 
                                        for m in all_metrics.values()) / total_predictions
        }
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_metrics": overall_metrics,
            "category_metrics": all_metrics,
            "detailed_results": all_results
        }
        
        # Save report
        report_path = self.results_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self.print_summary(report)
        
        return report

    def print_summary(self, report: Dict):
        """Print test results summary"""
        logger.info("\n" + "="*50)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*50)
        
        logger.info("\nOverall Metrics:")
        for metric, value in report['overall_metrics'].items():
            logger.info(f"{metric}: {value:.3f}")
        
        logger.info("\nCategory Results:")
        for category, metrics in report['category_metrics'].items():
            logger.info(f"\n{category}:")
            logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
            logger.info(f"Avg Confidence: {metrics['average_confidence']:.3f}")
            logger.info(f"Avg Processing Time: {metrics['average_processing_time']:.3f}s")
        
        logger.info(f"\nDetailed report saved to: {self.results_dir}")

async def run_tests():
    """Main function to run tests"""
    tester = SentimentTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(run_tests()) 