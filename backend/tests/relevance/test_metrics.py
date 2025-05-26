"""
Comprehensive metrics test suite for cryptocurrency relevance detection.
"""
import pytest
from app.services.sentiment.relevance_detection import (
    Term, TermType, normalize_text, check_relevance,
    RelevanceDetector
)
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class TestMetrics:
    """Metrics for test results."""
    total_tests: int
    successful_tests: int
    false_positives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_confidence: float
    avg_processing_time: float
    category_accuracy: Dict[str, float]

class TestDataset:
    """Test dataset with known outcomes."""
    
    def __init__(self):
        # Format: (text, expected_relevant, category)
        self.test_cases: List[Tuple[str, bool, str]] = [
            # Basic Cryptocurrency Terms (Category: basic)
            ("Bitcoin and Ethereum are leading cryptocurrencies", True, "basic"),
            ("BTC price is rising while ETH consolidates", True, "basic"),
            ("The weather is nice today", False, "basic"),
            ("Just bought some groceries", False, "basic"),
            
            # Trading Pairs (Category: trading)
            ("BTC/USDT trading at 50,000", True, "trading"),
            ("ETH/BTC pair showing strength", True, "trading"),
            ("USD/EUR forex trading", False, "trading"),
            ("Trading stocks on NYSE", False, "trading"),
            
            # Technical Terms (Category: technical)
            ("The blockchain network uses smart contracts", True, "technical"),
            ("Implementing zero-knowledge proofs on Ethereum", True, "technical"),
            ("Database uses token authentication", False, "technical"),
            ("The API token needs to be renewed", False, "technical"),
            
            # DeFi Terms (Category: defi)
            ("Yield farming in liquidity pools", True, "defi"),
            ("Providing liquidity to AMM protocols", True, "defi"),
            ("Farming vegetables in the garden", False, "defi"),
            ("Pool party this weekend", False, "defi"),
            
            # Mixed Content (Category: mixed)
            ("Bitcoin trading while checking stock market", True, "mixed"),
            ("Mining ETH and playing games", True, "mixed"),
            ("Playing games with tokens", False, "mixed"),
            ("Token system in the game", False, "mixed"),
            
            # Context Sensitivity (Category: context)
            ("The market analysis shows bullish trends for BTC", True, "context"),
            ("Cryptocurrency prices are volatile", True, "context"),
            ("Market analysis for commodities", False, "context"),
            ("Bullish on tech stocks", False, "context"),
            
            # Edge Cases (Category: edge)
            ("btc", True, "edge"),
            ("blockchain", True, "edge"),
            ("token", False, "edge"),
            ("mining", False, "edge"),
            
            # Sentiment (Category: sentiment)
            ("Bullish on Bitcoin's future", True, "sentiment"),
            ("Crypto market showing strength 🚀", True, "sentiment"),
            ("Bearish on tech stocks", False, "sentiment"),
            ("Market crash incoming", False, "sentiment"),
            
            # Complex Cases (Category: complex)
            ("Implementing DeFi protocols on Ethereum L2 solutions", True, "complex"),
            ("Smart contract audit revealed security concerns", True, "complex"),
            ("Smart contract for real estate", False, "complex"),
            ("Blockchain technology in supply chain", False, "complex"),
        ]

    def get_categories(self) -> List[str]:
        """Get unique categories."""
        return list(set(cat for _, _, cat in self.test_cases))

    def get_cases_by_category(self, category: str) -> List[Tuple[str, bool]]:
        """Get test cases for a specific category."""
        return [(text, expected) for text, expected, cat in self.test_cases if cat == category]

def run_metrics_test() -> TestMetrics:
    """Run comprehensive metrics test."""
    import time
    
    detector = RelevanceDetector()
    dataset = TestDataset()
    
    total_tests = len(dataset.test_cases)
    successful_tests = 0
    false_positives = 0
    false_negatives = 0
    total_confidence = 0.0
    total_time = 0.0
    
    # Category-specific metrics
    category_results = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for text, expected_relevant, category in dataset.test_cases:
        start_time = time.time()
        result = detector.analyze_text(text)
        end_time = time.time()
        
        # Update timing metrics
        total_time += (end_time - start_time)
        
        # Update confidence metrics
        total_confidence += result.confidence_score
        
        # Update accuracy metrics
        if result.is_relevant == expected_relevant:
            successful_tests += 1
            category_results[category]["correct"] += 1
        else:
            if result.is_relevant and not expected_relevant:
                false_positives += 1
            else:
                false_negatives += 1
        
        category_results[category]["total"] += 1
    
    # Calculate overall metrics
    accuracy = successful_tests / total_tests
    precision = successful_tests / (successful_tests + false_positives) if (successful_tests + false_positives) > 0 else 0
    recall = successful_tests / (successful_tests + false_negatives) if (successful_tests + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate category accuracy
    category_accuracy = {
        category: results["correct"] / results["total"]
        for category, results in category_results.items()
    }
    
    return TestMetrics(
        total_tests=total_tests,
        successful_tests=successful_tests,
        false_positives=false_positives,
        false_negatives=false_negatives,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        avg_confidence=total_confidence / total_tests,
        avg_processing_time=total_time / total_tests,
        category_accuracy=category_accuracy
    )

def test_metrics():
    """Run metrics test and assert minimum performance requirements."""
    metrics = run_metrics_test()
    
    # Basic assertions for minimum performance
    assert metrics.accuracy >= 0.80, f"Accuracy {metrics.accuracy:.2%} below minimum threshold of 80%"
    assert metrics.precision >= 0.75, f"Precision {metrics.precision:.2%} below minimum threshold of 75%"
    assert metrics.recall >= 0.75, f"Recall {metrics.recall:.2%} below minimum threshold of 75%"
    assert metrics.f1_score >= 0.75, f"F1 Score {metrics.f1_score:.2%} below minimum threshold of 75%"
    
    # Print detailed report
    print("\n=== Cryptocurrency Relevance Detection Metrics ===")
    print(f"\nOverall Metrics:")
    print(f"Total Tests: {metrics.total_tests}")
    print(f"Successful Tests: {metrics.successful_tests}")
    print(f"False Positives: {metrics.false_positives}")
    print(f"False Negatives: {metrics.false_negatives}")
    print(f"Accuracy: {metrics.accuracy:.2%}")
    print(f"Precision: {metrics.precision:.2%}")
    print(f"Recall: {metrics.recall:.2%}")
    print(f"F1 Score: {metrics.f1_score:.2%}")
    print(f"Average Confidence: {metrics.avg_confidence:.3f}")
    print(f"Average Processing Time: {metrics.avg_processing_time*1000:.2f}ms")
    
    print("\nCategory-specific Accuracy:")
    for category, accuracy in metrics.category_accuracy.items():
        print(f"{category.capitalize()}: {accuracy:.2%}")

if __name__ == "__main__":
    test_metrics() 