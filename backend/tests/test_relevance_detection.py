"""
Comprehensive test suite for the cryptocurrency relevance detection system.
"""
import pytest
from typing import Dict, Any
import logging

from app.services.sentiment.relevance_detection import (
    Term, TermType, normalize_text, check_relevance,
    RelevanceDetector, RelevanceLevel, ContextType
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TestRelevanceDetection:
    """Test suite for relevance detection functionality."""
    
    test_results = {
        'total_tests': 0,
        'passed_tests': 0,
        'categories': {}
    }

    @pytest.fixture
    def detector(self) -> RelevanceDetector:
        """Create a RelevanceDetector instance for testing."""
        return RelevanceDetector()

    @classmethod
    def log_test_result(cls, category: str, passed: bool):
        """Log test result and update metrics."""
        if category not in cls.test_results['categories']:
            cls.test_results['categories'][category] = {'passed': 0, 'total': 0}
        
        cls.test_results['categories'][category]['total'] += 1
        cls.test_results['total_tests'] += 1
        
        if passed:
            cls.test_results['categories'][category]['passed'] += 1
            cls.test_results['passed_tests'] += 1

    def test_empty_text(self, detector: RelevanceDetector):
        """Test handling of empty text."""
        result = detector.analyze_text("")
        passed = (
            result.level == RelevanceLevel.IRRELEVANT and
            result.confidence == 1.0 and
            not result.token_matches and
            not result.supporting_evidence and
            not result.conflicting_evidence
        )
        self.log_test_result('Empty Text', passed)
        
        assert result.level == RelevanceLevel.IRRELEVANT
        assert result.confidence == 1.0
        assert not result.token_matches
        assert not result.supporting_evidence
        assert not result.conflicting_evidence

    def test_high_relevance_crypto_specific(self, detector: RelevanceDetector):
        """Test detection of highly relevant cryptocurrency content."""
        text = "Bitcoin price reached $50,000 today as Ethereum network sees increased DeFi activity"
        result = detector.analyze_text(text)
        
        passed = (
            result.level == RelevanceLevel.HIGH and
            result.confidence >= 0.8 and
            result.context_info.context_type == ContextType.CRYPTO_SPECIFIC and
            any("Bitcoin" in evidence for evidence in result.supporting_evidence) and
            any("Ethereum" in evidence for evidence in result.supporting_evidence) and
            any("DeFi" in evidence for evidence in result.supporting_evidence)
        )
        self.log_test_result('High Relevance', passed)
        
        assert result.level == RelevanceLevel.HIGH
        assert result.confidence >= 0.8
        assert result.context_info.context_type == ContextType.CRYPTO_SPECIFIC
        assert any("Bitcoin" in evidence for evidence in result.supporting_evidence)
        assert any("Ethereum" in evidence for evidence in result.supporting_evidence)
        assert any("DeFi" in evidence for evidence in result.supporting_evidence)

    def test_medium_relevance_mixed_context(self, detector: RelevanceDetector):
        """Test detection of medium relevance with mixed context."""
        text = "The technology behind blockchain could revolutionize supply chain management"
        result = detector.analyze_text(text)
        
        assert result.level == RelevanceLevel.MEDIUM
        assert 0.6 <= result.confidence < 0.8
        assert result.context_info.context_type == ContextType.MIXED
        assert any("blockchain" in evidence.lower() for evidence in result.supporting_evidence)
        assert any("Mixed context" in note for note in result.disambiguation_notes)

    def test_low_relevance_ambiguous(self, detector: RelevanceDetector):
        """Test detection of low relevance with ambiguous context."""
        text = "The token system will be implemented in our database architecture"
        result = detector.analyze_text(text)
        
        assert result.level in [RelevanceLevel.LOW, RelevanceLevel.AMBIGUOUS]
        assert result.confidence < 0.6
        assert any("token" in evidence.lower() for evidence in result.supporting_evidence)
        assert any("database" in evidence.lower() for evidence in result.conflicting_evidence)

    def test_technical_content_boost(self, detector: RelevanceDetector):
        """Test confidence boost for technical cryptocurrency content."""
        text = "Implementing zkSNARKs for Layer 2 scaling solutions with optimistic rollups"
        result = detector.analyze_text(text)
        
        assert result.level == RelevanceLevel.HIGH
        assert result.confidence >= 0.8
        assert result.context_info.context_type == ContextType.TECHNICAL
        assert any("Layer 2" in evidence for evidence in result.supporting_evidence)
        assert any("technical" in evidence.lower() for evidence in result.supporting_evidence)

    def test_price_pattern_detection(self, detector: RelevanceDetector):
        """Test accurate detection of cryptocurrency price patterns."""
        text = "BTC/USD trading at $48,500 with ETH reaching $3,200"
        result = detector.analyze_text(text)
        
        assert result.level == RelevanceLevel.HIGH
        assert result.confidence >= 0.8
        assert any("BTC" in evidence for evidence in result.supporting_evidence)
        assert any("ETH" in evidence for evidence in result.supporting_evidence)
        assert any("price" in evidence.lower() for evidence in result.supporting_evidence)

    def test_defi_term_combinations(self, detector: RelevanceDetector):
        """Test proper handling of DeFi term combinations."""
        text = "Providing liquidity to the yield farming pool with 120% APY"
        result = detector.analyze_text(text)
        
        assert result.level == RelevanceLevel.HIGH
        assert result.confidence >= 0.8
        assert any("yield" in evidence.lower() for evidence in result.supporting_evidence)
        assert any("liquidity" in evidence.lower() for evidence in result.supporting_evidence)
        assert any("APY" in evidence for evidence in result.supporting_evidence)

    def test_mixed_content_handling(self, detector: RelevanceDetector):
        """Test improved handling of mixed content."""
        text = "Comparing blockchain technology with traditional database systems for supply chain"
        result = detector.analyze_text(text)
        
        assert result.level == RelevanceLevel.MEDIUM
        assert result.context_info.context_type == ContextType.MIXED
        assert any("blockchain" in evidence.lower() for evidence in result.supporting_evidence)
        assert any("database" in evidence.lower() for evidence in result.conflicting_evidence)
        assert any("Mixed context" in note for note in result.disambiguation_notes)

    def test_context_disambiguation(self, detector: RelevanceDetector):
        """Test context disambiguation capabilities."""
        text = "Token authentication system for API security using JWT"
        result = detector.analyze_text(text)
        
        assert result.level in [RelevanceLevel.LOW, RelevanceLevel.IRRELEVANT]
        assert result.confidence < 0.6
        assert any("token" in evidence.lower() for evidence in result.supporting_evidence)
        assert any("authentication" in evidence.lower() or "API" in evidence 
                  for evidence in result.conflicting_evidence)

    def test_sentiment_analysis_integration(self, detector: RelevanceDetector):
        """Test integration of sentiment analysis in relevance detection."""
        text = "Bullish on Bitcoin! 🚀 The market is showing strong upward momentum"
        result = detector.analyze_text(text)
        
        assert result.level == RelevanceLevel.HIGH
        assert result.confidence >= 0.8
        assert any("sentiment" in note.lower() for note in result.disambiguation_notes)
        assert any("Bitcoin" in evidence for evidence in result.supporting_evidence)
        assert any("bullish" in evidence.lower() for evidence in result.supporting_evidence)

    def test_multiple_token_boost(self, detector: RelevanceDetector):
        """Test confidence boost for multiple token matches."""
        text = "Trading BTC, ETH, and SOL on major cryptocurrency exchanges"
        result = detector.analyze_text(text)
        
        assert result.level == RelevanceLevel.HIGH
        assert result.confidence >= 0.8
        assert len([e for e in result.supporting_evidence 
                   if any(token in e for token in ["BTC", "ETH", "SOL"])]) >= 3

    def test_context_validation(self, detector: RelevanceDetector):
        """Test context validation requirements."""
        text = "The wallet contains important documents and keys"
        result = detector.analyze_text(text)
        
        assert result.level in [RelevanceLevel.LOW, RelevanceLevel.IRRELEVANT]
        assert result.confidence < 0.6
        assert any("wallet" in evidence.lower() for evidence in result.supporting_evidence)
        assert any("No validation context" in evidence 
                  for evidence in result.conflicting_evidence)

    def test_edge_cases(self, detector: RelevanceDetector):
        """Test handling of edge cases."""
        edge_cases = [
            "Mining data from the database",
            "Staking out the competition",
            "Chain of command in the organization",
            "Token of appreciation",
            "Exchanging business cards"
        ]
        
        for text in edge_cases:
            result = detector.analyze_text(text)
            assert result.level in [RelevanceLevel.LOW, RelevanceLevel.IRRELEVANT]
            assert result.confidence < 0.6
            assert len(result.conflicting_evidence) > 0

    def test_metrics_access(self, detector: RelevanceDetector):
        """Test access to relevance detection metrics."""
        metrics = detector.get_relevance_metrics()
        
        assert isinstance(metrics, dict)
        assert "confidence_thresholds" in metrics
        assert "pattern_types" in metrics
        assert "context_types" in metrics
        assert "relevance_levels" in metrics
        assert all(isinstance(threshold, float) 
                  for threshold in metrics["confidence_thresholds"].values())

    @pytest.mark.parametrize("text,expected_level,min_confidence", [
        ("Bitcoin price analysis shows bullish trend", RelevanceLevel.HIGH, 0.8),
        ("Blockchain technology in supply chain", RelevanceLevel.MEDIUM, 0.6),
        ("Token based authentication system", RelevanceLevel.LOW, 0.4),
        ("Regular business meeting notes", RelevanceLevel.IRRELEVANT, 0.0),
    ])
    def test_parametrized_cases(self, detector: RelevanceDetector, 
                              text: str, expected_level: RelevanceLevel, 
                              min_confidence: float):
        """Test various cases with different expected outcomes."""
        result = detector.analyze_text(text)
        assert result.level == expected_level
        assert result.confidence >= min_confidence 

    @classmethod
    def print_test_summary(cls):
        """Print comprehensive test results summary."""
        logger.info("\n" + "="*50)
        logger.info("RELEVANCE DETECTION TEST SUMMARY")
        logger.info("="*50)
        
        overall_accuracy = (cls.test_results['passed_tests'] / cls.test_results['total_tests']) * 100
        logger.info(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
        logger.info(f"Total Tests: {cls.test_results['total_tests']}")
        logger.info(f"Passed Tests: {cls.test_results['passed_tests']}")
        
        logger.info("\nCategory Breakdown:")
        for category, results in cls.test_results['categories'].items():
            accuracy = (results['passed'] / results['total']) * 100
            logger.info(f"{category}:")
            logger.info(f"  Accuracy: {accuracy:.2f}%")
            logger.info(f"  Passed: {results['passed']}/{results['total']}")
        
        logger.info("="*50)

    def pytest_sessionfinish(self, session):
        """Print test summary after all tests complete."""
        self.print_test_summary() 