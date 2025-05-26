"""
Comprehensive test suite for cryptocurrency relevance detection.
"""
import pytest
from app.services.sentiment.relevance_detection import (
    Term, TermType, normalize_text, check_relevance,
    RelevanceDetector, RelevanceLevel, TokenType
)

@pytest.fixture
def detector():
    """Create a RelevanceDetector instance for testing."""
    return RelevanceDetector(
        confidence_threshold=0.6,
        context_threshold=0.5,
        min_token_matches=1
    )

class TestBasicTerms:
    """Test basic cryptocurrency term detection."""
    
    def test_basic_crypto_terms(self, detector):
        """Test detection of basic cryptocurrency terms."""
        text = "Bitcoin and Ethereum are leading cryptocurrencies"
        result = detector.analyze_text(text)
        assert result.is_relevant
        assert result.confidence_score > 0.8
        assert len(result.token_matches) >= 3
        assert any(m["type"] == TokenType.CRYPTOCURRENCY.value for m in result.token_matches)

    def test_common_abbreviations(self, detector):
        """Test detection of common cryptocurrency abbreviations."""
        text = "BTC/USD and ETH/USD trading pairs"
        result = detector.analyze_text(text)
        assert result.is_relevant
        assert result.confidence_score > 0.7
        assert any(m["token"].lower() in ["btc", "eth"] for m in result.token_matches)

    def test_negative_case(self, detector):
        """Test non-cryptocurrency content."""
        text = "The weather is nice today"
        result = detector.analyze_text(text)
        assert not result.is_relevant
        assert result.confidence_score < 0.3
        assert len(result.token_matches) == 0

class TestTradingPairs:
    """Test trading pair detection."""
    
    def test_standard_pairs(self, detector):
        """Test standard trading pair formats."""
        text = "BTC/USDT trading at 50,000"
        result = detector.analyze_text(text)
        assert result.is_relevant
        assert result.confidence_score > 0.7
        assert any("trading_pair" in m["type"] for m in result.token_matches)

    def test_price_formatting(self, detector):
        """Test price format detection."""
        text = "ETH price: $2,500.50"
        result = detector.analyze_text(text)
        assert result.is_relevant
        assert result.confidence_score > 0.7

    def test_multiple_pairs(self, detector):
        """Test multiple trading pair detection."""
        text = "Trading BTC/USDT, ETH/USDT, and SOL/USD pairs"
        result = detector.analyze_text(text)
        assert result.is_relevant
        assert result.confidence_score > 0.8
        assert len([m for m in result.token_matches if "trading_pair" in m["type"]]) >= 2

class TestTechnicalTerms:
    """Test technical cryptocurrency term detection."""
    
    def test_blockchain_terms(self, detector):
        """Test blockchain-related terms."""
        text = "The blockchain network uses smart contracts for consensus"
        result = detector.analyze_text(text)
        assert result.is_relevant
        assert result.is_technical
        assert result.confidence_score > 0.7

    def test_defi_terms(self, detector):
        """Test DeFi-related terms."""
        text = "Yield farming in liquidity pools with high APY"
        result = detector.analyze_text(text)
        assert result.is_relevant
        assert any(m["type"] == TokenType.DEFI_TERM.value for m in result.token_matches)

    def test_technical_disambiguation(self, detector):
        """Test technical term disambiguation."""
        text = "The database uses token-based authentication"
        result = detector.analyze_text(text)
        assert not result.is_relevant
        assert result.confidence_score < 0.5

class TestContextSensitivity:
    """Test context-sensitive detection."""
    
    def test_positive_context(self, detector):
        """Test positive cryptocurrency context."""
        text = "The market analysis shows bullish trends for BTC"
        result = detector.analyze_text(text)
        assert result.is_relevant
        assert result.context_analysis["sentiment_analysis"]["score"] > 0

    def test_negative_context(self, detector):
        """Test negative cryptocurrency context."""
        text = "The stock market and bond trading session"
        result = detector.analyze_text(text)
        assert not result.is_relevant
        assert len(result.context_analysis.get("negative_contexts", [])) > 0

    def test_ambiguous_context(self, detector):
        """Test ambiguous context handling."""
        text = "Trading volume increased significantly"
        result = detector.analyze_text(text)
        assert not result.is_relevant  # Too ambiguous without specific crypto terms

class TestConfidenceScores:
    """Test confidence score calculation."""
    
    def test_high_confidence(self, detector):
        """Test high confidence scenarios."""
        text = "Bitcoin and Ethereum prices surge in cryptocurrency markets"
        result = detector.analyze_text(text)
        assert result.confidence_score > 0.8
        assert result.is_relevant

    def test_medium_confidence(self, detector):
        """Test medium confidence scenarios."""
        text = "Digital currency adoption is growing"
        result = detector.analyze_text(text)
        assert 0.4 < result.confidence_score < 0.8

    def test_low_confidence(self, detector):
        """Test low confidence scenarios."""
        text = "The digital economy is changing"
        result = detector.analyze_text(text)
        assert result.confidence_score < 0.4
        assert not result.is_relevant

class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_empty_text(self, detector):
        """Test empty text handling."""
        result = detector.analyze_text("")
        assert not result.is_relevant
        assert result.confidence_score == 0.0

    def test_mixed_content(self, detector):
        """Test mixed cryptocurrency and non-cryptocurrency content."""
        text = "Bitcoin trading while playing video games and checking stocks"
        result = detector.analyze_text(text)
        assert result.is_relevant  # Should still detect crypto content
        assert 0.5 < result.confidence_score < 0.9  # But with reduced confidence

    def test_case_sensitivity(self, detector):
        """Test case sensitivity handling."""
        text_lower = "bitcoin and ethereum"
        text_upper = "BITCOIN and ETHEREUM"
        text_mixed = "BiTcOiN and EtHeReUm"
        
        result_lower = detector.analyze_text(text_lower)
        result_upper = detector.analyze_text(text_upper)
        result_mixed = detector.analyze_text(text_mixed)
        
        assert result_lower.is_relevant
        assert result_upper.is_relevant
        assert result_mixed.is_relevant
        assert abs(result_lower.confidence_score - result_upper.confidence_score) < 0.1
        assert abs(result_lower.confidence_score - result_mixed.confidence_score) < 0.1

class TestExplanationGeneration:
    """Test explanation generation."""
    
    def test_relevant_explanation(self, detector):
        """Test explanation for relevant content."""
        text = "Bitcoin price analysis shows bullish trends"
        result = detector.analyze_text(text)
        assert "relevant" in result.explanation.lower()
        assert "confidence" in result.explanation.lower()
        assert str(round(result.confidence_score, 2)) in result.explanation

    def test_non_relevant_explanation(self, detector):
        """Test explanation for non-relevant content."""
        text = "The weather forecast shows rain"
        result = detector.analyze_text(text)
        assert "not" in result.explanation.lower() or "no" in result.explanation.lower()
        assert len(result.explanation) > 0

    def test_context_explanation(self, detector):
        """Test context information in explanation."""
        text = "Trading cryptocurrencies in the bull market"
        result = detector.analyze_text(text)
        assert any(word in result.explanation.lower() 
                  for word in ["context", "trading", "market"])

class TestMixedContent:
    """Test handling of mixed content types."""
    
    def test_crypto_with_stocks(self, detector):
        """Test mixed cryptocurrency and stock market content."""
        text = "Bitcoin and Tesla stock prices are both up today"
        result = detector.analyze_text(text)
        assert result.is_relevant  # Should detect crypto despite stock mention
        assert result.confidence_score < 0.9  # But with reduced confidence

    def test_crypto_with_gaming(self, detector):
        """Test mixed cryptocurrency and gaming content."""
        text = "Mining Bitcoin while playing Minecraft"
        result = detector.analyze_text(text)
        assert result.is_relevant
        assert result.confidence_score > 0.6

    def test_ambiguous_terms(self, detector):
        """Test handling of ambiguous terms."""
        text = "The token system in the game uses blockchain technology"
        result = detector.analyze_text(text)
        assert not result.is_relevant  # Gaming context should reduce relevance 