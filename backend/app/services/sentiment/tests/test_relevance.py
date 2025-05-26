"""
Test module for relevance detection functionality.
"""

from app.services.sentiment.relevance_detection import check_relevance, get_relevance_explanation
from app.services.sentiment.blocked_content_detector import BlockedContentDetector
import logging
from typing import List, Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

test_metrics = {
    'total_tests': 0,
    'passed_tests': 0,
    'categories': {}
}

def print_analysis(text: str, category: str, expected_relevant: bool):
    """Helper function to print relevance analysis results."""
    is_relevant, relevance_score, matched_terms = check_relevance(text)
    explanation = get_relevance_explanation(matched_terms)
    
    # Update metrics
    if category not in test_metrics['categories']:
        test_metrics['categories'][category] = {'passed': 0, 'total': 0}
    
    test_metrics['total_tests'] += 1
    test_metrics['categories'][category]['total'] += 1
    
    if is_relevant == expected_relevant:
        test_metrics['passed_tests'] += 1
        test_metrics['categories'][category]['passed'] += 1
    
    logger.info("\n" + "="*50)
    logger.info(f"Category: {category}")
    logger.info(f"Text: '{text}'")
    logger.info(f"Is Relevant: {is_relevant}")
    logger.info(f"Expected Relevant: {expected_relevant}")
    logger.info(f"Relevance Score: {relevance_score:.2f}")
    logger.info(f"Explanation: {explanation}")
    logger.info("="*50)

def run_test_cases(category: str, test_cases: List[Tuple[str, bool]]):
    """Run a set of test cases and track results."""
    logger.info(f"\nTesting {category}:")
    
    for text, expected_relevant in test_cases:
        print_analysis(text, category, expected_relevant)

def print_test_summary():
    """Print comprehensive test results summary."""
    logger.info("\n" + "="*50)
    logger.info("RELEVANCE DETECTION TEST SUMMARY")
    logger.info("="*50)
    
    overall_accuracy = (test_metrics['passed_tests'] / test_metrics['total_tests']) * 100
    logger.info(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
    logger.info(f"Total Tests: {test_metrics['total_tests']}")
    logger.info(f"Passed Tests: {test_metrics['passed_tests']}")
    
    logger.info("\nCategory Breakdown:")
    for category, results in test_metrics['categories'].items():
        accuracy = (results['passed'] / results['total']) * 100
        logger.info(f"{category}:")
        logger.info(f"  Accuracy: {accuracy:.2f}%")
        logger.info(f"  Passed: {results['passed']}/{results['total']}")
    
    logger.info("="*50)

def test_relevance_detection():
    """Test relevance detection with comprehensive test cases."""
    
    # Category 1: Completely irrelevant text
    irrelevant_cases = [
        ("I just ate a sandwich", False),
        ("The weather is nice today", False),
        ("My favorite color is blue", False),
        ("She went to the store yesterday", False),
        ("The cat jumped over the fence", False),
        ("Today's football match was exciting", False),
        ("The new restaurant downtown is amazing", False),
        ("My car needs an oil change", False),
        ("The movie was really entertaining", False),
        ("Planning a vacation next month", False)
    ]
    
    # Category 2: Ambiguous or partially relevant text
    ambiguous_cases = [
        ("The digital market is evolving", False),
        ("The price of technology stocks is falling", False),
        ("Global payment systems are changing", False),
        ("Mining operations in Australia", False),
        ("The chain of events led to success", False),
        ("A new digital transformation initiative", False),
        ("The value of the dollar is dropping", False),
        ("Trading hours for the stock market", False),
        ("Digital wallet for mobile payments", False),
        ("The network is experiencing issues", False),
        ("Secure transactions in online banking", False),
        ("Mining gold in South Africa", False),
        ("The exchange rate is fluctuating", False),
        ("A token of appreciation", False),
        ("The hash browns were delicious", False)
    ]
    
    # Category 3: Clearly crypto-related text
    crypto_cases = [
        ("Bitcoin price reaches new ATH", True),
        ("ETH gas fees are high today", True),
        ("DeFi protocols show strong growth", True),
        ("NFT marketplace OpenSea launches new features", True),
        ("Cryptocurrency adoption grows globally", True),
        ("Blockchain technology revolutionizes finance", True),
        ("BTC/USD trading pair hits resistance", True),
        ("Ethereum 2.0 staking rewards increase", True),
        ("Binance announces new token listing", True),
        ("Cardano smart contracts go live", True),
        ("USDT market cap reaches new milestone", True),
        ("Solana network throughput peaks", True),
        ("DeFi yield farming strategies", True),
        ("NFT gaming tokens surge in value", True),
        ("Bitcoin mining difficulty adjusts", True)
    ]
    
    # Category 4: Edge cases and compound terms
    edge_cases = [
        ("", False),
        ("BITCOIN", True),
        ("bit-coin", True),
        ("The blockchain technology platform", True),
        ("My friend mines gold", False),
        ("This is a chain of events", False),
        ("Digital currency transformation", False),
        ("Crypto wallet security", True),
        ("BTC-USD trading pair", True),
        ("ETH/BTC", True),
        ("btc.price", True),
        ("crypto_market", True),
        ("blockchain-based", True),
        ("non-fungible-token", True),
        ("defi.protocol", True),
        ("Web3.0", True),
        ("meta-verse token", True),
        ("proof-of-stake", True),
        ("gas.fees", True),
        ("smart.contract", True)
    ]
    
    # Category 5: Technical and market terms
    technical_cases = [
        ("Layer 2 scaling solutions improve efficiency", True),
        ("Smart contract deployment on Ethereum", True),
        ("Staking rewards distribution", True),
        ("Hash rate reaches record levels", True),
        ("Liquidity pool optimization", True),
        ("The validator network expands", True),
        ("Zero-knowledge proof implementation", True),
        ("Cross-chain bridge security", True),
        ("MEV protection in DeFi protocols", True),
        ("Consensus mechanism upgrade", True),
        ("Multi-sig wallet configuration", True),
        ("Oracle price feed integration", True),
        ("Gas optimization techniques", True),
        ("Proof of work vs proof of stake", True),
        ("Sharding implementation details", True)
    ]
    
    # Category 6: Context-dependent terms
    context_cases = [
        ("The price of bread is rising", False),
        ("Bitcoin price analysis", True),
        ("Gas prices at the pump", False),
        ("ETH gas optimization", True),
        ("A chain of stores", False),
        ("Blockchain network security", True),
        ("Mining equipment for coal", False),
        ("Crypto mining farm setup", True),
        ("Token economy in education", False),
        ("ERC20 token standard", True),
        ("Network congestion in traffic", False),
        ("Network fees on Ethereum", True),
        ("Exchange of foreign currency", False),
        ("DEX exchange volumes", True),
        ("Wallet lost in the mall", False)
    ]
    
    # Category 7: Real-world news headlines
    news_cases = [
        ("SEC Approves Bitcoin ETF Applications", True),
        ("Major Bank Integrates Crypto Trading", True),
        ("New Cryptocurrency Tax Regulations", True),
        ("DeFi Protocol Suffers Security Breach", True),
        ("NFT Sales Volume Hits Monthly High", True),
        ("Central Bank Digital Currency Plans", True),
        ("Crypto Mining's Environmental Impact", True),
        ("Web3 Gaming Platform Launch", True),
        ("Metaverse Token Market Analysis", True),
        ("Institutional Bitcoin Adoption Trends", True)
    ]
    
    # Category 8: Complex mixed content
    mixed_cases = [
        ("Bitcoin mining affects GPU prices in gaming market", True),
        ("Digital art NFTs transform traditional galleries", True),
        ("Blockchain voting systems face security challenges", True),
        ("Cryptocurrency regulations impact global markets", True),
        ("DeFi protocols integrate with traditional finance", True),
        ("Smart contract insurance for crypto assets", True),
        ("Cross-chain bridges enable token transfers", True),
        ("Layer 2 scaling reduces gas fees on Ethereum", True),
        ("Web3 social platforms use token incentives", True),
        ("Metaverse land prices in virtual worlds", True)
    ]
    
    # Run all test categories
    categories = [
        ("Irrelevant Text", irrelevant_cases),
        ("Ambiguous Text", ambiguous_cases),
        ("Crypto-Related Text", crypto_cases),
        ("Edge Cases", edge_cases),
        ("Technical Terms", technical_cases),
        ("Context-Dependent Terms", context_cases),
        ("News Headlines", news_cases),
        ("Mixed Content", mixed_cases)
    ]
    
    for category_name, test_cases in categories:
        run_test_cases(category_name, test_cases)
    
    print_test_summary()

def test_gibberish():
    """Test how the system handles gibberish text."""
    logger.info("\nTesting Gibberish Text Detection:")
    
    test_cases = [
        "afjusahjfaof",
        "123456789",
        "qwertyuiop",
        "!@#$%^&*()",
        "btcafjusahjfaof",  # Gibberish with a crypto term
        "afjusahjfaofeth",  # Gibberish with a crypto term
        "afjusahjfaof bitcoin afjusahjfaof",  # Gibberish surrounding a crypto term
        "afjbtcahjfaof",  # Crypto term embedded in gibberish
        "AFJUSAHJFAOF",  # Uppercase gibberish
        "afju sahj faof"  # Spaced gibberish
    ]
    
    for text in test_cases:
        print_analysis(text, "Gibberish Text", False)

def test_edge_cases():
    """Test edge cases and potential manipulation attempts."""
    test_cases = [
        # Manipulation attempts
        ("BITCOIN!!!!! TO THE MOON!!!!!!", False),  # Excessive punctuation
        ("btc btc btc btc btc btc btc", False),    # Repetitive spam
        ("1BTC = 100k$$$$$$", False),              # Symbol spam
        
        # Valid edge cases
        ("eth2.0+defi+nft+web3", True),           # Valid compressed format
        ("$btc $eth $sol analysis", True),        # Valid ticker format
        ("BTC/USD trading pair", True),           # Valid trading pair format
        ("btc-usdt eth-usdt", True),              # Valid trading pairs
        ("100xLeverage#Bitcoin", True),           # Valid trading term
        ("⚡️Lightning Network⚡️", True),         # Valid with emojis
        
        # Mixed cases
        ("!!!BITCOIN!!! price analysis", True),    # Valid despite some manipulation
        ("ETH 2.0 upgrade!!!!!", True),           # Valid despite excitement
        ("btc/eth pair 🚀🌙", True),              # Valid with emojis
        ("$btc $eth $doge 🚀", True),             # Valid with tickers and emojis
        
        # Special formats
        ("BTC-USD", True),                        # Trading pair format
        ("ETH/USDT", True),                       # Alternative pair format
        ("$BTC.D", True),                         # Technical indicator
        ("BTC.dominance", True),                  # Market metric
        
        # Technical edge cases
        ("zk-rollup implementation", True),       # Technical term with hyphen
        ("L2/scaling-solution", True),            # Technical with special chars
        ("PoS>PoW efficiency", True),             # Comparison format
        ("v2.5-upgrade", True),                   # Version number
        
        # Context-dependent edge cases
        ("high gas today", True),                 # Context makes it relevant
        ("chain is slow", True),                  # Context makes it relevant
        ("bridge is down", True),                 # Context makes it relevant
        
        # Borderline cases
        ("100x gains!!!", False),                 # Not enough context
        ("moon soon 🚀", False),                  # Not enough context
        ("going up up up!!!", False),             # Not enough context
        ("best investment!!!", False),            # Not enough context
    ]
    
    logger.info("\nTesting Edge Cases and Manipulation Detection:")
    for text, expected_relevant in test_cases:
        print_analysis(text, "Edge Cases", expected_relevant)

if __name__ == "__main__":
    test_relevance_detection()
    test_gibberish()
    test_edge_cases() 