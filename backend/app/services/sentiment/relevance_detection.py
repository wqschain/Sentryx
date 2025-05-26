"""
Module for detecting cryptocurrency-related content relevance.
"""

from typing import Tuple, Set, Dict, List, Any
import re
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RelevanceLevel(Enum):
    """Levels of content relevance."""
    IRRELEVANT = "irrelevant"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AMBIGUOUS = "ambiguous"  # Added back the AMBIGUOUS level

class ContextType(Enum):
    """Types of content context."""
    CRYPTO_SPECIFIC = "crypto_specific"
    TECHNICAL = "technical"
    MIXED = "mixed"
    IRRELEVANT = "irrelevant"

@dataclass
class ContextInfo:
    """Information about the context of analyzed content."""
    context_type: ContextType
    primary_terms: List[str]
    secondary_terms: List[str]
    context_terms: List[str]
    confidence: float

@dataclass
class RelevanceScore:
    """Score and details for content relevance analysis."""
    level: RelevanceLevel
    confidence: float
    supporting_evidence: Dict[str, Any]
    explanation: str
    context_info: ContextInfo = None
    token_matches: List[str] = None
    conflicting_evidence: List[str] = None
    disambiguation_notes: List[str] = None

class TermType(Enum):
    """Types of terms used in relevance detection."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TECHNICAL = "technical"
    WEB3 = "web3"
    CONTEXT_DEPENDENT = "context_dependent"
    CONTEXT = "context"

@dataclass
class Term:
    """Definition of a term used in relevance detection."""
    text: str
    type: TermType
    weight: float
    requires_context: bool = False
    aliases: List[str] = None
    compound_pairs: List[str] = None
    min_confidence: float = 0.3    # Minimum confidence threshold for term relevance
    max_boost: float = 2.0         # Maximum boost from context and compound terms
    decay_factor: float = 0.15     # Weight decay with distance in context window
    boost_threshold: float = 0.5   # Threshold for applying context boosts

    def __post_init__(self):
        """Initialize and validate term attributes."""
        self.aliases = self.aliases or []
        self.compound_pairs = self.compound_pairs or []
        
        # Validate weights and factors
        self.weight = max(0.0, min(1.0, self.weight))
        self.min_confidence = max(0.0, min(1.0, self.min_confidence))
        self.max_boost = max(1.0, self.max_boost)
        self.decay_factor = max(0.0, min(1.0, self.decay_factor))
        
    def calculate_match_weight(self, context_score: float = 0.0, compound_match: bool = False) -> float:
        """Calculate the final weight for a term match considering context and compounds."""
        base_weight = self.weight
        
        # Apply context boost if above threshold
        if context_score > self.boost_threshold:
            context_boost = min(context_score * (1 - self.decay_factor), self.max_boost - 1)
            base_weight *= (1 + context_boost)
        
        # Apply compound term boost
        if compound_match:
            base_weight *= 1.2  # 20% boost for compound terms
        
        # Ensure weight doesn't exceed max_boost
        return min(base_weight, self.weight * self.max_boost)

# Comprehensive term definitions with expanded coverage
TERMS = [
    # Primary Cryptocurrencies
    Term("bitcoin", TermType.PRIMARY, 0.6, aliases=["btc", "xbt"], compound_pairs=["price", "market", "trading", "wallet", "mining"]),
    Term("ethereum", TermType.PRIMARY, 0.6, aliases=["eth", "ether"], compound_pairs=["gas", "smart contract", "defi", "nft"]),
    Term("solana", TermType.PRIMARY, 0.5, aliases=["sol"]),
    Term("cardano", TermType.PRIMARY, 0.5, aliases=["ada"]),
    Term("ripple", TermType.PRIMARY, 0.5, aliases=["xrp"]),
    Term("dogecoin", TermType.PRIMARY, 0.5, aliases=["doge"]),
    Term("binance", TermType.PRIMARY, 0.5, aliases=["bnb", "bsc"]),
    
    # Stablecoins
    Term("tether", TermType.PRIMARY, 0.5, aliases=["usdt"]),
    Term("usdc", TermType.PRIMARY, 0.5, aliases=["usd coin"]),
    Term("dai", TermType.PRIMARY, 0.5),
    
    # Primary Crypto Concepts
    Term("cryptocurrency", TermType.PRIMARY, 0.9, aliases=["crypto", "coin", "token"], compound_pairs=["market", "exchange", "trading", "wallet"]),
    Term("blockchain", TermType.TECHNICAL, 0.8, aliases=["distributed ledger", "dlt"], compound_pairs=["technology", "platform", "network", "protocol"]),
    Term("defi", TermType.WEB3, 0.8, aliases=["decentralized finance"], compound_pairs=["protocol", "yield", "farming", "liquidity"]),
    Term("nft", TermType.WEB3, 0.7, aliases=["non-fungible token"], compound_pairs=["collection", "marketplace", "mint", "floor price"]),
    Term("dao", TermType.PRIMARY, 0.5, aliases=["decentralized autonomous organization"]),
    Term("dex", TermType.PRIMARY, 0.5, aliases=["decentralized exchange", "dexes"]),
    
    # Technical Terms
    Term("smart contract", TermType.TECHNICAL, 0.8, compound_pairs=["deploy", "audit", "function", "solidity"]),
    Term("hash rate", TermType.TECHNICAL, 0.5, aliases=["hashrate", "hashing power"]),
    Term("validator", TermType.TECHNICAL, 0.5, aliases=["validators", "validation"], compound_pairs=["node", "stake", "pool"]),
    Term("staking", TermType.TECHNICAL, 0.5, aliases=["staked", "staker"], compound_pairs=["pool", "reward", "validator"]),
    Term("liquidity pool", TermType.TECHNICAL, 0.5, aliases=["lp", "amm pool"], compound_pairs=["provider", "token", "pair"]),
    Term("layer 2", TermType.TECHNICAL, 0.5, aliases=["l2", "layer two", "rollup"], compound_pairs=["scaling", "solution", "network"]),
    Term("zero knowledge", TermType.TECHNICAL, 0.5, aliases=["zk", "zkp"], compound_pairs=["proof", "rollup", "scaling"]),
    Term("proof of stake", TermType.TECHNICAL, 0.5, aliases=["pos"], compound_pairs=["consensus", "mechanism"]),
    Term("proof of work", TermType.TECHNICAL, 0.5, aliases=["pow"], compound_pairs=["mining", "consensus"]),
    Term("consensus", TermType.TECHNICAL, 0.4, compound_pairs=["mechanism", "algorithm"]),
    Term("sharding", TermType.TECHNICAL, 0.4, compound_pairs=["scaling", "implementation"]),
    Term("multi sig", TermType.TECHNICAL, 0.4, aliases=["multisig"], compound_pairs=["wallet", "transaction"]),
    Term("oracle", TermType.TECHNICAL, 0.4, compound_pairs=["price feed", "network"]),
    Term("erc20", TermType.TECHNICAL, 0.5, aliases=["erc 20"], compound_pairs=["token", "standard", "contract"]),
    Term("erc721", TermType.TECHNICAL, 0.5, aliases=["erc 721"], compound_pairs=["token", "standard", "nft"]),
    
    # Web3 Terms
    Term("web3", TermType.WEB3, 0.4, aliases=["web 3", "web 3.0"], compound_pairs=["platform", "application", "development"]),
    Term("metaverse", TermType.WEB3, 0.4, compound_pairs=["token", "platform", "land", "virtual"]),
    Term("gamefi", TermType.WEB3, 0.4, aliases=["play to earn", "p2e"], compound_pairs=["token", "gaming", "reward"]),
    Term("depin", TermType.WEB3, 0.4, aliases=["decentralized physical infrastructure"]),
    Term("soulbound", TermType.WEB3, 0.4, compound_pairs=["token", "nft"]),
    
    # Secondary Terms (Need Context)
    Term("token", TermType.SECONDARY, 0.3, requires_context=True, compound_pairs=["crypto", "digital", "blockchain", "standard", "holder"]),
    Term("coin", TermType.SECONDARY, 0.3, requires_context=True, compound_pairs=["crypto", "digital", "blockchain", "alt"]),
    Term("mining", TermType.TECHNICAL, 0.7, requires_context=True, compound_pairs=["pool", "rig", "hashrate", "difficulty"]),
    Term("wallet", TermType.SECONDARY, 0.6, requires_context=True, compound_pairs=["address", "private key", "seed phrase"]),
    Term("exchange", TermType.SECONDARY, 0.6, requires_context=True, compound_pairs=["rate", "trading", "volume", "order"]),
    
    # Context Dependent Terms
    Term("price", TermType.CONTEXT_DEPENDENT, 0.2, requires_context=True, compound_pairs=["bitcoin", "eth", "crypto", "token"]),
    Term("gas", TermType.CONTEXT_DEPENDENT, 0.2, requires_context=True, compound_pairs=["eth", "fee", "gwei"]),
    Term("chain", TermType.CONTEXT_DEPENDENT, 0.2, requires_context=True, compound_pairs=["block", "side", "cross"]),
    Term("bridge", TermType.CONTEXT_DEPENDENT, 0.2, requires_context=True, compound_pairs=["cross chain", "token", "asset"]),
    Term("protocol", TermType.CONTEXT_DEPENDENT, 0.2, requires_context=True, compound_pairs=["defi", "blockchain", "lending"]),
    Term("network", TermType.CONTEXT_DEPENDENT, 0.2, requires_context=True, compound_pairs=["blockchain", "crypto", "node"]),
    
    # Context Terms
    Term("digital currency", TermType.CONTEXT, 0.2),
    Term("decentralized", TermType.CONTEXT, 0.2),
    Term("distributed ledger", TermType.CONTEXT, 0.2),
    Term("peer to peer", TermType.CONTEXT, 0.2, aliases=["p2p"]),
    Term("virtual currency", TermType.CONTEXT, 0.2)
]

def normalize_text(text: str) -> str:
    """Normalize text for consistent matching."""
    # Convert to lowercase
    text = text.lower()
    
    # Replace common variations
    replacements = {
        "btc": "bitcoin",
        "eth": "ethereum",
        "crypto": "cryptocurrency",
        "defi": "decentralized finance",
        "nft": "non-fungible token"
    }
    
    for old, new in replacements.items():
        text = re.sub(rf'\b{old}\b', new, text)
    
    # Remove special characters except those in compound terms
    text = re.sub(r'[^\w\s-]', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text

def find_term_matches(text: str, term: Term) -> List[str]:
    """Find all matches for a term and its aliases in the text."""
    matches = []
    normalized_text = normalize_text(text.lower())
    
    # Check main term
    if re.search(rf'\b{re.escape(term.text.lower())}\b', normalized_text):
        matches.append(term.text)
    
    # Check aliases
    for alias in term.aliases or []:
        if re.search(rf'\b{re.escape(alias.lower())}\b', normalized_text):
            matches.append(alias)
    
    # Check compound pairs
    for pair in term.compound_pairs or []:
        compound = f"{term.text} {pair}"
        if re.search(rf'\b{re.escape(compound.lower())}\b', normalized_text):
            matches.append(compound)
    
    return matches

def check_relevance(text: str) -> Tuple[bool, float, Dict[str, Any]]:
    """Check if text is relevant to cryptocurrency."""
    detector = RelevanceDetector()
    result = detector.analyze_text(text)
    
    is_relevant = result.level != RelevanceLevel.IRRELEVANT
    confidence = result.confidence
    details = result.supporting_evidence
    
    return is_relevant, confidence, details

def get_relevance_explanation(matches: Dict[str, List[str]]) -> str:
    """Generate a human-readable explanation of relevance analysis."""
    if not matches:
        return "No cryptocurrency-related terms found in the text."
    
    explanations = []
    
    if matches.get(TermType.PRIMARY.value):
        explanations.append(
            f"Found primary crypto terms: {', '.join(matches[TermType.PRIMARY.value])}"
        )
    
    if matches.get(TermType.TECHNICAL.value):
        explanations.append(
            f"Found technical terms: {', '.join(matches[TermType.TECHNICAL.value])}"
        )
    
    if matches.get(TermType.SECONDARY.value):
        explanations.append(
            f"Found supporting terms: {', '.join(matches[TermType.SECONDARY.value])}"
        )
    
    if matches.get(TermType.WEB3.value):
        explanations.append(
            f"Found Web3 terms: {', '.join(matches[TermType.WEB3.value])}"
        )
    
    return " ".join(explanations)

def analyze_context_window(text: str, match_info: dict, window_size: int = 5) -> float:
    """Analyze the context window around term matches."""
    words = text.lower().split()
    context_score = 0.0
    total_windows = 0
    
    for term_type, matches in match_info.items():
        for match in matches:
            try:
                # Find all occurrences of the match
                match_words = match.lower().split()
                for i in range(len(words) - len(match_words) + 1):
                    if words[i:i+len(match_words)] == match_words:
                        # Analyze window before match
                        start = max(0, i - window_size)
                        window_before = words[start:i]
                        
                        # Analyze window after match
                        end = min(len(words), i + len(match_words) + window_size)
                        window_after = words[i+len(match_words):end]
                        
                        # Calculate context score for this window
                        window_score = _calculate_window_score(window_before + window_after, match_info)
                        context_score += window_score
                        total_windows += 1
            except Exception as e:
                logger.error(f"Error analyzing context window for {match}: {e}")
                continue
    
    return context_score / max(1, total_windows)

def _calculate_window_score(window_words: List[str], match_info: dict) -> float:
    """Calculate context relevance score for a window of words."""
    score = 0.0
    relevant_terms = 0
    
    # Count relevant terms in window
    for term_type, matches in match_info.items():
        for match in matches:
            match_words = match.lower().split()
            if any(word in window_words for word in match_words):
                if term_type in [TermType.PRIMARY.value, TermType.TECHNICAL.value]:
                    score += 0.5
                elif term_type == TermType.SECONDARY.value:
                    score += 0.3
                elif term_type == TermType.CONTEXT.value:
                    score += 0.2
                relevant_terms += 1
    
    # Normalize score based on window size
    if relevant_terms > 0:
        score = score / (1 + 0.1 * len(window_words))  # Slight penalty for larger windows
    
    return min(1.0, score)

def detect_manipulation(text: str) -> Tuple[float, Dict[str, float]]:
    """Detect potential manipulation attempts in the text."""
    details = {
        'unique_ratio': 0.0,
        'repetition': 0,
        'punctuation': 0.0,
        'symbols': 0.0,
        'caps': 0,
        'numbers': 0,
        'urls': 0.0,
        'tags': 0
    }
    
    if not text:
        return 0.0, details
    
    # Calculate unique word ratio
    words = text.lower().split()
    if words:
        unique_ratio = len(set(words)) / len(words)
        details['unique_ratio'] = 1 - unique_ratio  # Higher score means more repetition
    
    # Check for excessive repetition
    for word in set(words):
        count = words.count(word)
        if count > 3:
            details['repetition'] += (count - 3) * 0.1
    
    # Check punctuation and symbols
    total_chars = len(text)
    if total_chars > 0:
        punct_count = len(re.findall(r'[!?]{2,}', text))
        details['punctuation'] = min(1.0, punct_count * 0.2)
        
        symbol_count = len(re.findall(r'[$@#%&*]+', text))
        details['symbols'] = min(1.0, symbol_count / total_chars)
    
    # Check for excessive caps
    caps_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
    if caps_ratio > 0.5:
        details['caps'] = min(1.0, (caps_ratio - 0.5) * 2)
    
    # Check for numeric spam
    number_count = len(re.findall(r'\d+', text))
    if number_count > 3:
        details['numbers'] = min(1.0, (number_count - 3) * 0.2)
    
    # Check for URLs and tags
    urls = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
    details['urls'] = min(1.0, urls * 0.3)
    
    tags = len(re.findall(r'#\w+', text))
    details['tags'] = min(1.0, tags * 0.2)
    
    # Calculate overall manipulation score
    weights = {
        'unique_ratio': 0.2,
        'repetition': 0.2,
        'punctuation': 0.15,
        'symbols': 0.15,
        'caps': 0.1,
        'numbers': 0.1,
        'urls': 0.05,
        'tags': 0.05
    }
    
    manipulation_score = sum(score * weights[metric] for metric, score in details.items())
    
    # Add confidence level to details
    if manipulation_score < 0.2:
        details['confidence'] = "Very Low"
    elif manipulation_score < 0.4:
        details['confidence'] = "Low"
    elif manipulation_score < 0.6:
        details['confidence'] = "Medium"
    elif manipulation_score < 0.8:
        details['confidence'] = "High"
    else:
        details['confidence'] = "Very High"
    
    return manipulation_score, details

class RelevanceDetector:
    """Detector for cryptocurrency-related content."""
    
    def __init__(self):
        """Initialize the detector with configuration."""
        self.relevance_metrics = {
            'confidence_thresholds': {
                RelevanceLevel.HIGH: 0.8,
                RelevanceLevel.MEDIUM: 0.6,
                RelevanceLevel.LOW: 0.4,
                RelevanceLevel.IRRELEVANT: 0.0
            },
            'pattern_types': [t.value for t in TermType],
            'context_types': [t.value for t in ContextType],
            'relevance_levels': [l.value for l in RelevanceLevel]
        }

    def analyze_text(self, text: str) -> RelevanceScore:
        """Analyze text for cryptocurrency relevance."""
        if not text:
            return RelevanceScore(
                level=RelevanceLevel.IRRELEVANT,
                confidence=1.0,
                supporting_evidence=[],
                explanation="Empty text",
                token_matches=[],
                conflicting_evidence=[],
                disambiguation_notes=[]
            )

        # Normalize text
        normalized_text = normalize_text(text)
        
        # Find all term matches
        matches = {term_type.value: [] for term_type in TermType}
        token_matches = []
        for term in TERMS:
            term_matches = find_term_matches(normalized_text, term)
            if term_matches:
                matches[term.type.value].extend(term_matches)
                token_matches.extend(term_matches)

        # Analyze context
        context_score = analyze_context_window(normalized_text, matches)
        
        # Check for manipulation
        manipulation_score, manipulation_details = detect_manipulation(text)
        
        # Calculate relevance metrics
        primary_score = len(matches[TermType.PRIMARY.value]) * 0.5
        technical_score = len(matches[TermType.TECHNICAL.value]) * 0.4
        secondary_score = len(matches[TermType.SECONDARY.value]) * 0.3
        context_dependent_score = len(matches[TermType.CONTEXT_DEPENDENT.value]) * 0.2
        
        total_score = (
            primary_score +
            technical_score +
            secondary_score * (1 + context_score) +  # Context boost for secondary terms
            context_dependent_score * (2 * context_score)  # Stronger context boost for context-dependent terms
        )
        
        # Apply manipulation penalty
        if manipulation_score > 0.4:
            total_score *= (1 - manipulation_score * 0.5)
        
        # Determine relevance level and confidence
        if total_score >= self.relevance_metrics['confidence_thresholds'][RelevanceLevel.HIGH]:
            level = RelevanceLevel.HIGH
            confidence = min(1.0, total_score / 2)
        elif total_score >= self.relevance_metrics['confidence_thresholds'][RelevanceLevel.MEDIUM]:
            level = RelevanceLevel.MEDIUM
            confidence = total_score * 0.8
        elif total_score >= self.relevance_metrics['confidence_thresholds'][RelevanceLevel.LOW]:
            level = RelevanceLevel.LOW
            confidence = total_score * 0.6
        else:
            level = RelevanceLevel.IRRELEVANT
            confidence = max(0.0, 1 - total_score)

        # Determine context type
        if matches[TermType.PRIMARY.value] and matches[TermType.TECHNICAL.value]:
            context_type = ContextType.TECHNICAL
        elif matches[TermType.PRIMARY.value]:
            context_type = ContextType.CRYPTO_SPECIFIC
        elif any(matches[t] for t in [TermType.SECONDARY.value, TermType.CONTEXT_DEPENDENT.value]):
            context_type = ContextType.MIXED
        else:
            context_type = ContextType.IRRELEVANT

        # Create context info
        context_info = ContextInfo(
            context_type=context_type,
            primary_terms=matches[TermType.PRIMARY.value],
            secondary_terms=matches[TermType.SECONDARY.value],
            context_terms=matches[TermType.CONTEXT.value],
            confidence=context_score
        )

        # Generate disambiguation notes
        disambiguation_notes = []
        if context_score > 0.5:
            disambiguation_notes.append(f"Strong contextual evidence (score: {context_score:.2f})")
        if manipulation_score > 0.3:
            disambiguation_notes.append(f"Potential manipulation detected (score: {manipulation_score:.2f})")
        
        # Collect conflicting evidence
        conflicting_evidence = []
        if manipulation_score > 0.4:
            conflicting_evidence.extend([
                f"High manipulation score: {manipulation_score:.2f}",
                f"Manipulation type: {manipulation_details['confidence']}"
            ])
        
        # Generate explanation
        if matches[TermType.PRIMARY.value]:
            explanation = f"Found crypto-specific terms: {', '.join(matches[TermType.PRIMARY.value])}"
        elif matches[TermType.TECHNICAL.value]:
            explanation = f"Found technical terms: {', '.join(matches[TermType.TECHNICAL.value])}"
        elif matches[TermType.SECONDARY.value]:
            explanation = f"Found secondary terms: {', '.join(matches[TermType.SECONDARY.value])}"
        else:
            explanation = "No cryptocurrency-related terms found in the text."

        return RelevanceScore(
            level=level,
            confidence=confidence,
            supporting_evidence={
                'crypto_terms': token_matches,
                'context_terms': matches[TermType.CONTEXT.value],
                'manipulation': {
                    'score': manipulation_score,
                    'details': manipulation_details
                },
                'analysis': {
                    'term_details': [
                        {
                            'term': match,
                            'type': term_type,
                            'weight': total_score / max(1, len(token_matches)),
                            'context_score': context_score,
                            'matches': matches[term_type]
                        }
                        for term_type in matches
                        for match in matches[term_type]
                    ],
                    'threshold': self.relevance_metrics['confidence_thresholds'][level],
                    'modifiers': {
                        'technical_bonus': bool(matches[TermType.TECHNICAL.value]),
                        'web3_bonus': bool(matches[TermType.WEB3.value]),
                        'multiple_secondary': len(matches[TermType.SECONDARY.value]) > 1,
                        'multiple_types': len([t for t in matches if matches[t]]) > 1
                    }
                }
            },
            explanation=explanation,
            context_info=context_info,
            token_matches=token_matches,
            conflicting_evidence=conflicting_evidence,
            disambiguation_notes=disambiguation_notes
        )

    def get_relevance_metrics(self) -> Dict[str, Any]:
        """Get the current relevance detection metrics and thresholds."""
        return self.relevance_metrics.copy() 