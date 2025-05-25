from typing import Dict, List, Tuple

# Keywords that strongly indicate sentiment
SENTIMENT_KEYWORDS = {
    "positive": [  # Bullish indicators
        # Price and Market (weight: 2.0)
        "all-time high", "surge", "rally", "gain", "record high",
        "outperform", "breakthrough", "milestone",
        
        # Adoption (weight: 1.8)
        "adoption", "integration", "partnership", "launch",
        "institutional", "mainstream", "accepts", "implements",
        
        # Technical (weight: 1.5)
        "golden cross", "support level", "buy pressure",
        "higher highs", "higher lows", "breakout confirmed",
        
        # DeFi and NFT (weight: 1.5)
        "yield farming", "staking rewards", "liquidity mining",
        "blue chip nfts", "floor price up", "trading volume up"
    ],
    "negative": [  # Bearish indicators
        # Security Issues (weight: 2.0)
        "hack", "exploit", "breach", "vulnerability", "stolen",
        "attack", "compromise", "scam", "fraud",
        
        # Market Decline (weight: 2.0)
        "crash", "collapse", "plunge", "decline", "dump",
        "sell-off", "liquidation", "capitulation",
        
        # Regulatory (weight: 1.8)
        "ban", "restrict", "crackdown", "investigation",
        "illegal", "non-compliant", "violation",
        
        # Technical (weight: 1.5)
        "death cross", "resistance broken", "lower lows",
        "lower highs", "breakdown"
    ],
    "neutral": [
        # Market State (weight: 1.0)
        "stable", "unchanged", "steady", "consolidation",
        
        # Technical (weight: 1.0)
        "volume", "hashrate", "difficulty", "network stats",
        
        # Development (weight: 1.0)
        "update", "maintenance", "implementation", "testing"
    ]
}

# Context modifiers that can affect sentiment
CONTEXT_MODIFIERS = {
    "positive_to_negative": [
        # Strong negative modifiers (weight: 1.5)
        "not", "fail", "reject", "delay",
        # Moderate negative modifiers (weight: 1.2)
        "despite", "however", "but", "although"
    ],
    "negative_to_positive": [
        # Strong positive modifiers (weight: 1.5)
        "recover", "resolve", "fix", "restore",
        # Moderate positive modifiers (weight: 1.2)
        "improve", "progress", "overcome", "address"
    ],
    "amplifiers": [
        # Strong amplifiers (weight: 2.0)
        "significant", "massive", "major", "critical",
        # Moderate amplifiers (weight: 1.5)
        "very", "highly", "strongly", "substantial"
    ]
}

def get_keyword_weight(keyword: str, sentiment: str) -> float:
    """Get the weight for a specific keyword based on its category"""
    
    # Price and market movement keywords
    if any(term in keyword for term in ["high", "surge", "rally", "crash", "plunge", "decline"]):
        return 2.0
    
    # Security and risk keywords
    if any(term in keyword for term in ["hack", "exploit", "breach", "scam", "fraud"]):
        return 2.0
    
    # Adoption and integration keywords
    if any(term in keyword for term in ["adoption", "integration", "partnership", "accepts"]):
        return 1.8
    
    # Regulatory keywords
    if any(term in keyword for term in ["ban", "regulate", "compliance", "legal"]):
        return 1.8
    
    # Technical analysis keywords
    if any(term in keyword for term in ["cross", "support", "resistance", "breakout"]):
        return 1.5
    
    # DeFi and NFT keywords
    if any(term in keyword for term in ["yield", "farming", "nft", "defi"]):
        return 1.5
    
    return 1.0

def get_modifier_weight(modifier: str, modifier_type: str) -> float:
    """Get the weight for a specific modifier based on its type and strength"""
    
    # Strong modifiers
    if modifier in ["not", "fail", "reject", "recover", "resolve", "fix"]:
        return 1.5
    
    # Moderate modifiers
    if modifier in ["despite", "however", "but", "improve", "progress"]:
        return 1.2
    
    # Strong amplifiers
    if modifier in ["significant", "massive", "major", "critical"]:
        return 2.0
    
    # Moderate amplifiers
    if modifier in ["very", "highly", "strongly"]:
        return 1.5
    
    return 1.0

def calculate_keyword_score(text: str) -> Dict[str, float]:
    """Calculate sentiment scores based on keyword presence with category-specific weights"""
    text = text.lower()
    scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    
    # Check for keywords with different weights
    for sentiment, keywords in SENTIMENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                weight = get_keyword_weight(keyword, sentiment)
                scores[sentiment] += weight
    
    # Apply context modifiers with weights
    for modifier_type, modifiers in CONTEXT_MODIFIERS.items():
        for modifier in modifiers:
            if modifier in text:
                weight = get_modifier_weight(modifier, modifier_type)
                
                if modifier_type == "positive_to_negative":
                    temp = scores["positive"]
                    scores["positive"] *= (1.0 - 0.3 * weight)
                    scores["negative"] += temp * (0.4 * weight)
                
                elif modifier_type == "negative_to_positive":
                    temp = scores["negative"]
                    scores["positive"] += temp * (0.4 * weight)
                    scores["negative"] *= (1.0 - 0.3 * weight)
                
                elif modifier_type == "amplifiers":
                    for sentiment in scores:
                        scores[sentiment] *= (1.0 + 0.2 * weight)
    
    # Reduce neutral bias
    if scores["neutral"] > 0:
        scores["neutral"] *= 0.7
    
    # Boost non-neutral scores if they're close
    max_non_neutral = max(scores["positive"], scores["negative"])
    if max_non_neutral > scores["neutral"] * 0.7:
        scores["neutral"] *= 0.5
    
    # Normalize scores
    total = sum(scores.values()) or 1.0
    return {k: v/total for k, v in scores.items()}

def adjust_sentiment(
    model_sentiment: str,
    model_confidence: float,
    text: str
) -> Tuple[str, float]:
    """Adjust model sentiment using rules and keywords"""
    
    # Get keyword-based scores
    keyword_scores = calculate_keyword_score(text.lower())
    
    # More aggressive weighting based on confidence
    if model_confidence < 0.7:
        model_weight = 0.2  # Reduced model weight for low confidence
        keyword_weight = 0.8  # Increased keyword weight
    elif model_confidence < 0.85:
        model_weight = 0.5
        keyword_weight = 0.5
    else:
        model_weight = 0.7
        keyword_weight = 0.3
    
    combined_scores = {
        "negative": 0.0,
        "neutral": 0.0,
        "positive": 0.0
    }
    
    # Add model's score with neutral penalty
    if model_sentiment == "neutral":
        # Reduce the impact of neutral predictions
        combined_scores[model_sentiment] += (model_confidence * model_weight * 0.7)
    else:
        combined_scores[model_sentiment] += model_confidence * model_weight
    
    # Add keyword scores with neutral penalty
    for sentiment, score in keyword_scores.items():
        if sentiment == "neutral":
            # Reduce the impact of neutral keywords
            combined_scores[sentiment] += score * keyword_weight * 0.7
        else:
            combined_scores[sentiment] += score * keyword_weight
    
    # Get final sentiment with preference for non-neutral
    if max(combined_scores["positive"], combined_scores["negative"]) > combined_scores["neutral"] * 0.8:
        # If any non-neutral score is close to neutral, prefer the non-neutral
        combined_scores["neutral"] *= 0.8
    
    final_sentiment = max(combined_scores.items(), key=lambda x: x[1])
    
    return final_sentiment[0], final_sentiment[1] 