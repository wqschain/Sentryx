from typing import List, Tuple
import random
from .test_data import TEST_CASES
import re

class SentimentDataAugmenter:
    def __init__(self):
        self.price_values = ['$10,000', '$20,000', '$30,000', '$40,000', '$50,000', '$60,000', 
                           '$15,000', '$25,000', '$35,000', '$45,000', '$55,000']
        self.crypto_names = ['Bitcoin', 'Ethereum', 'BTC', 'ETH', 'Cardano', 'Solana', 
                           'ADA', 'SOL', 'Binance Coin', 'BNB', 'XRP', 'Dogecoin', 'DOGE']
        self.percentage_values = ['5%', '10%', '15%', '20%', '25%', '30%', '40%', '50%']
        self.time_frames = ['hourly', 'daily', 'weekly', 'monthly', 'yearly']
        self.market_terms = ['market', 'trading', 'price', 'volume', 'momentum']
        self.sentiment_intensifiers = {
            'positive': ['strongly', 'significantly', 'massively', 'extremely', 'incredibly'],
            'negative': ['severely', 'dramatically', 'heavily', 'sharply', 'critically'],
            'neutral': ['slightly', 'moderately', 'somewhat', 'relatively', 'marginally']
        }
        
    def substitute_values(self, text: str) -> str:
        """Replace numeric values while maintaining sentiment"""
        # Replace prices
        for value in self.price_values:
            if value in text:
                return text.replace(value, random.choice(self.price_values))
        
        # Replace percentages
        for value in self.percentage_values:
            if value in text:
                return text.replace(value, random.choice(self.percentage_values))
        
        return text
    
    def synonym_replacement(self, text: str) -> str:
        """Replace crypto terms with synonyms"""
        for name in self.crypto_names:
            if name in text:
                return text.replace(name, random.choice(self.crypto_names))
        return text
    
    def add_time_context(self, text: str) -> str:
        """Add time context to the text"""
        if not any(frame in text.lower() for frame in self.time_frames):
            time_frame = random.choice(self.time_frames)
            for term in self.market_terms:
                if term in text.lower():
                    return text.replace(term, f"{time_frame} {term}")
        return text
    
    def create_variations(self, text: str, sentiment: str, confidence: float) -> List[Tuple[str, str, float]]:
        """Create multiple variations of a single example"""
        variations = []
        
        # Basic variations
        variations.extend([
            (self.substitute_values(text), sentiment, confidence),
            (self.synonym_replacement(text), sentiment, confidence),
            (self.add_time_context(text), sentiment, confidence)
        ])
        
        # Combine transformations
        combined = self.substitute_values(self.synonym_replacement(text))
        variations.append((combined, sentiment, confidence))
        
        combined_time = self.add_time_context(combined)
        variations.append((combined_time, sentiment, confidence))
        
        # Add intensity variations
        intensifiers = self.sentiment_intensifiers[sentiment]
        for intensifier in intensifiers:
            # Add intensity at the start
            var = f"{intensifier} {text.lower()}"
            conf_adjustment = 0.1 if intensifier in ['strongly', 'severely'] else 0.05
            new_conf = min(confidence + conf_adjustment, 1.0)
            variations.append((var, sentiment, new_conf))
            
            # Add intensity in the middle (for certain patterns)
            if any(term in text.lower() for term in self.market_terms):
                for term in self.market_terms:
                    if term in text.lower():
                        var = text.lower().replace(term, f"{intensifier} {term}")
                        variations.append((var, sentiment, new_conf))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for v, s, c in variations:
            if v not in seen and v != text:
                seen.add(v)
                unique_variations.append((v, s, c))
        
        return unique_variations

def augment_dataset() -> dict:
    """Augment the entire dataset"""
    augmenter = SentimentDataAugmenter()
    augmented_data = {}
    
    for category, cases in TEST_CASES.items():
        augmented_data[category] = list(cases)  # Start with original cases
        for text, sentiment, confidence in cases:
            variations = augmenter.create_variations(text, sentiment, confidence)
            augmented_data[category].extend(variations)
    
    return augmented_data

def get_balanced_dataset() -> List[Tuple[str, str, float]]:
    """Get a balanced dataset with equal representation of sentiments"""
    augmented = augment_dataset()
    all_examples = []
    
    # Collect all examples
    for category in augmented.values():
        all_examples.extend(category)
    
    # Separate by sentiment
    positive = [x for x in all_examples if x[1] == 'positive']
    negative = [x for x in all_examples if x[1] == 'negative']
    neutral = [x for x in all_examples if x[1] == 'neutral']
    
    # Find target size (use the largest class size for maximum data)
    target_size = max(len(positive), len(negative), len(neutral))
    
    # Oversample smaller classes to match the largest
    balanced_positive = random.choices(positive, k=target_size)
    balanced_negative = random.choices(negative, k=target_size)
    balanced_neutral = random.choices(neutral, k=target_size)
    
    # Combine and shuffle
    balanced_data = balanced_positive + balanced_negative + balanced_neutral
    random.shuffle(balanced_data)
    
    return balanced_data

def prepare_dataset() -> Tuple[List[str], List[str]]:
    """Prepare dataset for model training by returning texts and labels"""
    dataset = get_balanced_dataset()
    texts = [item[0] for item in dataset]
    labels = [item[1] for item in dataset]
    return texts, labels 