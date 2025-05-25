from typing import Dict, List, Tuple
from collections import Counter
import logging
from .data_augmentation import augment_dataset, get_balanced_dataset
from .test_data import TEST_CASES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetValidator:
    def __init__(self):
        self.original_data = TEST_CASES
        self.augmented_data = augment_dataset()
        self.balanced_data = get_balanced_dataset()
    
    def validate_distribution(self) -> Dict:
        """Validate sentiment distribution in datasets"""
        distributions = {}
        
        # Original distribution
        original_sentiments = []
        for cases in self.original_data.values():
            original_sentiments.extend([case[1] for case in cases])
        distributions['original'] = Counter(original_sentiments)
        
        # Augmented distribution
        augmented_sentiments = []
        for cases in self.augmented_data.values():
            augmented_sentiments.extend([case[1] for case in cases])
        distributions['augmented'] = Counter(augmented_sentiments)
        
        # Balanced distribution
        balanced_sentiments = [case[1] for case in self.balanced_data]
        distributions['balanced'] = Counter(balanced_sentiments)
        
        return distributions
    
    def validate_confidence_scores(self) -> Dict:
        """Analyze confidence score distribution"""
        scores = {
            'original': [],
            'augmented': [],
            'balanced': []
        }
        
        # Original scores
        for cases in self.original_data.values():
            scores['original'].extend([case[2] for case in cases])
            
        # Augmented scores
        for cases in self.augmented_data.values():
            scores['augmented'].extend([case[2] for case in cases])
            
        # Balanced scores
        scores['balanced'] = [case[2] for case in self.balanced_data]
        
        return {
            dataset: {
                'mean': sum(s)/len(s),
                'min': min(s),
                'max': max(s)
            } for dataset, s in scores.items() if s
        }
    
    def validate_category_coverage(self) -> Dict:
        """Analyze category distribution"""
        coverage = {}
        
        # Original coverage
        coverage['original'] = {
            category: len(cases) for category, cases in self.original_data.items()
        }
        
        # Augmented coverage
        coverage['augmented'] = {
            category: len(cases) for category, cases in self.augmented_data.items()
        }
        
        return coverage
    
    def validate_text_quality(self) -> Dict:
        """Basic text quality checks"""
        quality_metrics = {
            'original': {'avg_length': 0, 'unique_tokens': set()},
            'augmented': {'avg_length': 0, 'unique_tokens': set()},
            'balanced': {'avg_length': 0, 'unique_tokens': set()}
        }
        
        # Original quality
        original_texts = []
        for cases in self.original_data.values():
            original_texts.extend([case[0] for case in cases])
        quality_metrics['original']['avg_length'] = sum(len(t.split()) for t in original_texts) / len(original_texts)
        quality_metrics['original']['unique_tokens'] = len(set(' '.join(original_texts).split()))
        
        # Augmented quality
        augmented_texts = []
        for cases in self.augmented_data.values():
            augmented_texts.extend([case[0] for case in cases])
        quality_metrics['augmented']['avg_length'] = sum(len(t.split()) for t in augmented_texts) / len(augmented_texts)
        quality_metrics['augmented']['unique_tokens'] = len(set(' '.join(augmented_texts).split()))
        
        # Balanced quality
        balanced_texts = [case[0] for case in self.balanced_data]
        quality_metrics['balanced']['avg_length'] = sum(len(t.split()) for t in balanced_texts) / len(balanced_texts)
        quality_metrics['balanced']['unique_tokens'] = len(set(' '.join(balanced_texts).split()))
        
        return quality_metrics
    
    def run_validation(self) -> Dict:
        """Run all validation checks"""
        validation_results = {
            'distribution': self.validate_distribution(),
            'confidence_scores': self.validate_confidence_scores(),
            'category_coverage': self.validate_category_coverage(),
            'text_quality': self.validate_text_quality()
        }
        
        # Log validation results
        logger.info("Dataset Validation Results:")
        logger.info("1. Sentiment Distribution:")
        for dataset, dist in validation_results['distribution'].items():
            logger.info(f"  {dataset}: {dict(dist)}")
        
        logger.info("\n2. Confidence Scores:")
        for dataset, scores in validation_results['confidence_scores'].items():
            logger.info(f"  {dataset}: {scores}")
        
        logger.info("\n3. Category Coverage:")
        for dataset, coverage in validation_results['category_coverage'].items():
            logger.info(f"  {dataset}: {coverage}")
        
        logger.info("\n4. Text Quality Metrics:")
        for dataset, metrics in validation_results['text_quality'].items():
            logger.info(f"  {dataset}: {metrics}")
        
        return validation_results

def validate_dataset():
    """Main function to validate the dataset"""
    validator = DatasetValidator()
    return validator.run_validation() 