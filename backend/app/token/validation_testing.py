import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import List, Tuple, Dict
import logging
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationTester:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.label_mapping = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }
        
        self.results_log = []
        
    def predict_sentiment(self, text: str) -> Tuple[str, float]:
        """Predict sentiment for a single piece of text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1)
            confidence = torch.max(probs).item()
            
        predicted_label = self.label_mapping[prediction.item()]
        return predicted_label, confidence
    
    def validate_batch(self, validation_data: List[Tuple[str, str, float]]) -> Dict:
        """Run validation on a batch of real-world data"""
        results = {
            "predictions": [],
            "actual_labels": [],
            "confidences": [],
            "texts": []
        }
        
        for text, true_label, _ in validation_data:
            pred_label, confidence = self.predict_sentiment(text)
            
            results["predictions"].append(pred_label)
            results["actual_labels"].append(true_label)
            results["confidences"].append(confidence)
            results["texts"].append(text)
            
            # Log individual prediction
            self.results_log.append({
                "text": text,
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            })
        
        return results
    
    def analyze_results(self, results: Dict) -> Dict:
        """Analyze validation results and generate metrics"""
        predictions = results["predictions"]
        actual_labels = results["actual_labels"]
        confidences = results["confidences"]
        
        # Calculate confusion matrix
        labels = ["negative", "neutral", "positive"]
        conf_matrix = confusion_matrix(actual_labels, predictions, labels=labels)
        
        # Generate classification report
        class_report = classification_report(actual_labels, predictions, labels=labels, output_dict=True)
        
        # Analyze confidence distribution
        confidence_analysis = {
            "mean": np.mean(confidences),
            "std": np.std(confidences),
            "min": np.min(confidences),
            "max": np.max(confidences)
        }
        
        # Analyze errors
        errors = []
        for i, (pred, actual, conf, text) in enumerate(zip(
            predictions, actual_labels, confidences, results["texts"]
        )):
            if pred != actual:
                errors.append({
                    "text": text,
                    "predicted": pred,
                    "actual": actual,
                    "confidence": conf
                })
        
        return {
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
            "confidence_analysis": confidence_analysis,
            "errors": errors
        }
    
    def plot_confusion_matrix(self, results: Dict, save_path: str = None):
        """Plot and optionally save confusion matrix visualization"""
        conf_matrix = np.array(results["confusion_matrix"])
        labels = ["negative", "neutral", "positive"]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_results(self, results: Dict, analysis: Dict, output_dir: str):
        """Save validation results and analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_path = f"{output_dir}/validation_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump({
                "results": results,
                "analysis": analysis,
                "log": self.results_log
            }, f, indent=4)
        
        # Save confusion matrix plot
        plot_path = f"{output_dir}/confusion_matrix_{timestamp}.png"
        self.plot_confusion_matrix(analysis, save_path=plot_path)
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Confusion matrix plot saved to {plot_path}")

def load_real_world_data(file_path: str) -> List[Tuple[str, str, float]]:
    """Load real-world validation data from CSV/JSON file"""
    # Implementation depends on your data format
    # This is a placeholder - implement based on your data source
    pass

def main():
    # Example usage
    model_path = "intensive_training_20250522_180404/phase_3"  # Use your latest model
    validator = ValidationTester(model_path)
    
    # Load your real-world validation data
    validation_data = [
        ("Bitcoin surges to $55,000 as institutional demand grows", "positive", 0.8),
        ("Market uncertainty grows amid regulatory concerns", "negative", 0.7),
        # Add more real-world examples...
    ]
    
    # Run validation
    results = validator.validate_batch(validation_data)
    
    # Analyze results
    analysis = validator.analyze_results(results)
    
    # Save results
    validator.save_results(results, analysis, "validation_results")
    
    # Print summary
    logger.info("\nValidation Summary:")
    logger.info(f"Total samples: {len(validation_data)}")
    logger.info(f"Accuracy: {analysis['classification_report']['accuracy']:.4f}")
    logger.info("\nPer-class F1 Scores:")
    for label in ['negative', 'neutral', 'positive']:
        logger.info(f"{label}: {analysis['classification_report'][label]['f1-score']:.4f}")

if __name__ == "__main__":
    main() 