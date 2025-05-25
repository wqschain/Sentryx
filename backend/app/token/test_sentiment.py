from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from .sentiment_rules import adjust_sentiment
from .test_data import TEST_CASES, get_all_test_cases
from collections import defaultdict
import json
from datetime import datetime

def analyze_sentiment(tokenizer, model, text: str) -> tuple[str, float]:
    """Analyze sentiment of a given text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_scores = probabilities[0].tolist()
    
    # CryptoBERT classes: positive (2), negative (0), neutral (1)
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    # Get all sentiment probabilities
    sentiments_with_scores = {
        sentiment_map[i]: score 
        for i, score in enumerate(sentiment_scores)
    }
    
    # Always apply sentiment rules to refine the classification
    final_sentiment, final_confidence = adjust_sentiment(
        sentiment_map[torch.argmax(probabilities[0]).item()],
        max(sentiment_scores),
        text
    )
    
    return final_sentiment, final_confidence

def calculate_metrics(results: list) -> dict:
    """Calculate accuracy metrics from test results"""
    metrics = {
        "total_cases": len(results),
        "correct_predictions": 0,
        "accuracy": 0.0,
        "confusion_matrix": defaultdict(lambda: defaultdict(int)),
        "category_accuracy": defaultdict(lambda: {"correct": 0, "total": 0, "accuracy": 0.0}),
        "confidence_analysis": {
            "average_confidence": 0.0,
            "correct_confidence": 0.0,
            "incorrect_confidence": 0.0
        }
    }
    
    total_confidence = 0.0
    correct_confidence = 0.0
    incorrect_confidence = 0.0
    incorrect_count = 0
    
    for result in results:
        category = result["category"]
        predicted = result["predicted_sentiment"]
        actual = result["expected_sentiment"]
        confidence = result["confidence"]
        
        # Update confusion matrix
        metrics["confusion_matrix"][actual][predicted] += 1
        
        # Update category metrics
        metrics["category_accuracy"][category]["total"] += 1
        
        # Update confidence metrics
        total_confidence += confidence
        
        if predicted == actual:
            metrics["correct_predictions"] += 1
            metrics["category_accuracy"][category]["correct"] += 1
            correct_confidence += confidence
        else:
            incorrect_confidence += confidence
            incorrect_count += 1
    
    # Calculate accuracy metrics
    metrics["accuracy"] = metrics["correct_predictions"] / metrics["total_cases"]
    
    # Calculate category accuracies
    for category in metrics["category_accuracy"]:
        cat_metrics = metrics["category_accuracy"][category]
        if cat_metrics["total"] > 0:
            cat_metrics["accuracy"] = cat_metrics["correct"] / cat_metrics["total"]
    
    # Calculate confidence metrics
    metrics["confidence_analysis"]["average_confidence"] = total_confidence / metrics["total_cases"]
    metrics["confidence_analysis"]["correct_confidence"] = (
        correct_confidence / metrics["correct_predictions"] if metrics["correct_predictions"] > 0 else 0.0
    )
    metrics["confidence_analysis"]["incorrect_confidence"] = (
        incorrect_confidence / incorrect_count if incorrect_count > 0 else 0.0
    )
    
    return metrics

def run_sentiment_analysis():
    """Run sentiment analysis tests and generate detailed report"""
    print("Initializing CryptoBERT model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "ElKulako/cryptobert",
        trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "ElKulako/cryptobert",
        trust_remote_code=True
    )
    
    print("\nRunning sentiment analysis tests...")
    results = []
    
    for category, test_cases in TEST_CASES.items():
        print(f"\nTesting {category} cases:")
        print("-" * 80)
        
        for text, expected_sentiment, expected_confidence in test_cases:
            predicted_sentiment, confidence = analyze_sentiment(tokenizer, model, text)
            
            result = {
                "category": category,
                "text": text,
                "predicted_sentiment": predicted_sentiment,
                "expected_sentiment": expected_sentiment,
                "confidence": confidence,
                "expected_confidence": expected_confidence
            }
            results.append(result)
            
            # Print individual result
            print(f"\nText: {text}")
            print(f"Expected: {expected_sentiment} (conf: {expected_confidence:.2f})")
            print(f"Predicted: {predicted_sentiment} (conf: {confidence:.2f})")
            print(f"{'✓' if predicted_sentiment == expected_sentiment else '✗'}")
    
    # Calculate and print metrics
    metrics = calculate_metrics(results)
    
    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": "ElKulako/cryptobert",
        "total_cases": metrics["total_cases"],
        "overall_accuracy": metrics["accuracy"],
        "metrics": metrics,
        "results": results
    }
    
    # Save detailed report
    with open("sentiment_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total test cases: {metrics['total_cases']}")
    print(f"Overall accuracy: {metrics['accuracy']*100:.1f}%")
    print("\nCategory Accuracy:")
    for category, cat_metrics in metrics["category_accuracy"].items():
        print(f"- {category}: {cat_metrics['accuracy']*100:.1f}%")
    
    print("\nConfidence Analysis:")
    conf_analysis = metrics["confidence_analysis"]
    print(f"- Average confidence: {conf_analysis['average_confidence']:.2f}")
    print(f"- Correct predictions confidence: {conf_analysis['correct_confidence']:.2f}")
    print(f"- Incorrect predictions confidence: {conf_analysis['incorrect_confidence']:.2f}")
    
    print("\nDetailed report saved to sentiment_analysis_report.json")

if __name__ == "__main__":
    run_sentiment_analysis() 