import logging
import sys
import os
import json
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.token.model_training import SentimentTrainer, train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def intensive_training():
    """Run an intensive training session with multiple phases"""
    logger.info("Starting intensive training session...")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"intensive_training_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Training phases
    phases = [
        {
            'name': 'Phase 1 - High Learning Rate',
            'epochs': 10,
            'learning_rate': 5e-5,
            'batch_size': 16,
            'weight_decay': 0.01
        },
        {
            'name': 'Phase 2 - Medium Learning Rate',
            'epochs': 15,
            'learning_rate': 2e-5,
            'batch_size': 32,
            'weight_decay': 0.02
        },
        {
            'name': 'Phase 3 - Fine Tuning',
            'epochs': 20,
            'learning_rate': 5e-6,
            'batch_size': 32,
            'weight_decay': 0.03
        }
    ]
    
    # Track best metrics across all phases
    best_metrics = None
    best_accuracy = 0.0
    
    # Run each training phase
    for i, phase in enumerate(phases, 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting {phase['name']}")
        logger.info(f"{'='*50}")
        
        # Create phase-specific output directory
        phase_output_dir = os.path.join(base_output_dir, f"phase_{i}")
        os.makedirs(phase_output_dir, exist_ok=True)
        
        # Initialize trainer (new trainer for each phase)
        trainer = SentimentTrainer()
        
        # Train for this phase
        metrics = trainer.train(
            output_dir=phase_output_dir,
            num_epochs=phase['epochs'],
            learning_rate=phase['learning_rate'],
            weight_decay=phase['weight_decay']
        )
        
        # Save phase metrics
        metrics_file = os.path.join(phase_output_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Update best metrics if improved
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_metrics = metrics.copy()
            best_metrics['phase'] = phase['name']
        
        # Print phase summary
        logger.info(f"\nPhase {i} Results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info("F1 Scores:")
        logger.info(f"- Negative: {metrics['negative_f1']:.4f}")
        logger.info(f"- Neutral:  {metrics['neutral_f1']:.4f}")
        logger.info(f"- Positive: {metrics['positive_f1']:.4f}")
        logger.info(f"Validation Loss: {metrics['val_loss']:.4f}")
    
    # Save best overall metrics
    best_metrics_file = os.path.join(base_output_dir, 'best_metrics.json')
    with open(best_metrics_file, 'w') as f:
        json.dump(best_metrics, f, indent=4)
    
    logger.info("\n" + "="*50)
    logger.info("Training Complete!")
    logger.info("="*50)
    logger.info(f"\nBest Performance (from {best_metrics['phase']}):")
    logger.info(f"Accuracy: {best_metrics['accuracy']:.4f}")
    logger.info("F1 Scores:")
    logger.info(f"- Negative: {best_metrics['negative_f1']:.4f}")
    logger.info(f"- Neutral:  {best_metrics['neutral_f1']:.4f}")
    logger.info(f"- Positive: {best_metrics['positive_f1']:.4f}")
    logger.info(f"Validation Loss: {best_metrics['val_loss']:.4f}")
    logger.info(f"\nAll models and metrics saved in: {base_output_dir}")

if __name__ == "__main__":
    # Print system information
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run intensive training
    intensive_training() 