import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Enable parallel tokenization explicitly

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from typing import Dict, List, Tuple
import logging
import os
import random
from tqdm import tqdm
from .data_augmentation import get_balanced_dataset, prepare_dataset

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class SentimentTrainer:
    def __init__(self, model_name: str = "ElKulako/cryptobert"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model with dropout
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=3,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2
        )
        
        # Add extra dropout layer
        self.model.dropout = torch.nn.Dropout(0.3)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
    def prepare_data(self, batch_size: int = 16) -> tuple:
        """Prepare and split the dataset with larger validation set"""
        dataset = get_balanced_dataset()
        texts = [item[0] for item in dataset]
        sentiments = [item[1] for item in dataset]
        
        # Convert sentiments to numeric labels
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        labels = [sentiment_map[s] for s in sentiments]
        
        # Create full dataset
        full_dataset = CryptoSentimentDataset(texts, labels, self.tokenizer)
        
        # Calculate split sizes (70% train, 30% validation)
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = total_size - train_size
        
        logger.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")
        
        # Split dataset with fixed random seed
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(RANDOM_SEED)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            generator=torch.Generator().manual_seed(RANDOM_SEED)
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size
        )
        
        return train_loader, val_loader
    
    def compute_metrics(self, predictions, labels) -> Dict:
        """Compute comprehensive training metrics"""
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == labels)
        
        # Calculate per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(['negative', 'neutral', 'positive']):
            mask = labels == i
            pred_mask = predictions == i
            
            if np.any(mask):
                # True positives, false positives, false negatives
                tp = np.sum(predictions[mask] == i)
                fp = np.sum(pred_mask & (labels != i))
                fn = np.sum(mask & (predictions != i))
                
                # Calculate precision, recall, f1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics.update({
                    f"{class_name}_precision": precision,
                    f"{class_name}_recall": recall,
                    f"{class_name}_f1": f1
                })
        
        metrics = {
            "accuracy": accuracy,
            **class_metrics
        }
        
        return metrics
    
    def train(self, 
              output_dir: str = "models/crypto_sentiment", 
              num_epochs: int = 10,
              learning_rate: float = 2e-5,
              weight_decay: float = 0.01):
        """Train the model with improved training loop"""
        train_loader, val_loader = self.prepare_data()
        
        # Setup optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup learning rate scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,  # Warm-up for 10% of training
            anneal_strategy='cos'
        )
        
        # Training loop
        best_accuracy = 0.0
        best_metrics = None
        patience = 3
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc="Training")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass with dropout
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'lr': scheduler.get_last_lr()[0]
                })
            
            avg_train_loss = total_loss / len(train_loader)
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            self.model.eval()
            val_predictions = []
            val_labels = []
            val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
                    val_predictions.extend(outputs.logits.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            metrics = self.compute_metrics(
                np.array(val_predictions),
                np.array(val_labels)
            )
            metrics['val_loss'] = val_loss / len(val_loader)
            
            logger.info("Validation Results:")
            for key, value in metrics.items():
                logger.info(f"{key}: {value:.4f}")
            
            # Save best model and early stopping
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_metrics = metrics
                patience_counter = 0
                
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Save model and tokenizer
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                logger.info(f"New best model saved to {output_dir}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        logger.info("\nBest Model Metrics:")
        for key, value in best_metrics.items():
            logger.info(f"{key}: {value:.4f}")
        
        return best_metrics

def train_model():
    """Main function to train the model"""
    trainer = SentimentTrainer()
    metrics = trainer.train()
    return metrics

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == labels)
    
    # Calculate per-class metrics
    class_metrics = {}
    for i, class_name in enumerate(["negative", "neutral", "positive"]):
        class_pred = predictions == i
        class_true = labels == i
        
        true_pos = np.sum(class_pred & class_true)
        false_pos = np.sum(class_pred & ~class_true)
        false_neg = np.sum(~class_pred & class_true)
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
    
    return {
        "accuracy": accuracy,
        **{f"{class_name}_{metric}": value 
           for class_name, metrics in class_metrics.items() 
           for metric, value in metrics.items()}
    }

def fine_tune_model(
    output_dir: str = "fine_tuned_cryptobert",
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    save_steps: int = 500,
    eval_steps: int = 100
):
    """Fine-tune CryptoBERT on crypto sentiment dataset"""
    
    logger.info("Loading CryptoBERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    model = AutoModelForSequenceClassification.from_pretrained(
        "ElKulako/cryptobert",
        num_labels=3
    )
    
    logger.info("Preparing dataset...")
    texts, labels = prepare_dataset()
    dataset = CryptoSentimentDataset(texts, labels, tokenizer)
    
    # Split dataset into train and eval
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        warmup_steps=100,
        weight_decay=weight_decay,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        learning_rate=learning_rate,
        evaluation_strategy="steps",
        save_steps=save_steps,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    
    logger.info("Starting fine-tuning...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving fine-tuned model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate the model
    logger.info("Evaluating fine-tuned model...")
    eval_results = trainer.evaluate()
    
    logger.info("Evaluation Results:")
    for metric, value in eval_results.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return eval_results

if __name__ == "__main__":
    train_model() 