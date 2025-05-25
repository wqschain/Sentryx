from app.token.model_training import train_model
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    logging.info("Starting model training with augmented dataset...")
    try:
        metrics = train_model()
        
        logging.info("\nTraining completed! Final metrics:")
        for metric_name, value in metrics.items():
            logging.info(f"{metric_name}: {value:.4f}")
            
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise 