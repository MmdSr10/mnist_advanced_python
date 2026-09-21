import os
import sys

from src.utils.logger import get_logger
from src.utils.exceptions import MNISTProjectError
from src.utils.config import config
from src.data.dataloader import load_and_preprocess_data
from src.models.cnn_model import MNISTClassifier
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator

logger = get_logger(__name__)

def main() -> None:
    logger.info("==================================================")
    logger.info("Starting Advanced MNIST CNN Project")
    logger.info("==================================================")
    
    try:
        # فاز ۱: بارگذاری داده‌ها (استفاده از Context Manager درون تابع)
        logger.info(">>> Phase 1: Data Loading & Preprocessing")
        (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()
        
        # فاز ۲: ساخت معماری مدل (تست عملکرد Decorator)
        logger.info(">>> Phase 2: Building CNN Architecture")
        classifier = MNISTClassifier()
        model = classifier.build()
        
        # فاز ۳: آموزش مدل (استفاده از Generator)
        logger.info(">>> Phase 3: Training Model")
        trainer = ModelTrainer(model)
        trainer.compile_model()
        trainer.train_with_generator(X_train, y_train)
        
        # فاز ۴: ارزیابی مدل
        logger.info(">>> Phase 4: Evaluating Model")
        saved_model_path = os.path.join(config.paths.model_save_dir, "cnn_model.keras")
        evaluator = ModelEvaluator(model_path=saved_model_path)
        evaluator.evaluate(X_test, y_test)
        
        logger.info("==================================================")
        logger.info("Project executed successfully!")
        logger.info("==================================================")
        
    except MNISTProjectError as e:
        logger.error(f"Project Execution Failed (Known Error): {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()