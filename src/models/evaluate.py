import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report

from src.utils.config import config
from src.utils.logger import get_logger
from src.utils.exceptions import MNISTProjectError

logger = get_logger(__name__)

class ModelEvaluator:
    """Class responsible for evaluating the trained model."""
    
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = self._load_trained_model()

    def _load_trained_model(self):
        """Loads the model from the specified path."""
        try:
            model = load_model(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {str(e)}")
            raise MNISTProjectError(f"Model loading error: {e}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Evaluates the model on the test dataset and logs the metrics."""
        try:
            logger.info("Starting model evaluation...")
            loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"Test Loss: {loss:.4f}")
            logger.info(f"Test Accuracy: {accuracy:.4f}")
            
            # Generate predictions for detailed report
            predictions = self.model.predict(X_test)
            y_pred = np.argmax(predictions, axis=1)
            
            report = classification_report(y_test, y_pred)
            logger.info(f"Classification Report:\n{report}")
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise MNISTProjectError(f"Evaluation failed: {e}")