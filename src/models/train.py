import os 
import numpy as np
from  keras import Model
from src.utils.config import config
from src.utils.logger import get_logger
from src.utils.exceptions import MNISTProjectError
from src.data.dataloader import batch_generator

logger=get_logger(__name__)

class ModelTrainer:

    def __init__(self,model:Model) -> None:

        self.model=model
        self.epochs=config.model.epochs
        self.batch_size=config.model.batch_size
        self.save_dir = config.paths.model_save_dir

    def compile_model(self)-> None:

        try:
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("Model compiled successfully.")
        except Exception as e:
            logger.error(f'failed to compile model {str(e)}')
            raise MNISTProjectError(f"Compilation error: {e}")


    def train_with_generator(self,X_train: np.ndarray, y_train: np.ndarray) -> None:

        try:

            logger.info("Starting custom training loop using batch generator...")          
            os.makedirs(self.save_dir, exist_ok=True)

            batches_per_epoch=len(X_train) // self.batch_size

            for epoch in range(self.epochs):
                logger.info(f"--- Epoch {epoch + 1}/{self.epochs} ---")

                data_gen = batch_generator(X_train, y_train, self.batch_size)

                epoch_loss = 0.0
                epoch_acc = 0.0

                for step, (X_batch, y_batch) in enumerate(data_gen):

                    loss, acc = self.model.train_on_batch(X_batch, y_batch)
                    epoch_loss += loss
                    epoch_acc += acc
                    
                    if step % 100 == 0:
                        logger.info(f"Step {step}/{batches_per_epoch} - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

                avg_loss = epoch_loss / batches_per_epoch
                avg_acc = epoch_acc / batches_per_epoch
                logger.info(f"Epoch {epoch + 1} completed - Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_acc:.4f}")        

            save_path = os.path.join(self.save_dir, "cnn_model.keras")
            self.model.save(save_path)
            logger.info(f"Training completed. Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error during custom generator training: {str(e)}")
            raise MNISTProjectError(f"Generator training failed: {e}")    
        