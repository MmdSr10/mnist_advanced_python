import time
from functools import wraps
from typing import Callable, Any
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras import Model
from src.utils.config import config
from src.utils.logger import get_logger
from src.utils.exceptions import ModelBuildError

logger = get_logger(__name__)

def log_build_time(func:Callable) -> Callable:

    @wraps(func)
    def wrapper(*args,**kwargs:Any)-> Any:
        start_time=time.time()

        logger.info(f"Starting execution of method: {func.__name__} to build the model...")

        try:

            result = func(*args, **kwargs)

        except Exception as e:

            logger.error(f"Error executing method {func.__name__}: {str(e)}")
            raise ModelBuildError(f"Model building failed in {func.__name__}") from e  

        end_time=time.time()

        logger.info(f"Finished execution of method: {func.__name__} - Build time: {end_time - start_time:.4f} seconds")

        return result

    return wrapper
class MNISTClassifier:
    
    def __init__(self) -> None:

        self.input_shape = config.model.image_shape
        self.num_classes = config.model.num_classes

    @log_build_time
    def build(self) -> Model:
       
        try:
            model = Sequential([
                Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=self.input_shape),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                
                Conv2D(64, kernel_size=(3, 3), activation="relu"),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                
                Flatten(),
                Dropout(0.5),
                Dense(self.num_classes, activation="softmax")
            ])
            logger.info("Model architecture built successfully and layers added.")
            return model
            
        except Exception as e:
            raise ModelBuildError(f"Error defining neural network layers: {str(e)}")