import time
from contextlib import contextmanager
from typing import Tuple,Generator
import numpy as np
from keras.datasets import mnist
from src.utils.config import config
from src.utils.logger import get_logger
from src.utils.exceptions import DataLoadError

logger = get_logger(__name__)

@contextmanager
def data_loading_context():

    start_time = time.time()

    logger.info

    try:
        yield

    except Exception as e:

        logger.error(f"Eroor in loading Data : {str(e)}")
        raise DataLoadError(f"Failed to load data: {e}")

    finally:

        elapsed_time = time.time() - start_time
        logger.info(f"Operation Completed ,time: {elapsed_time:.2f} s")

def load_and_preprocess_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

    with data_loading_context():
        try:
            (X_train, y_train), (X_test, y_test) = mnist.load_data()

            X_train = X_train.astype("float32") / 255.0
            X_test = X_test.astype("float32") / 255.0

            X_train = np.expand_dims(X_train, -1)
            X_test = np.expand_dims(X_test, -1)

            logger.info(f"Data Is Ready. Data Shape: {X_train.shape}")

            return (X_train, y_train), (X_test, y_test)

        except Exception as e:

            raise DataLoadError(f"Eroro loading data from Keras:  {str(e)}")

def batch_generator(X: np.ndarray, y: np.ndarray, batch_size: int = config.model.batch_size) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:

    num_samples = X.shape[0]
    indices = np.arange(num_samples)

    np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):

        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
       
        yield X[batch_indices], y[batch_indices]