import logging
from pathlib import Path
from src.utils.config import config

def get_logger(name:str) -> logging.Logger : 

    logger=logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        
        Path(config.paths.log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(f"{config.paths.log_dir}/project_run.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger