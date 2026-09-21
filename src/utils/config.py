from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class ModelConfig: 
    batch_size:int=64
    epochs:int=10
    learning_rate:float=0.001
    image_shape:Tuple[int,int,int]=(28,28,1)
    num_classes: int = 10

@dataclass(frozen=True)
class PathConfig:
    data_dir: str = "data"
    model_save_dir: str = "saved_models"
    log_dir: str = "logs"

@dataclass(frozen=True)
class ProjectConfig:
    model: ModelConfig = ModelConfig()
    paths: PathConfig = PathConfig()

config = ProjectConfig()
