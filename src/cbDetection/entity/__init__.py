from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: Path
    local_data_file: Path
    unzip_dir: Path 


@dataclass(frozen=True)
class DataCleaningConfig:
    root_dir: Path
    data_path: Path
    

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    cleaned_data_path: Path
    preprocessor_name: str


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_name: str
    params_xgboost: dict
    params_logistic_regression: dict
    params_naive_bayes: dict
    params_decision_tree: dict
    params_random_forest: dict