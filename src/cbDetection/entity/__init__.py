from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_file: Path
    raw_data_path: Path
    cleaned_data_path: Path


@dataclass
class DataTransformationConfig:
    root_dir: Path
    preprocessor_path: Path


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    trained_model_path: Path