import os
from pathlib import Path

from cbDetection.constants import *
from cbDetection.utils.common import read_yaml, create_directories
from cbDetection.entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    # Data ingestion config
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_file=config.source_file,
            raw_data_path=config.raw_data_path,
            cleaned_data_path=config.cleaned_data_path
        )
        return data_ingestion_config
    
    # Data transformation config
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        
        create_directories([config.root_dir])
        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            preprocessor_path=Path(config.preprocessor_path)
        )
        return data_transformation_config
    
    # Model trainer config
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        
        create_directories([config.root_dir])
        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path)
        )
        return model_trainer_config
    
   