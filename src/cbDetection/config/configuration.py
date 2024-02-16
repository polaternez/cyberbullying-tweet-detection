import os
from pathlib import Path

from cbDetection.constants import *
from cbDetection.utils.common import read_yaml, create_directories
from cbDetection.entity import (
    DataIngestionConfig,
    DataCleaningConfig,
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
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )
        return data_ingestion_config
    
    # Data cleaning config
    def get_data_cleaning_config(self) -> DataCleaningConfig:
        config = self.config.data_cleaning
        
        create_directories([config.root_dir])
        data_cleaning_config = DataCleaningConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
        )
        return data_cleaning_config
    
    # Data transformation config
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        
        create_directories([config.root_dir])
        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            cleaned_data_path=Path(config.cleaned_data_path),
            preprocessor_name=config.preprocessor_name
        )
        return data_transformation_config
    
    # Model trainer config
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params

        check_null = lambda x: x if x else dict() 

        create_directories([config.root_dir])
        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            model_name=config.model_name,

            params_xgboost=check_null(params.xgboost),
            params_logistic_regression=check_null(params.logistic_regression),
            params_naive_bayes=check_null(params.naive_bayes),
            params_decision_tree=check_null(params.decision_tree),
            params_random_forest=check_null(params.random_forest)
        )
        return model_trainer_config
    
   