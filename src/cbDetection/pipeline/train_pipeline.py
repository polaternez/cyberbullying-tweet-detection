from cbDetection.config.configuration import ConfigurationManager
from cbDetection.components.data_ingestion import DataIngestion
from cbDetection.components.data_cleaning import DataCleaning
from cbDetection.components.data_transformation import DataTransformation
from cbDetection.components.model_trainer import ModelTrainer


def main():
    # Data ingestion
    config_manager = ConfigurationManager() 
    data_ingestion_config = config_manager.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion.download_file()
    data_ingestion.extract_zip_file()

    # Data cleaning
    data_cleaning_config = config_manager.get_data_cleaning_config()
    data_cleaning = DataCleaning(data_cleaning_config)
    data_cleaning.initiate_data_cleaning()
    
    # Data transformation
    data_transformation_config = config_manager.get_data_transformation_config()
    data_transformation = DataTransformation(data_transformation_config)
    (X_train, y_train), (X_test, y_test) = data_transformation.initiate_data_transformation()
    
    # Model training
    model_trainer_config = config_manager.get_model_trainer_config()
    model_trainer = ModelTrainer(model_trainer_config)
    print(model_trainer.initiate_model_trainer((X_train, y_train), (X_test, y_test)))


if __name__ == "__main__":
    main()    