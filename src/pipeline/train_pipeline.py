from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def main():
    data_ingestion = DataIngestion()
    cleaned_data_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    (X_train, y_train), (X_test, y_test), _ = data_transformation.initiate_data_transformation(cleaned_data_path)
    
    model_trainer = ModelTrainer()
    
    print(model_trainer.initiate_model_trainer((X_train, y_train), (X_test, y_test)))

if __name__ == "__main__":
    main()    