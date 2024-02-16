import os
import sys
from pathlib import Path
import time
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from cbDetection.entity import DataTransformationConfig
from cbDetection.utils.exception import CustomException
from cbDetection import logger
from cbDetection.utils.ml_helper import save_object
from cbDetection.utils.text_cleaning import clean_text


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_preprocessor(self):
        '''
        Creates a data preprocessor
        '''
        transform_pipeline = Pipeline([
            ("count_vect", CountVectorizer()),
            ("tfidf_trans", TfidfTransformer())
        ])
        return transform_pipeline
    
    def initiate_data_transformation(self):
        """
        Transforms data and saves the preprocessor
        """
        logger.info("Starting data transformation")
        try:
            cleaned_df = pd.read_csv(self.config.cleaned_data_path)
            logger.info(f"Loaded cleaned data from {self.config.cleaned_data_path}")

            # train-test split
            X = cleaned_df["cleaned_text"].values.astype("U")    # "U" for Unicode string
            y = cleaned_df["is_cyberbullying"].values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=0.3,
                                                                random_state=42)
            logger.info(f"Train-test split: Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
            
            # Data preprocessing
            preprocessor = self.get_preprocessor()
            logger.info("Preprocessor created")

            # Data preprocessing
            start_time = time.time()
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            processing_time = time.time() - start_time
            logger.info(f"Applied preprocessor to train/test data (time: {processing_time:.2f}s)")

            # Save the preprocessor
            save_object(
                file_path=os.path.join(self.config.root_dir, self.config.preprocessor_name),
                obj=preprocessor
            )
            logger.info(f"Preprocessor saved to {self.config.preprocessor_name}")
            
            return (
                (X_train_processed, y_train),
                (X_test_processed, y_test)
            )
        except Exception as e:
            raise CustomException(e, sys)
        
        

