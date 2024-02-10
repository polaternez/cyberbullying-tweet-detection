import os
import sys
from pathlib import Path
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


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.data_transformation_config = config

    def get_preprocessor(self):
        '''
        Creates a data preprocessor
        '''
        transform_pipeline = Pipeline([
            ("count_vect", CountVectorizer()),
            ("tfidf_trans", TfidfTransformer())
        ])
        return transform_pipeline
    
    def initiate_data_transformation(self, cleaned_data_path):
        """
        Transforms data and saves the preprocessor
        """
        try:
            # Load the cleaned data
            cleaned_tweets_df = pd.read_csv(cleaned_data_path)
            logger.info("Cleaned dataset loaded")

            # Drop null rows
            cleaned_tweets_df = cleaned_tweets_df.dropna()

            # train-test split
            X = cleaned_tweets_df["cleaned_text"].values.astype("U")    # "U" for Unicode string
            y = cleaned_tweets_df["is_cyberbullying"].values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=0.3,
                                                                random_state=42)
            logger.info(f"#Train rows: {X_train.shape[0]} - #Test rows: {X_test.shape[0]}")
            
            # Data preprocessing
            preprocessor = self.get_preprocessor()
            logger.info("Preprocessor is created")

            logger.info("Applying the preprocessor to the training dataframe and testing dataframe")
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # Save the preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_path,
                obj=preprocessor
            )
            logger.info(f"Preprocessor saved.")
            
            return (
                (X_train_processed, y_train),
                (X_test_processed, y_test),
                self.data_transformation_config.preprocessor_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
        
        

