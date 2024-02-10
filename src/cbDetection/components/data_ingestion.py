import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from cbDetection.entity import DataIngestionConfig
from cbDetection.utils.exception import CustomException
from cbDetection import logger
from cbDetection.utils.text_cleaning import clean_text


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.ingestion_config = config

    def initiate_data_ingestion(self):
        logger.info("Initiating data ingestion")
        try:
            raw_df = pd.read_csv(self.ingestion_config.source_file)
            logger.info('Dataset loaded')

            # Save the raw dataframe
            raw_df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logger.info("Initiated basic data cleaning")
            # Add 'is_cyberbullying' column
            raw_df["is_cyberbullying"] = [
                0 if x=="not_cyberbullying" else 1 for x in raw_df["cyberbullying_type"]
            ]

            # Clean tweet text
            raw_df["cleaned_text"] = raw_df["tweet_text"].apply(clean_text)

            # # Save the cleaned dataframe
            cleaned_df = raw_df[['cleaned_text', 'is_cyberbullying']]
            cleaned_df.to_csv(self.ingestion_config.cleaned_data_path, index=False)
            logger.info("Data ingestion completed")

            return self.ingestion_config.cleaned_data_path
        except Exception as e:
            raise CustomException(e, sys)
        

        


    
