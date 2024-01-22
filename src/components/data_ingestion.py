import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.text_cleaning import clean_text

@dataclass
class DataIngestionConfig:
    data_dir: Path = Path('artifacts/data')
    raw_data_path: Path = data_dir / 'data.csv'
    cleaned_data_path: Path = data_dir / 'cleaned_data.csv'
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion")

        try:
            raw_df = pd.read_csv('data\\cyberbullying_tweets_v2.csv')

            logging.info('Dataset loaded')

            # Save the raw dataframe
            self.ingestion_config.data_dir.mkdir(parents=True, exist_ok=True)
            raw_df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Initiated basic data cleaning")

            # Add 'is_cyberbullying' column
            raw_df["is_cyberbullying"] = [0 if x=="not_cyberbullying" else 1 for x in raw_df["cyberbullying_type"]]

            # Clean tweet text
            raw_df["cleaned_text"] = raw_df["tweet_text"].apply(clean_text)

            # # Save the cleaned dataframe
            cleaned_df = raw_df[['cleaned_text', 'is_cyberbullying']]
            cleaned_df.to_csv(self.ingestion_config.cleaned_data_path, index=False)

            logging.info("Data ingestion completed")

            return self.ingestion_config.cleaned_data_path
        
        except Exception as e:
            raise CustomException(e, sys)
        


    
