import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from cbDetection.entity import DataCleaningConfig
from cbDetection.utils.exception import CustomException
from cbDetection import logger
from cbDetection.utils.text_cleaning import clean_text


class DataCleaning:
    def __init__(self, config: DataCleaningConfig):
        self.config = config

    def initiate_data_cleaning(self):
        logger.info("Starting data cleaning process")
        try:
            dataset = pd.read_csv(self.config.data_path)
            logger.info(f"Loaded dataset from {self.config.data_path}")

            # Add 'is_cyberbullying' column
            dataset["is_cyberbullying"] = [
                0 if x=="not_cyberbullying" else 1 for x in dataset["cyberbullying_type"]
            ]
            # Clean tweet text
            dataset["cleaned_text"] = dataset["tweet_text"].apply(clean_text)
            logger.info("Cleaned tweet text using clean_text function")

            cleaned_df = dataset[['cleaned_text', 'is_cyberbullying']]

            # Drop null rows
            dropped_rows = cleaned_df.dropna().shape[0] - cleaned_df.shape[0]
            cleaned_df = cleaned_df.dropna()
            if dropped_rows > 0:
                logger.warning(
                    f"Dropped {dropped_rows} rows containing missing values"
                )

            cleaned_df.to_csv(os.path.join(self.config.root_dir, "cleaned_tweets.csv"), index=False)
            logger.info("Data cleaning completed successfully")
        except Exception as e:
            raise CustomException(e, sys)
        

        


    
