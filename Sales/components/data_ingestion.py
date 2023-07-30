import os
import sys
from Sales.Logger.logger import logging
from Sales.Exception.exception import CustomException
import pandas as pd
from dataclasses import dataclass

TRAIN_RAW_DATA_PATH = r"F:\DOCUMENTS\Project\Stores-Sales-Prediction\raw_data\Train.csv"
TEST_RAW_DATA_PATH = r"F:\DOCUMENTS\Project\Stores-Sales-Prediction\raw_data\Test.csv"

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

def read_train_data():
    ingestion_config = DataIngestionConfig()
    logging.info("Reading the Train data")
    try:
        df = pd.read_csv(TRAIN_RAW_DATA_PATH)
        os.makedirs(os.path.dirname(ingestion_config.train_data_path), exist_ok=True)
        df.to_csv(ingestion_config.train_data_path, index=False, header=True)
        logging.info("Reading the Train data successful")
        return ingestion_config.train_data_path
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)


def read_test_data():
    ingestion_config = DataIngestionConfig()
    logging.info("Reading the test data")
    try:
        df = pd.read_csv(TEST_RAW_DATA_PATH)
        os.makedirs(os.path.dirname(ingestion_config.test_data_path), exist_ok=True)
        df.to_csv(ingestion_config.test_data_path, index=False, header=True)
        logging.info("Reading the test data successful")
        return ingestion_config.test_data_path
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
        