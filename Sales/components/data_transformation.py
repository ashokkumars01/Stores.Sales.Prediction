import pandas as pd
import numpy as np
from Sales.Logger.logger import logging
from Sales.Exception.exception import CustomException
from Sales.components.data_ingestion import read_train_data, read_test_data
import os, sys
from statistics import mode
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    processed_train_data_path: str = os.path.join('processed_data', 'train.csv')
    processed_test_data_path: str = os.path.join('processed_data', 'test.csv')

def Data_Transformer(train_data_path, test_data_path):
    '''
    Data Transformation for Train data
    '''
    transformation_config = DataTransformationConfig()
    logging.info("Started data transformation for Train Data")
    
    logging.info("Started reading the train data")
    try:
        train_data = pd.read_csv(train_data_path)
        logging.info("Reading the train data completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    train_data.drop('Item_Identifier',axis=1, inplace=True)
    #train_data.drop('Outlet_Identifier', axis=1, inplace=True)

    logging.info(" Started Filling the null Values of Item weight")
    try:
        mean = train_data.groupby('Item_Type')['Item_Weight'].mean()
        for i in range(len(mean)):
            c1 = (train_data['Item_Type']==mean.index[i])&(train_data['Item_Weight'].isna()==True)
            train_data['Item_Weight'] = np.select([c1], [mean[i]], train_data['Item_Weight'])
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    logging.info("Completed filling the null values")

    logging.info("Started filling null values of Outlet size")
    try:
        train_data['Outlet_Size'].fillna(mode(train_data['Outlet_Size']),inplace=True)
        logging.info("Filling null values of Outlet size is completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    logging.info("Started replacing the values of Item fat content")
    try:
        fat_content = {"low fat": "Low Fat",
                    "LF": "Low Fat",
                    "reg":"Regular"}
        train_data["Item_Fat_Content"]= train_data["Item_Fat_Content"].replace(fat_content)
        logging.info("Replacing the values of Item fat content completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    le = LabelEncoder()
    logging.info("Applying label encoding to Item type")
    try:
        #train_data['Item_Type'] = le.fit_transform(train_data['Item_Type'])
        cols = ['Item_Type', 'Outlet_Identifier',]
        for i in cols:
            train_data[i] = le.fit_transform(train_data[i])
        logging.info("Applying label encoding to Item type completed")
    except ValueError as val:
        logging.exception(val)
        raise CustomException(val, sys)
    except KeyError as key:
        logging.exception(key)
        raise CustomException(key, sys)
    except NameError as name:
        logging.exception(name)
        raise CustomException(name, sys)
    
    logging.info("Started Label encoding to Item fat content")
    try:
        train_data.replace({"Low Fat":0, "Regular":1}, inplace=True)
        logging.info("Label encoding to Item fat content completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    logging.info("Started Label encoding to outlet size")
    try:
        train_data.replace({"Small":0, "Medium":1, "High":2}, inplace=True)
        logging.info("Label encoding to outlet size completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    logging.info("Started Label encoding to outlet location type")
    try:
        train_data.replace({"Tier 1":0, "Tier 2":1, "Tier 3":2}, inplace=True)
        logging.info(" Label encoding to outlet location type completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    logging.info("Started Label encoding to outlet type")
    try:
        train_data.replace({"Grocery Store":0, "Supermarket Type1":1, "Supermarket Type2":2, "Supermarket Type3":3}, inplace=True)
        logging.info("Label encoding to outlet type completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    logging.info("Completed data transformation for Train Data")
    
    '''
    Data Transformation for Test data
    '''
    logging.info("Started data transformation for Test Data")
    logging.info("Started reading the test data")
    try:
        test_data = pd.read_csv(test_data_path)
        logging.info("Reading the test data completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    test_data.drop('Item_Identifier',axis=1, inplace=True)
    #test_data.drop('Outlet_Identifier', axis=1, inplace=True)

    logging.info(" Started Filling the null Values of Item weight")
    try:
        mean = test_data.groupby('Item_Type')['Item_Weight'].mean()
        for i in range(len(mean)):
            c1 = (test_data['Item_Type']==mean.index[i])&(test_data['Item_Weight'].isna()==True)
            test_data['Item_Weight'] = np.select([c1], [mean[i]], test_data['Item_Weight'])
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    logging.info("Completed filling the null values")

    logging.info("Started filling null values of Outlet size")
    try:
        test_data['Outlet_Size'].fillna(mode(test_data['Outlet_Size']),inplace=True)
        logging.info("Filling null values of Outlet size is completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    logging.info("Started replacing the values of Item fat content")
    try:
        fat_content = {"low fat": "Low Fat",
                    "LF": "Low Fat",
                    "reg":"Regular"}
        test_data["Item_Fat_Content"]= test_data["Item_Fat_Content"].replace(fat_content)
        logging.info("Replacing the values of Item fat content completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    le = LabelEncoder()
    logging.info("Applying label encoding to Item type")
    try:
        #test_data['Item_Type'] = le.fit_transform(test_data['Item_Type'])
        cols = ['Item_Type', 'Outlet_Identifier',]
        for i in cols:
            test_data[i] = le.fit_transform(test_data[i])
        logging.info("Applying label encoding to Item type completed")
    except ValueError as val:
        logging.exception(val)
        raise CustomException(val, sys)
    except KeyError as key:
        logging.exception(key)
        raise CustomException(key, sys)
    except NameError as name:
        logging.exception(name)
        raise CustomException(name, sys)
    
    logging.info("Started Label encoding to Item fat content")
    try:
        test_data.replace({"Low Fat":0, "Regular":1}, inplace=True)
        logging.info("Label encoding to Item fat content completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    logging.info("Started Label encoding to outlet size")
    try:
        test_data.replace({"Small":0, "Medium":1, "High":2}, inplace=True)
        logging.info("Label encoding to outlet size completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    logging.info("Started Label encoding to outlet location type")
    try:
        test_data.replace({"Tier 1":0, "Tier 2":1, "Tier 3":2}, inplace=True)
        logging.info(" Label encoding to outlet location type completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    logging.info("Started Label encoding to outlet type")
    try:
        test_data.replace({"Grocery Store":0, "Supermarket Type1":1, "Supermarket Type2":2, "Supermarket Type3":3}, inplace=True)
        logging.info("Label encoding to outlet type completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    logging.info("Completed data transformation for Test Data")
    
    os.makedirs(os.path.dirname(transformation_config.processed_train_data_path), exist_ok=True)
    train_data.to_csv(transformation_config.processed_train_data_path, index=False, header=True)
    os.makedirs(os.path.dirname(transformation_config.processed_test_data_path), exist_ok=True)
    test_data.to_csv(transformation_config.processed_test_data_path, index=False, header=True)

    return (transformation_config.processed_train_data_path, transformation_config.processed_test_data_path)

def initiate_data_transform():
    train_data_path = read_train_data()
    test_data_path = read_test_data()
    return (train_data_path, test_data_path)


train_data_path, test_data_path = initiate_data_transform()
Data_Transformer(train_data_path, test_data_path)
