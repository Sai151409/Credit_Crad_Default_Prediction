from credit_card.exception import CreditCardException
from credit_card.logger import logging
import pandas as pd
from credit_card.constant import *
import numpy as np
import yaml
import dill
import os, sys

def write_yaml(file_path : str, data : dict = None) :
    """
    Create a yaml file
    file_path : str
    data : dict
    """ 
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file=file_path, mode='w') as file_obj :
            if data is not None :
                yaml.dump(data, file_obj)
    except Exception as e:
        raise CreditCardException(e, sys) from e
    
    
def read_yaml(file_path:str) -> dict:
    """
    Read yaml file then returns the dict
    file_path : str
    """
    try:
        with open(file_path, 'rb') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise CreditCardException(e, sys) from e
    
    
def load_data(file_path : str, schema_file_path : str) -> pd.DataFrame:
    try:
        dataset = read_yaml(schema_file_path)
        schema = dataset[DATA_SCHEMA_COLUMNS_KEY]
        dataframe = pd.read_csv(file_path)
        
        error_message = ""
        
        for column in dataframe.columns:
            if column in (schema.keys()):
                dataframe[column].astype(schema[column])
            else:
                error_message = f"{error_message} \nColumn : {column} is not in the schema"
        if len(error_message) > 0:
                raise Exception(error_message)
        
        return dataframe
    except Exception as e:
        raise CreditCardException(e, sys) from e
    
    
def save_numpy_array(file_path : str, array : np.array):
    """
    Save numpy array data into file
    file_path : str location of file to save
    array : np.array data to solve
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CreditCardException(e, sys) from e
    
    
def load_numpy_array(file_path : str) -> np.array:
    """
    load numpy array data from file
    file_path : str location of file to load
    return : np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file=file_obj)
    except Exception as e:
        raise CreditCardException(e, sys) from e
    
    
def save_object(file_path : str, obj):
    """
    file_path : str
    obj : Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CreditCardException(e, sys) from e


def load_object(file_path : str):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CreditCardException(e, sys) from e
