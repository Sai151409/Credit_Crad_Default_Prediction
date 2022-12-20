import logging
from datetime import datetime
import os
import pandas as pd

LOG_DIR = 'credit_card_logs'

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_NAME = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"

file_path = os.path.join(LOG_DIR, LOG_FILE_NAME)

logging.basicConfig(filename=file_path,
                    filemode='w',
                    level=logging.INFO,
                    format='[%(asctime)s]^;%(levelname)s^;%(lineno)d^;%(filename)s^;%(funcName)s()^;%(message)s')


def get_log_dataframe(filepath:str):
    """
    It returns the dataframe of credit card logs

    Args:
        filepath (str): log file path
    """
    data = []
    with open(filepath, 'r') as file:
        for line in file.readlines():
            data.append(line)
            
    log_data = pd.DataFrame(data, columns = ['Timestamp', 'log_level', 'error_lineno',
                                             'error_filename', 'error_funcname', 'message'])
    
    log_data['error_message'] = log_data['Timestamp'].astype(str) + ':$' + log_data['message']
    
    return log_data['error_message']
