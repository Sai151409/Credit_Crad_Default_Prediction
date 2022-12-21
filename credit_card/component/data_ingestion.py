from credit_card.logger import logging
from credit_card.exception import CreditCardException
import os, sys
from credit_card.entity.config_entity import DataIngestionConfig
from six.moves import urllib
from credit_card.entity.artifact_entity import DataIngestionArtifact
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


class DataIngestion:
    
    def __init__(self, data_ingestion_config : DataIngestionConfig):
        try:
            logging.info(f'{"==" * 30} Data Ingestion log started {"==" * 30}')
            self.data_ingestion_config = data_ingestion_config  
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def download_credit_card_dataset(self):
        try:
            download_url = self.data_ingestion_config.download_url
            
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            
            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)
            
            os.makedirs(raw_data_dir, exist_ok=True)
            
            file_path = os.path.join(raw_data_dir, 'credit_card.csv')
            
            urllib.request.urlretrieve(download_url, file_path)
            
            logging.info(f'Downloading the dataset form {download_url} to {raw_data_dir}')
            
            file_name = os.listdir(raw_data_dir)[0]
            
            raw_data_file_path = os.path.join(raw_data_dir, file_name)
            
            return raw_data_file_path
            
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def split_data_as_train_test(self, raw_data_file_path  :str) -> DataIngestionArtifact:
        try:
            file_name = os.path.basename(raw_data_file_path)
            
            credit_card_dataset_file_path = raw_data_file_path
            
            credit_card_dataframe = pd.read_excel(credit_card_dataset_file_path, header = 1)
            
            logging.info('Splitting the dataset into train and test')
            
            strat_train_set = None
            strat_test_set = None
            
            split = StratifiedShuffleSplit(n_splits=1, test_size = 0.3, random_state=42)
            
            for train_index, test_index in split.split(credit_card_dataframe, credit_card_dataframe['default payment next month']):
                strat_train_set = credit_card_dataframe.loc[train_index]
                strat_test_set = credit_card_dataframe.loc[test_index]
                
            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir, file_name)
            
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir, file_name)
            
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok=True)
                logging.info(f"Exporting the data set into :[{train_file_path}]")
                strat_train_set.to_csv(train_file_path, index = False)
                
            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)
                logging.info(f"Exporting the data set into :[{test_file_path}]")
                strat_test_set.to_csv(test_file_path, index = False)
            
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=train_file_path,
                test_file_path=test_file_path,
                is_ingested=True,
                message="Data Ingestion Completed Successfully"
            )
            
            logging.info(f'Data ingestion artifact : {data_ingestion_artifact}')
            
            return data_ingestion_artifact  
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            raw_data_file_path = self.download_credit_card_dataset()
            
            return self.split_data_as_train_test(
                raw_data_file_path=raw_data_file_path
            )
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def __del__(self):
        logging.info(f'{">>" * 20}Data Ingestion log completed.{">>"*20}')

            