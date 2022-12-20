from credit_card.logger import logging
from credit_card.exception import CreditCardException
from credit_card.entity.config_entity import DataValidationConfig
from credit_card.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
import os, sys
import pandas as pd
from credit_card.util.util import read_yaml
import numpy as np
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import json


class DataValidation:
    
    def __init__(self, 
                 data_ingestion_artifact : DataIngestionArtifact, 
                 data_validation_config : DataValidationConfig):
        try:
            logging.info(f'{">>" * 30} Data Validation log : {">>" * 30}')
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def is_train_test_exists(self) -> bool:
        try:
            logging.info('Checking the train and test data sets are available')
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            is_train_file_exists = os.path.exists(train_file_path)
            is_test_file_exists = os.path.exists(test_file_path)
            
            if is_train_file_exists and is_test_file_exists:
                logging.info('Train and Test data sets are available')
                return True
            else:
                raise Exception('Train or Test data set is not available')
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def get_train_test_df(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_train_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            return train_df, test_train_df
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def validate_data_schema(self) -> bool:
        try:
            logging.info('Validating the train, test datasets and schema file')
            train_df, test_df = self.get_train_test_df()
            schema = read_yaml(self.data_validation_config.schema_file_path)
            validate  = False
            if list(train_df.columns) == list(test_df.columns):
                if len(train_df.columns) <= len(schema['columns']):
                    if list(train_df.columns) == list(test_df.columns) == list(schema['columns'].keys()):
                        for i in schema['domain_value'].keys():
                            train = set(train_df[i].unique()) 
                            test = set(test_df[i].unique()) 
                            sche = set(schema['domain_value'][i])
                            if train.issubset(sche) and test.issubset(sche):
                                continue
                            else:
                                unknown_domain_values = [j for j in train if j not in sche]
                                logging.info(f'Unknown domain values in {i} : [{unknown_domain_values}]')
                                logging.info('We are going to remove these from the dataset')
                                index = []
                                for j in unknown_domain_values:
                                    index.append(list(train_df[train_df[i] == j].index))
                                index = list(np.array(index).reshape(-1))
                                train_df.drop(index=index, inplace=True)
                                index = []
                                for j in unknown_domain_values:
                                    index.append(list(test_df[test_df[i] == j].index))
                                index = list(np.array(index).reshape(-1))
                                test_df.drop(index=index, inplace=True)
                                continue
                        validate =  True
                        logging.info("Validation of train_df, test_df and schema file successfully completed.")
                        return validate, train_df, test_df
                    else:
                        raise Exception("Datasets don't have necessary columns")       
                else:
                    logging.info('Unnecessary columns in the dataset. So we are removing those columns')
                    columns = [i for i in train_df.columns if i not in schema['columns'].keys()]
                    train_df.drop(columns=columns, inplace=True)
                    columns = [i for i in test_df.columns if i not in schema['columns'].keys()]
                    test_df.drop(columns=columns, inplace=True)
                    logging.info('Sucessfuly removed the unnecessary columns from the train and test dataset')
                    if list(train_df.columns) == list(test_df.columns) == list(schema['columns'].keys()):
                        for i in schema['domain_value'].keys():
                            train = set(train_df[i].unique()) 
                            test = set(test_df[i].unique()) 
                            sche = set(schema['domain_value'][i])
                            if train.issubset(sche) and test.issubset(sche):
                                continue
                            else:
                                unknown_domain_values = [j for j in train if j not in sche]
                                logging.info(f'Unknown domain values in {i} : [{unknown_domain_values}]')
                                logging.info('We are going to remove these from the dataset')
                                index = []
                                for j in unknown_domain_values:
                                    index.append(list(train_df[train_df[i] == j].index))
                                index = list(np.array(index).reshape(-1))
                                train_df.drop(index=index, inplace=True)
                                index = []
                                for j in unknown_domain_values:
                                    index.append(list(test_df[test_df[i] == j].index))
                                index = list(np.array(index).reshape(-1))
                                test_df.drop(index=index, inplace=True)
                                continue
                        validate = True
                        logging.info("Validation of train_df, test_df and schema file successfully completed.")
                        return validate, train_df, test_df
                    else:
                        raise Exception("Datasets don't have necessary columns")                 
            else:
                raise Exception("Train and Test Data Sets don't have common columns")    
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def validated_train_and_test_file_path(self):
        try:
            _, train_df, test_df = self.validate_data_schema()
            file_name = 'credit_card.csv'
            
            validated_train_file_path = os.path.join(
                self.data_validation_config.validated_train_dir,
                file_name
            )
            
            validated_test_file_path = os.path.join(
                self.data_validation_config.validated_test_dir,
                file_name
            )
            
            if train_df is not None:
                os.makedirs(self.data_validation_config.validated_train_dir, exist_ok=True)
                logging.info(f'Validated data set into {validated_train_file_path}')
                train_df.to_csv(validated_train_file_path, index = False)
                
            if test_df is not None:
                os.makedirs(self.data_validation_config.validated_test_dir, exist_ok=True)
                logging.info(f'Validated data set into {validated_test_file_path}')
                test_df.to_csv(validated_test_file_path, index = False)
                
            return validated_train_file_path, validated_test_file_path
                
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def save_data_drift_report(self, train_file_path, test_file_path):
        try:
            profile = Profile(sections=[DataDriftProfileSection()])
            
            train_df_file_path = train_file_path
            
            test_df_file_path = test_file_path
            
            train_df = pd.read_csv(train_df_file_path)
            
            test_df = pd.read_csv(test_df_file_path)
            
            profile.calculate(train_df, test_df)
            
            report  = json.loads(profile.json())
            
            report_file_path = self.data_validation_config.report_file_path
            
            report_dir = os.path.dirname(report_file_path)
            
            os.makedirs(report_dir, exist_ok=True)
            
            with open(report_file_path, 'w') as report_file:
                json.dump(report, report_file, indent=6)
            return report
            
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def save_data_drift_report_page(self, train_file_path, test_file_path):
        try:
            dashboard = Dashboard(tabs=[DataDriftTab()])
            
            train_df_file_path = train_file_path
            
            test_df_file_path = test_file_path
            
            train_df = pd.read_csv(train_df_file_path)
            
            test_df = pd.read_csv(test_df_file_path)
            
            dashboard.calculate(train_df, test_df)
            
            report_page_file_path = self.data_validation_config.report_page_file_path
            
            report_page_dir = os.path.dirname(report_page_file_path)
            
            os.makedirs(report_page_dir, exist_ok=True)
            
            dashboard.save(report_page_file_path)
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def is_data_drift_found(self, train_file_path, report) -> bool:
        try:
            data_drift = set()
            report = report
            train_file_path = train_file_path
            train_df = pd.read_csv(train_file_path)
            for i in train_df.columns:
                data_drift.add(report['data_drift']['data']['metrics'][i]['drift_detected'])
                
            if data_drift == {False}:
                logging.info(f'Detection of data drift : {data_drift}')
                logging.info(f'Data is not drifted')
                return True
            else:
                message = 'Data is drifted'
                raise Exception(message)
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    @staticmethod
    def outliers(df, i):
        try:
            q1, q2, q3 = np.quantile(df[i], [0.25, 0.5, 0.75])
            iqr = q3 - q1
            upper_whisker = q3 + (3 * iqr)
            lower_whisker = q1 - (3 * iqr)
            percentage = (len(df[(df[i] > upper_whisker) | (df[i] < lower_whisker)])/len(df)) * 100
            return f'Outliers percentage of {i} : {percentage}'
        except Exception as e:
            raise Exception(e, sys) from e
    
        
    def exploratory_data_analysis(self, train_file_path) :
        try:
            train_file_path = train_file_path
            train_df = pd.read_csv(train_file_path)
            schema = read_yaml(self.data_validation_config.schema_file_path)
            target_variable = schema['target_column'][0]
            df = train_df.drop(target_variable, axis = 1)
            range = df.shape
            columns = df.columns
            null_value_count = sum(df.isnull().sum())
            numerical_columns = []
            categorical_columns = []
            outliers = []
            threshold = 20
            for i in columns:
                l = len(df[i].unique())
                if l > threshold:
                    numerical_columns.append(i)
                else:
                    categorical_columns.append(i)
            
            unique_values_of_target_variable = train_df[target_variable].unique()
            percentage = []
            for i in unique_values_of_target_variable:
                percentage.append((len(train_df[train_df[target_variable] == i])/len(train_df)) * 100)
            for i in numerical_columns:
                outliers.append(DataValidation.outliers(df=df, i=i))
    
            logging.info(f'''Imbalanced dataset : {percentage}'
            Range : {range}\n
            columns : {columns}\n
            Null_Values : {null_value_count}\n
            numerical_columns : {numerical_columns}\n
            categorical_columns : {categorical_columns}
            outliers : {outliers}''')
        except Exception as e:
            raise CreditCardException(e, sys) from e
            
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            self.is_train_test_exists()
            validated_train_file_path, validated_test_file_path = self.validated_train_and_test_file_path()
            report = self.save_data_drift_report(train_file_path=validated_train_file_path,
                                        test_file_path=validated_test_file_path)
            self.save_data_drift_report_page(train_file_path=validated_train_file_path,
                                             test_file_path=validated_test_file_path)
            
            self.is_data_drift_found(train_file_path=validated_test_file_path,
                                     report=report)
            self.exploratory_data_analysis(train_file_path = validated_train_file_path)
            data_validation_artifact = DataValidationArtifact(
                validated_train_file_path=validated_train_file_path,
                validated_test_file_path=validated_test_file_path,
                schema_file_path=self.data_validation_config.schema_file_path,
                report_file_path=self.data_validation_config.report_file_path,
                report_page_file_path=self.data_validation_config.report_page_file_path,
                is_validated=True,
                message='Data Validation Performed Successfully.'
            )
            
            logging.info(f'Data Validation Artifact : {data_validation_artifact}')
            
            return data_validation_artifact
        except Exception as e:
            raise CreditCardException(e, sys) from e 
        
    def __del__(self):
        logging.info(f'{">>"*20}Data Validation log completed.{"<<"*20}\n\n')