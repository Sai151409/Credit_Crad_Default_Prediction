from credit_card.logger import logging
import os, sys
from credit_card.exception import CreditCardException
from credit_card.entity.config_entity import DataTransformationConfig
from credit_card.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, \
    DataTransformationArtifact
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from credit_card.constant import *
from credit_card.util.util import *


class DataTransformation:
    
    def __init__(self, 
                 data_ingestion_artifact  : DataIngestionArtifact,
                 data_validation_artifact : DataValidationArtifact,
                 data_transfomation_config : DataTransformationConfig):
        try:
            logging.info(f'{">>" * 30} Data Transformation log Started {"<<" * 30}')
            
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transfomation_config
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def get_data_transformed_object(self) -> ColumnTransformer:
        try:
            columns = ['LIMIT_BAL', 'BILL_AMT1','BILL_AMT2',
                       'BILL_AMT3','PAY_AMT1','PAY_AMT2','PAY_AMT3']
            
            num_pipeline = Pipeline(steps=[
                ('scaler', PowerTransformer())
            ])
            
            preprocessing = ColumnTransformer([
                ('num_pipeline', num_pipeline, columns)
            ], remainder='passthrough')
            
            return preprocessing
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def initate_data_transformation(self):
        try:
            columns = ['LIMIT_BAL', 'PAY_0','PAY_2','PAY_3','BILL_AMT1','BILL_AMT2',
                       'BILL_AMT3','PAY_AMT1','PAY_AMT2','PAY_AMT3']
            
            logging.info('Obtaining Prerocessing Object')
            preprocessing_obj = self.get_data_transformed_object()
            
            logging.info('Obtaining train and test file path')
            train_file_path = self.data_validation_artifact.validated_train_file_path
            test_file_path = self.data_validation_artifact.validated_test_file_path
            
            schema_file_path = self.data_validation_artifact.schema_file_path
            
            logging.info('Loading the train and test datasets')
            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)
            
            schema = read_yaml(schema_file_path)
            
            target_column = schema[TARGET_COLUMN_KEY][0]
            
            logging.info('Splitting the input and output features from the train dataset')
            input_feature_train_df = train_df.drop(columns = [target_column], axis = 1)
            logging.info('Here we are going to make changes in features like education, marital status, payments')
            logging.info('Because it has some unnecessary negative values in the feature we make them zeros')
            condition = (input_feature_train_df['EDUCATION'] == 5) | (input_feature_train_df['EDUCATION'] == 6) | (input_feature_train_df['EDUCATION'] == 0)
            input_feature_train_df.loc[condition, 'EDUCATION'] = 3
            input_feature_train_df.loc[(input_feature_train_df['MARRIAGE'] == 0), 'MARRIAGE'] = 3
            for i in ['PAY_0', 'PAY_2', "PAY_3", 'PAY_4', 'PAY_5', 'PAY_6']:
                input_feature_train_df.loc[(input_feature_train_df[i] < 0), i] = 0
            input_feature_train_df = input_feature_train_df[columns]
            target_feature_train_df = train_df[target_column]
            
            logging.info('Splitting the input and output feature from the test dataset')
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            condition = (input_feature_test_df['EDUCATION'] == 5) | (input_feature_test_df['EDUCATION'] == 6) | (input_feature_test_df['EDUCATION'] == 0)
            input_feature_test_df.loc[condition, 'EDUCATION'] = 3
            input_feature_test_df.loc[(input_feature_test_df['MARRIAGE'] == 0), 'MARRIAGE'] = 3
            for i in ['PAY_0', 'PAY_2', "PAY_3", 'PAY_4', 'PAY_5', 'PAY_6']:
                input_feature_test_df.loc[(input_feature_test_df[i] < 0), i] = 0
            input_feature_test_df = input_feature_test_df[columns]
            target_feature_test_df = test_df[target_column]
            
            logging.info('Preprocessing Object on training dataframe and test dataframe')
            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_array, target_feature_train_df]
            test_arr = np.c_[input_feature_test_array, target_feature_test_df]
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
            
            train_file_name = os.path.basename(train_file_path).replace(".csv", ".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv", ".npz")
            
            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)
            
            logging.info('Saving transformed training and testing array.')
            save_numpy_array(file_path=transformed_train_file_path, array=train_arr)
            save_numpy_array(file_path=transformed_test_file_path, array=test_arr)
            
            logging.info('Saving the preproceesing object')
            preprocessed_object_file_path = self.data_transformation_config.preprocessed_object_file_path
            save_object(file_path=preprocessed_object_file_path,
                        obj=preprocessing_obj)
            
            data_transformation_artifact = DataTransformationArtifact(
                is_transformed=True,
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                preprocessed_object_file_path=preprocessed_object_file_path,
                message='Data Tranformation Performed Sucessfully'
            )
            
            logging.info(f'Data Transformation Artifact : {data_transformation_artifact}')
            
            return data_transformation_artifact
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def __del__(self):
        logging.info(f'{">>"*20}Data Transformation log completed.{"<<"*20}\n\n')     
            
            
        