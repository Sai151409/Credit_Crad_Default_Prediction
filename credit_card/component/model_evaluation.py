from credit_card.logger import logging
from credit_card.exception import CreditCardException
import os, sys
from credit_card.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, \
ModelTrainerArtifact, ModelEvaluationArtifact
from credit_card.entity.config_entity import ModelEvaluationConfig
from credit_card.util.util import *
from credit_card.entity.model_factory import *




class ModelEvaluation:
    
    def __init__(self, data_ingestion_artifact : DataIngestionArtifact,
                 data_validation_artifact : DataValidationArtifact,
                 model_trainer_artifact : ModelTrainerArtifact,
                 model_evaluation_config : ModelEvaluationConfig):
        try:
            logging.info(f"{'>>' * 30} Model Evaluation Log Started {'<<' * 30}")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_evaluation_config = model_evaluation_config
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def get_best_model(self):
        try:
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evalauation_file_path
            if not os.path.exists(model_evaluation_file_path):
                    write_yaml(file_path=model_evaluation_file_path, )
                    return model
            model_evaluation_content = read_yaml(model_evaluation_file_path)
            model_evaluation_content = dict() if model_evaluation_content is None else model_evaluation_content
            if BEST_MODEL_KEY not in model_evaluation_content:
                return model 
            model = load_object(model_evaluation_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            
            return model
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def update_evaluation_report(self, model_evaluation_artifact : ModelEvaluationArtifact):
        try:
            eval_file_path = self.model_evaluation_config.model_evalauation_file_path
            
            model_evaluation_content = read_yaml(eval_file_path)
            
            model_evaluation_content = dict() if model_evaluation_content is None else model_evaluation_content
            
            previous_model = None
            
            if BEST_MODEL_KEY in model_evaluation_content:
                previous_model = model_evaluation_content[BEST_MODEL_KEY]
                
            logging.info(f'Previous Best Model is  : {previous_model}')
            
            eval_result = {
                BEST_MODEL_KEY : 
                    {
                        MODEL_PATH_KEY : model_evaluation_artifact.evaluated_model_path
                    }
            }
            
            if previous_model is not None:
                model_history = {self.model_evaluation_config.time_stamp : previous_model}
                if HISTORY_KEY not in model_evaluation_content:
                    history = {HISTORY_KEY : model_history}
                    eval_result.update(history)
                else:
                    model_evaluation_content[HISTORY_KEY].update(model_history)
            model_evaluation_content.update(eval_result)
            logging.info(f'Updated eval result : {model_evaluation_content}')
            write_yaml(eval_file_path, model_evaluation_content)
            
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            
            trained_model_obj = load_object(trained_model_file_path)
            
            train_file_path = self.data_validation_artifact.validated_train_file_path
            test_file_path = self.data_validation_artifact.validated_test_file_path
            
            schema_file_path = self.data_validation_artifact.schema_file_path
            
            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)
            
            schema = read_yaml(file_path=schema_file_path)
            
            target_column = schema[TARGET_COLUMN_KEY][0]
            
            #target_column
            logging.info("Converting the target column into numpy array")
            train_target_array = np.array(train_df[target_column])
            test_target_array = np.array(test_df[target_column])
            logging.info("Conversion completed target column into numpy array")
            
            logging.info('Dropping the target column from the train and test dataframe')
            train_df.drop(columns=[target_column], axis = 1, inplace = True)
            test_df.drop(columns=[target_column], axis=1, inplace=True)
            
            feature_selection = ['LIMIT_BAL', 'PAY_0','PAY_2','PAY_3','BILL_AMT1','BILL_AMT2',
                       'BILL_AMT3','PAY_AMT1','PAY_AMT2','PAY_AMT3']
            
            train_df = train_df[feature_selection]
            test_df = test_df[feature_selection]
            
            model = self.get_best_model()
            
            if model is None:
                logging.info('Not Found any existing model.Hence the trained model')
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=True,
                    evaluated_model_path=trained_model_file_path
                )
                self.update_evaluation_report(model_evaluation_artifact=model_evaluation_artifact)
                logging.info(f"Model Accepted. Model eval artifact : {model_evaluation_artifact} created")
                return model_evaluation_artifact
            
            model_list = [model, trained_model_obj]
            
            metric_info_artifact = evaluate_classification_model(
                model_list=model_list,
                X_train = train_df,
                X_test = test_df,
                y_train=train_target_array,
                y_test=test_target_array,
                base_accuracy=self.model_trainer_artifact.model_accuracy
            )
            
            logging.info(f"Model Evaluation Completed. Model Metric Artifact : {metric_info_artifact}")
            
            if metric_info_artifact is None : 
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=False,
                    evaluated_model_path=trained_model_file_path)
                logging.info(model_evaluation_artifact)
                return model_evaluation_artifact
            
            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=True,
                    evaluated_model_path=trained_model_file_path
                )
                self.update_evaluation_report(model_evaluation_artifact=model_evaluation_artifact)
                logging.info(f'Model Accepted. Model Evaluated Artifact : {model_evaluation_artifact}')
            else:
                logging.info('Trained Model is no better than existing model hence not acceopting the trained model')
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=False, evaluated_model_path=trained_model_file_path)
            return model_evaluation_artifact
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def __del__(self):
        return f"{'>>' * 30} Model Evaluation log is completed.{'<<' * 30}"

            