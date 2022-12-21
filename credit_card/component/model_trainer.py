import os, sys
from credit_card.logger import logging
from credit_card.exception import CreditCardException
from credit_card.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from credit_card.entity.config_entity import ModelTrainerConfig
from credit_card.util.util import load_numpy_array,load_object,save_object
from credit_card.entity.model_factory import *


class CreditCardEstimatorModel:
    
    def __init__(self, preprocessing_obj, trained_model_obj):
        """
            preprocessing_obj : preprocessing_obj
            trained_model_obj : trained_model_obj
        """
        try:
            self.preprocessing_obj = preprocessing_obj
            self.trained_model_obj = trained_model_obj
        except Exception as e:
            raise CreditCardEstimatorModel(e, sys) from e
        
    def predict(self, X):
        """
        This function accepts raw input and then transformed raw input using preprocessing
        object which guarentees that inputs are in the same format as training data 
        At last it perform prediction on transformed features
        """
        try:
            transformed_object = self.preprocessing_obj.transform(X)
            return self.trained_model_obj.predict(transformed_object)
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def __repr__(self) -> str:
        return f"{type(self.trained_model_obj).__name__}()"
    
    def __str__(self) -> str:
        return f"{type(self.trained_model_obj).__name__}()"
    
    
class ModelTrainer:
    
    def __init__(self, model_trainer_config : ModelTrainerConfig,
                 data_transformation_artifact : DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def initiate_model_trainer(self):
        try:
            logging.info('Loading the transformed training dataset')
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_array = load_numpy_array(transformed_train_file_path)
            
            logging.info('Loading the transformed testing dataset')
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path
            test_array = load_numpy_array(transformed_test_file_path)
            
            logging.info('Splitting the training and testing input feature, output feature')
            X_train, y_train, X_test, y_test = train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1]
            
            logging.info('Extrtacting the model config file path')
            model_config_file_path = self.model_trainer_config.model_config_file_path
            
            logging.info('Extracting the model config file path')
            model_config_file_path = self.model_trainer_config.model_config_file_path
            
            logging.info('Initializing model factor by using model config yaml file')
            model_factory = ModelFactory(model_config_path=model_config_file_path)
            
            base_accuracy = self.model_trainer_config.base_accuracy
            logging.info(f"Expected base accuracy : {base_accuracy}")
            
            logging.info('Initiating operation model selection')
            best_model = model_factory.get_best_model(X=X_train, y=y_train, base_accuracy=base_accuracy)
            
            logging.info(f'Found Best Model on Training Data Set  : {best_model}')
            
            logging.info('Extracting trained model list')
            grid_searched_best_model_list:List[GridSearchBestModel]=model_factory.grid_serach_best_model_list
            
            model_list = [model.best_model for model in grid_searched_best_model_list]
            logging.info(f"Evaluation all trained model on training and testing dataset both")
            
            metric_info = evaluate_classification_model(
                model_list=model_list,
                X_train=X_train,
                y_train=y_train,
                X_test = X_test, 
                y_test = y_test,
                base_accuracy=base_accuracy
            )
            
            logging.info('Best found model on both training and testing dataset')
            
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            
            model_object = metric_info.model_object
            
            trained_model_file_path = self.model_trainer_config.model_trainer_file_path
            
            credit_card_model = CreditCardEstimatorModel(
                preprocessing_obj=preprocessing_obj,
                trained_model_obj=model_object
            )
            
            logging.info(f'Saving the model at path : {trained_model_file_path}')
            
            save_object(file_path=trained_model_file_path, obj=credit_card_model)
            
            model_trainer_artifact = ModelTrainerArtifact(
                is_trained=True,
                trained_model_file_path=trained_model_file_path,
                train_recall=metric_info.train_recall,
                test_recall=metric_info.test_recall,
                train_roc_score=metric_info.train_roc_score,
                test_roc_score=metric_info.test_roc_score,
                model_accuracy=metric_info.model_accuracy
            )
            
            logging.info(f'Model Trainer Artifact : {model_trainer_artifact}')
            
            return model_trainer_artifact
        
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def __del__(self):
        logging.info(f"{'>>' * 30} Model Trainer log is Completed{'<<' * 30}")
        