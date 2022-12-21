from collections import namedtuple
from credit_card.logger import logging
from credit_card.exception import CreditCardException
import os, sys, yaml
import numpy as np
import pandas as pd
import importlib
from typing import List
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, roc_auc_score



GRID_SEARCH_KEY = 'grid_search'
CLASS_KEY = 'class'
MODULE_KEY = 'module'
MODEL_SELECTION_KEY = 'model_selection'
PARAMS_KEY = 'params'
SEARCH_PARAM_GRID_KEY = 'search_param_grid'

InitializedModelDetail = namedtuple('IntializedModelDetail', [
    'model_serial_number', 'model', 'param_grid_search', 'model_name'
])


GridSearchBestModel = namedtuple('GridSearchBestModel',[
    'model_serial_number', 'model', 'best_model', 'best_parameters',  'best_score'])


BestModel = namedtuple('BestModel', [
    'model_serial_number', 'model', 'best_model', 'best_parameters', 'best_score'
])


MetricInfoArtifact = namedtuple('MetricInfoArtifact', [
    'model_name', 'model_object', 'train_roc_score', 'test_roc_score',
    'train_recall', 'test_recall', 'model_accuracy', 'index_number'
])


def evaluate_classification_model(model_list : list,
                              X_train : np.ndarray,
                              y_train : np.ndarray,
                              X_test : np.ndarray,
                              y_test : np.ndarray,
                              base_accuracy : float = 0.6) -> MetricInfoArtifact:
    """
    Description:
    This function compare multiple linear regerssion model return the best model

    Params:
        model_list (list): list of models
        X_train (np.ndarray): Training dataset input feature
        y_train (np.ndarray): Training dataset target feature
        X_test (np.ndarray): Testing dataset input feature
        y_test (np.ndarray): Testing dataset target feature
        base_accuracy (float, optional): _description_. Defaults to 0.6.

    Returns:
    It returns a namedtuple
        MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"]) 
    """
    try:
        index = 0 
        metric_info_artifact = None
        for model in model_list:
            model_name = str(model) # getting model name based on model object
            logging.info(f'{">>" * 30} Started evaluating model : [{type(model).__name__}]{"<<" * 30}')
            
            #Getting prediction for training and testing  dataset
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_recall = recall_score(y_train_pred, y_train)
            test_recall = recall_score(y_test_pred, y_test)
            
            train_roc_score = roc_auc_score(y_train, y_train_pred)
            test_roc_score = roc_auc_score(y_test, y_test_pred)
            
            model_accuracy = (2 * train_recall * test_recall) / (train_recall + test_recall)
            diff_test_train_acc = abs(train_recall - test_recall)
            
            # logging all important metric
            logging.info(f"{'>>' * 30} Score {'<<' * 30}") 
            logging.info(f"Train Recall Score \t\t Test Recall Score \t\t Average Score")
            logging.info(f"{train_recall} \t\t {test_recall} \t\t {model_accuracy}")
            
            logging.info(f"{'>>' * 30} Loss {'<<' * 30}")
            logging.info(f"Diff test and train : {diff_test_train_acc}")
            logging.info(f"Train roc auc score : {train_roc_score}")
            logging.info(f"Testing roc and score : {test_roc_score}")
            
            #if model accuracy is greater than base accuracy and train and test score is within certain thershold
            #we will accept that model as accepted model
            
            if model_accuracy >= base_accuracy and diff_test_train_acc < 0.05:
                base_accuracy = model_accuracy
                metric_info_artifact = MetricInfoArtifact(
                    model_name=model_name,
                    model_object=model,
                    train_recall=train_recall,
                    test_recall=test_recall,
                    train_roc_score=train_roc_score,
                    test_roc_score=test_roc_score,
                    model_accuracy=model_accuracy,
                    index_number=index    
                )
                logging.info(f"Acceptable model found : {metric_info_artifact}")
            index += 1
            
        if metric_info_artifact is None:
            logging.info(f"No model is higher accuracy than the base accuracy")
        return metric_info_artifact
    except Exception as e:
        raise CreditCardException(e, sys) from e


def get_sample_yaml_file(export_dir):
    try: 
        model_config = {
            "GRID_SEARCH_KEY" :
                {
                    CLASS_KEY : "GridSearchCV",
                    MODULE_KEY : "sklearn.model_selection",
                    PARAMS_KEY : 
                        {
                        "cv" : 3,
                        "verbose" : 1
                        }
                },
            "MODEL_SELECTION" : 
                {
                    "module 0" :
                        {
                            MODULE_KEY : "module_of_model",
                            CLASS_KEY : "ModelClassName",
                            PARAMS_KEY : 
                                {
                                    "param_name1" : "value_1",
                                    "param_name2" : "value_2"
                                },
                            SEARCH_PARAM_GRID_KEY :
                                {
                                    "param_name": ['param_value_1', 'param_value_2']
                                }
                        },
                        
                }
                
        }
        
        os.makedirs(export_dir, exist_ok=True)
        export_file_path = os.path.join(export_dir, "model.yaml")
        with open(export_file_path, "w") as file:
            yaml.dump(model_config, file)
        return export_file_path
    except Exception as e:
        raise CreditCardException(e, sys) from e
    
    

class ModelFactory:
    
    def __init__(self, model_config_path = r'C:\Users\Asus\ML_Projects\Machine_Learning_Project_2\config\model.yaml'):
        try:
            self.config = ModelFactory.read_parmas(file=model_config_path)
            self.model_config_info = self.config[GRID_SEARCH_KEY]
            self.grid_search_module = self.model_config_info[MODULE_KEY]
            self.grid_search_class = self.model_config_info[CLASS_KEY]
            self.grid_search_property_data = self.model_config_info[PARAMS_KEY]
            self.models_initialization_config = self.config[MODEL_SELECTION_KEY]
            
            self.initialized_model_list = None
            self.grid_serach_best_model_list = None
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    @staticmethod
    def update_property(instance_ref : object, property_data : dict):
        try:
            if not isinstance(property_data, dict):
                raise Exception('Property data parameter required to dictionary')
            print(property_data)
            for key, value in property_data.items():
                setattr(instance_ref, key, value) 
            return instance_ref
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    @staticmethod
    def class_for_name(module_name, class_name):
        try:
            #load the module, will raise ImportError if module cannot be loaded 
            module =  importlib.import_module(module_name)
            # get the class, will raise AttributeError if class cannot be found
            logging.info(f'Exceuting the command  : import {class_name}  form the module {module_name}')
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            raise  CreditCardException(e, sys) from e
        
    @staticmethod
    def read_parmas(file:str):
        try:
            with open(file, 'rb') as model_file:
                config: dict = yaml.safe_load(model_file)
            return config
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def execute_grid_search_operations(self, initialized_model : InitializedModelDetail,
                                                                 input_feature,
                                                                 output_feature):
        
        """
        excute_grid_search_operation(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return GridSearchOperation object
        """                                                          
        try:
            grid_search_cv_ref = ModelFactory.class_for_name(module_name=self.grid_search_module,
                                                             class_name=self.grid_search_class)
            cv = StratifiedKFold(n_splits=5)
            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model,
                                                param_grid =initialized_model.param_grid_search,
                                                cv=cv)
            grid_search_cv = ModelFactory.update_property(instance_ref=grid_search_cv,
                                                          property_data=self.grid_search_property_data)
            message = f'{">>" * 30} Training {type(initialized_model.model).__name__} started {"<<" * 30}'
            logging.info(message)
            grid_search_cv.fit(input_feature, output_feature)
            message = f'{">>" * 30} Training {type(initialized_model.model).__name__} completed {"<<" * 30}'
            logging.info(message)
            
            grid_search_best_model = GridSearchBestModel(
                model_serial_number=initialized_model.model_serial_number,
                model = initialized_model.model,
                best_model=grid_search_cv.best_estimator_,
                best_parameters=grid_search_cv.best_params_,
                best_score=grid_search_cv.best_score_
            )
            
            return grid_search_best_model
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
        
    def get_initialized_model_list(self) -> List[InitializedModelDetail]:
        """
        This Function returns the model list
        return List[ModelDetails]
        """
        try:
            initialized_model_list=[]
            for model_serial_number in self.models_initialization_config.keys():
                model_initialization_config = self.models_initialization_config[model_serial_number]
                model_obj_ref = ModelFactory.class_for_name(class_name=model_initialization_config[CLASS_KEY],
                                                            module_name=model_initialization_config[MODULE_KEY])
                model = model_obj_ref()
                
                if PARAMS_KEY in model_initialization_config:
                    model_object_property = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                    model = ModelFactory.update_property(instance_ref=model,
                                                         property_data=model_object_property)
                param_grid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"
                
                initialization_model = InitializedModelDetail(
                    model_serial_number=model_serial_number,
                    model = model,
                    model_name=model_name,
                    param_grid_search=param_grid_search
                )
                initialized_model_list.append(initialization_model)
            self.initialized_model_list = initialized_model_list
            
            return self.initialized_model_list 
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def initiate_best_parameter_search_for_initialized_model(self, initialized_model : InitializedModelDetail,
                                                             input_feature,
                                                             output_feature):
        """
        initiate_best_model_parameter_search(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return a GridSearchOperation
        """
        try:
            return self.execute_grid_search_operations(initialized_model=initialized_model,
                                                input_feature=input_feature,
                                                output_feature=output_feature)
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def initiate_best_parameter_search_for_initialized_models(self, initialized_model_list : List[InitializedModelDetail],
                                                              input_feature,
                                                              output_feature):
        try:
            grid_search_best_model_list = []
            for initialized_model in initialized_model_list:
                grid_search_best_model = self.initiate_best_parameter_search_for_initialized_model(
                    initialized_model=initialized_model,
                    input_feature=input_feature, 
                    output_feature=output_feature
                )
                grid_search_best_model_list.append(grid_search_best_model)
            self.grid_serach_best_model_list = grid_search_best_model_list
            
            return self.grid_serach_best_model_list
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    @staticmethod
    def get_model_details(model_details : List[InitializedModelDetail],
                          model_serial_number : str):
        try:
            for model_data in model_details:
                if model_data.model_serial_number == model_serial_number:
                    return model_data
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    @staticmethod
    def get_best_model_for_grid_searched_best_model_list(grid_searched_best_model_list : List[GridSearchBestModel],
                                                         base_accuracy = 0.6) -> BestModel:
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.info(f'Acceptable Model Found : {grid_searched_best_model}')
                    base_accuracy = grid_searched_best_model.best_score
                    
                    best_model = grid_searched_best_model
            if not best_model:
                raise Exception(f"None of the model has base accuracy : {base_accuracy}")
            logging.info(f'Best Model : {best_model}')
            
            return best_model
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def get_best_model(self, X, y, base_accuracy) -> BestModel:
        try:
            logging.info('Started Initializing model for config file')
            initialized_model_list = self.get_initialized_model_list()
            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y
            )
            
            return ModelFactory.get_best_model_for_grid_searched_best_model_list(
                grid_searched_best_model_list=grid_searched_best_model_list,
                base_accuracy=base_accuracy
            )
        except Exception as e:
            raise CreditCardException(e, sys) from e
            