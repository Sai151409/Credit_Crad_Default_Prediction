from credit_card.logger import logging
from credit_card.exception import CreditCardException
import os, sys
from collections import namedtuple
from credit_card.constant import *
from credit_card.config.configuration import Configuration
from credit_card.component.data_ingestion import DataIngestion
from credit_card.component.data_validation import DataValidation
from credit_card.component.data_transformation import DataTransformation
from credit_card.component.model_trainer import ModelTrainer
from credit_card.component.model_evaluation import ModelEvaluation
from credit_card.component.model_pusher import ModelPusher
from credit_card.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, \
    DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact
from threading import Thread
import uuid
import datetime
import pandas as pd
    
    
Experiment =  namedtuple('Experiment', 
                         ['experiment_id', 'initialization_timestamp', 'artifact_timestamp', 'running_status',
                          'start_time', 'stop_time', 'execution_time', 'message', 'experiment_file_path',
                          'accuracy', 'is_model_accepted'])
    

class Pipeline(Thread):
    
    experiment : Experiment = Experiment(*([None] * 11))
    experiment_file_path = None
    
    
    
    def __init__(self, config = Configuration()) -> None:
        try:
            os.makedirs(config.training_pipeline_artifact.artifact_dir, exist_ok=True)
            Pipeline.experiment_file_path = os.path.join(
                config.training_pipeline_artifact.artifact_dir, EXPERIMENT_DIR_KEY, EXPERIMENT_FILE_NAME
            )
            self.config = config
            super().__init__(daemon=False, name='pipeline')
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(
                data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CreditCardException(e, sys) from e
    def start_data_validation(self, 
                              data_ingestion_artifact : DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.config.get_data_validation_config()
            )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def start_data_transfomration(self, 
                                  data_ingestion_artifact : DataIngestionArtifact,
                                  data_validation_artifact : DataValidationArtifact):
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                data_transfomation_config=self.config.get_data_transformation_config()
            )
            return data_transformation.initate_data_transformation()
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def start_model_trainer(self,
                            data_transformation_artifact : DataTransformationArtifact):
        try:
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.config.get_model_trainer_config()
            )
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def start_model_evaluation(self, data_inegstion_artifact : DataIngestionArtifact,
                               data_validation_artifact : DataValidationArtifact,
                               model_trainer_artifact : ModelTrainerArtifact):
        try:
            model_evaluation = ModelEvaluation(
                data_ingestion_artifact=data_inegstion_artifact,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact,
                model_evaluation_config=self.config.get_model_evalutaion_config()
            )
            return model_evaluation.initiate_model_evaluation()
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def start_model_pusher(self, model_evaluation_artifact : ModelEvaluationArtifact):
        try:
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.config.get_model_pusher_config()
            )
            return model_pusher.initiate_model_puser()
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def run_pipeline(self):
        try:
            if Pipeline.experiment.running_status:
                logging.info('Pipeline is already running')
                return Pipeline.experiment
            logging.info('Pipeline Starting') 
            
            experiment_id = str(uuid.uuid4())
            
            Pipeline.experiment = Experiment(
                experiment_id=experiment_id,
                initialization_timestamp=self.config.time_stamp,
                artifact_timestamp=self.config.time_stamp,
                running_status=True,
                start_time=datetime.datetime.now(),
                stop_time=None,
                execution_time=None,
                experiment_file_path=Pipeline.experiment_file_path,
                is_model_accepted=None,
                message='Pipeline has been started', 
                accuracy = None 
            )
            
            logging.info(f'Pipeline Experiment : {Pipeline.experiment}')
            
            self.save_experiment()
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transfomration(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(
                data_inegstion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact)
            if model_evaluation_artifact.is_model_accepted:
                model_pusher_artifact = self.start_model_pusher(
                    model_evaluation_artifact=model_evaluation_artifact)
                logging.info(f'Model Pusher Artifact : {model_pusher_artifact}')
            else:
                logging.info('Trained Model is rejected')
            logging.info('Pipeline is completed')
            
            stop_time = datetime.datetime.now()
            
            Pipeline.experiment = Experiment(
                experiment_id=Pipeline.experiment.experiment_id,
                initialization_timestamp=self.config.time_stamp,
                artifact_timestamp=self.config.time_stamp,
                running_status=False,
                start_time=Pipeline.experiment.start_time,
                stop_time=stop_time,
                execution_time=stop_time - Pipeline.experiment.start_time,
                message='Pipeline has been Completed',
                experiment_file_path=Pipeline.experiment_file_path,
                accuracy=model_trainer_artifact.model_accuracy,
                is_model_accepted=model_evaluation_artifact.is_model_accepted,
            )
            logging.info(f'Pipeline Experiment : {Pipeline.experiment}')
            self.save_experiment()
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def run(self):
        try:
            self.run_pipeline()
        except Exception as e:
            raise e
        
    def save_experiment(self):
        try:
            if Pipeline.experiment.experiment_id is not None:
                experiment = Pipeline.experiment
                experiment_dict = experiment._asdict()
                experiment_dict : dict = {key : [value] for key, value in experiment_dict.items()}
                experiment_dict.update({
                    'created_time_stamp' : [datetime.datetime.now()],
                    'experiment_file_path' : [os.path.basename(Pipeline.experiment_file_path)]
                })
                experiment_dataframe = pd.DataFrame(experiment_dict)
                os.makedirs(os.path.dirname(Pipeline.experiment_file_path), exist_ok=True)
                if os.path.exists(Pipeline.experiment.experiment_file_path):
                    experiment_dataframe.to_csv(Pipeline.experiment.experiment_file_path, mode = 'a', index= False, header=False)
                else:
                    experiment_dataframe.to_csv(Pipeline.experiment.experiment_file_path, mode ='w', index = False, header=True)
            else:
                logging.info('First Start Experiment')
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    @classmethod
    def get_experiment_status(cls, limit : int = 5):
        try:
            if os.path.exists(Pipeline.experiment_file_path):
                experiment_dataframe = pd.read_csv(Pipeline.experiment.experiment_file_path)
                limit = -1 * int(limit)
                return experiment_dataframe[limit:].drop(columns=['experiment_file_path',
                                                                  'initialization_timestamp'], axis = 1)
            else:
                return pd.DataFrame()
        except Exception as e:
            raise CreditCardException(e, sys) from e