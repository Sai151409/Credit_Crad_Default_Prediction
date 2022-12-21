from  credit_card.logger import logging
from credit_card.exception import CreditCardException
import os, sys
from credit_card.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from credit_card.entity.config_entity import ModelPusherConfig
import shutil


class ModelPusher:
    
    def __init__(self, model_evaluation_artifact : ModelEvaluationArtifact,
                 model_pusher_config : ModelPusherConfig):
        try:
            logging.info(f'{">>" * 30} Model Pusher Log is Started. {"<<" * 30}')
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def export_module(self)-> ModelPusherArtifact:
        try:
            evaluated_model_path = self.model_evaluation_artifact.evaluated_model_path
            export_dir = self.model_pusher_config.export_dir_path
            file_name = os.path.basename(evaluated_model_path)
            export_model_file_path = os.path.join(export_dir, file_name)
            logging.info(f'Exporting model file : [{export_model_file_path}]')
            os.makedirs(export_dir, exist_ok=True)
            shutil.copy(src=evaluated_model_path, dst=export_model_file_path)
            logging.info(f"Trained model : [{evaluated_model_path}] is copied in export dir : [{export_model_file_path}]")
            model_pusher_artifact = ModelPusherArtifact(
                is_model_pusher=True,
                export_model_file_path=export_model_file_path
            )
            logging.info(f'Model Pusher Artifact : {model_pusher_artifact}')
            return model_pusher_artifact
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def initiate_model_puser(self):
        try:
            return self.export_module()
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def __del__(self):
        logging.info(f'{">>" * 30} Model Pusher Log is Completed. {"<<" * 30}')        
            