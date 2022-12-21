from collections import namedtuple


DataIngestionConfig = namedtuple('DataIngestionConfig', [
    'download_url', 'raw_data_dir', 'ingested_train_dir', 'ingested_test_dir'
])

DataValidationConfig = namedtuple('DataValidation', [
    'validated_train_dir', 'validated_test_dir','schema_file_path', 
    'report_file_path', 'report_page_file_path'
])

DataTransformationConfig = namedtuple('DataTransformationConfig', [
    'transformed_train_dir', 'transformed_test_dir', 
    'preprocessed_object_file_path',
])

ModelTrainerConfig = namedtuple('ModelTrainerConfig', [
    'model_trainer_file_path', 'base_accuracy', 'model_config_file_path'
])

ModelEvaluationConfig = namedtuple('ModelEvaluationConfig', [
    'model_evalauation_file_path', 'time_stamp'
])

ModelPusherConfig = namedtuple('ModelPusherConfig', [
    'export_dir_path'
])

TrainingPipelineConfig = namedtuple('TrainingPipelineArtifact', ['artifact_dir'])