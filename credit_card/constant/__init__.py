import os
from datetime import datetime

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"


CURRENT_TIME_STAMP = get_current_time_stamp()
CONFIG_FILE_NAME = 'config.yaml'
CONFIG_DIR = 'config'
ROOT_DIR = os.getcwd()
os.makedirs(CONFIG_DIR, exist_ok=True)

CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, CONFIG_FILE_NAME)

TRAINING_PIPELINE_CONFIG_KEY = 'training_pipeline_config'
TRAINING_PIPELINE_NAME_KEY = 'pipeline_name'
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = 'artifact_dir'

#Data Ingestion related variables

DATA_INGESTION_CONFIG_KEY = 'data_ingestion_config'
DATA_INGESTION_ARTIFACT_KEY = 'data_ingestion'
DATA_INGESTION_DOWNLOAD_URL_KEY = 'download_url'
DATA_INGESTION_RAW_DATA_DIR_KEY = 'raw_data_dir'
DATA_INGESTION_INGESTED_DIR_KEY = 'ingested_dir'
DATA_INGESTION_INGESTED_TRAIN_DIR_KEY = 'ingested_train_dir'
DATA_INGESTION_INGESTED_TEST_DIR_KEY = 'ingested_test_dir'

# Data Validation related variables

DATA_VALIDATION_CONFIG_KEY = 'data_validation_config'
DATA_VALIDATION_ARTIFACT_KEY = 'data_validation'
DATA_VALIDATION_VALIDATED_DATA_DIR_KEY = 'validated_data_dir'
DATA_VALIDATION_VALIDATED_TRAIN_DIR_KEY = 'validated_train_dir'
DATA_VALIDATION_VALIDATED_TEST_DIR_KEY = 'validated_test_dir'
DATA_VALIDATION_SCHEMA_DIR_KEY = 'schema_dir'
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = 'schema_file_name'
DATA_VALIDATION_REPORT_FILE_NAME_KEY = 'report_file_name'
DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY = 'report_page_file_name'

# Data Transformation related variables

DATA_TRANSFORMATION_CONFIG_KEY = 'data_transformation_config'
DATA_TRANSFORMATION_ARTIFACT_KEY = 'data_transformation'
DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY = 'transformed_dir'
DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR_KEY = 'transformed_train_dir'
DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR_KEY = 'transformed_test_dir'
DATA_TRANSFORMATION_PREPROCESSED_DIR_KEY = 'preprocessed_dir'
DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY = 'preprocessed_object_file_name'

# Model Trainer related variables

MODEL_TRAINER_CONFIG_KEY = 'model_trainer_config'
MODEL_TRAINER_ARTIFACT_KEY = 'model_trainer'
MODEL_TRAINER_TRAINED_MODEL_DIR_KEY = 'trained_model_dir'
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY = 'model_file_name' 
MODEL_TRAINER_BASE_ACCURACY_KEY = 'base_accuracy'
MODEL_TRAINER_CONFIG_FILE_DIR_KEY = 'model_config_dir'
MODEL_TRAINER_CONFIG_FILE_NAME_KEY = 'model_config_yaml_file'

# Model evaluation related variables

MODEL_EVALUATION_CONFIG_KEY = 'model_evaluation_config'
MODEL_EVALUATION_ARTIFACT_DIR = 'model_evaluation'
MODEL_EVALUATION_FILE_NAME_KEY = 'model_evaluation_file_name'

# Model Pusher related variables

MODEL_PUSHER_CONFIG_KEY = 'model_pusher_config'
MODEL_PUSHER_MODEL_EXPORT_DIR_KEY = 'model_export_dir'

# Experimented related variables

EXPERIMENT_DIR_KEY = "experiment"
EXPERIMENT_FILE_NAME = "experiment.csv"


BEST_MODEL_KEY = "best_model"
HISTORY_KEY = "history"
MODEL_PATH_KEY = "model_path"

DATA_SCHEMA_COLUMNS_KEY = "columns"
NUMERICAL_COLUMN_KEY = 'numerical_columns'
CATEGORICAL_COLUMN_KEY = 'categorical_columns'
TARGET_COLUMN_KEY = 'target_column'
COLUMN_PAY_0 = "PAY_0"
COLUMN_PAY_2 = "PAY_2"
COLUMN_PAY_3 = "PAY_3"
