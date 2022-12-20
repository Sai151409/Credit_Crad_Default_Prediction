from credit_card.logger import logging
from credit_card.exception import CreditCardException
from credit_card.config.configuration import Configuration
from credit_card.component.data_ingestion import DataIngestion
import sys
from credit_card.pipeline.pipeline import Pipeline



def main():
    try:
        pipeline = Pipeline(config=Configuration())
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f'{e}')
        print(e)
if __name__ == '__main__':
    main()