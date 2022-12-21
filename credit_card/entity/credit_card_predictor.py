from credit_card.logger import logging
from credit_card.exception import CreditCardException
import os, sys
import pandas as pd
from credit_card.util.util import *
from credit_card.config.configuration import Configuration
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, recall_score
from credit_card.constant import *
from credit_card.util.util import *
import numpy as np


class CreditCardData:
    
    def __init__(self, 
                 LIMIT_BAL : int,
                 SEX : int,
                EDUCATION : int,
                MARRIAGE : int,
                AGE : int,
                PAY_0 : int,
                PAY_2  : int,
                PAY_3 : int,
                PAY_4 : int,
                PAY_5 : int,
                PAY_6 : int,
                BILL_AMT1 : int,
                BILL_AMT2 : int,
                BILL_AMT3 : int,
                BILL_AMT4 : int,
                BILL_AMT5 : int,
                BILL_AMT6 : int,
                PAY_AMT1 : int,
                PAY_AMT2 : int,
                PAY_AMT3 : int,
                PAY_AMT4 : int,
                PAY_AMT5 : int,
                PAY_AMT6 : int,
                default_payment_next_month : int = None):
        try:
            self.LIMIT_BAL = LIMIT_BAL
            self.SEX = SEX
            self.EDUCATION = EDUCATION
            self.MARRIAGE = MARRIAGE
            self.AGE = AGE
            self.PAY_0 = PAY_0
            self.PAY_2 = PAY_2
            self.PAY_3 = PAY_3
            self.PAY_4 = PAY_4
            self.PAY_5 = PAY_5
            self.PAY_6 = PAY_6
            self.BILL_AMT1 = BILL_AMT1
            self.BILL_AMT2 = BILL_AMT2
            self.BILL_AMT3 = BILL_AMT3
            self.BILL_AMT4 = BILL_AMT4
            self.BILL_AMT5 = BILL_AMT5
            self.BILL_AMT6 = BILL_AMT6
            self.PAY_AMT1 = PAY_AMT1
            self.PAY_AMT2 = PAY_AMT2
            self.PAY_AMT3 = PAY_AMT3
            self.PAY_AMT4 = PAY_AMT4
            self.PAY_AMT5 = PAY_AMT5
            self.PAY_AMT6 = PAY_AMT6
            self.default_payment_next_month = default_payment_next_month
        except Exception as e:
            raise CreditCardException(e, sys) from e
      
    def credit_card_input_data_frame(self):
        try:
            data = self.get_credit_card_data_as_dict()
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            raise CreditCardException(e, sys) from e  
        
    def get_credit_card_data_as_dict(self):
        try:
            input_data = {
                'LIMIT_BAL' : [self.LIMIT_BAL],
                'SEX' : [self.SEX],
                'EDUCATION' : [self.EDUCATION],
                'MARRIAGE' : [self.MARRIAGE],
                'PAY_0' : [self.PAY_0],
                'PAY_2' : [self.PAY_2],
                'PAY_3' : [self.PAY_3],
                'PAY_4' : [self.PAY_4],
                'PAY_5' : [self.PAY_5],
                'PAY_6' : [self.PAY_6],
                'BILL_AMT1' : [self.BILL_AMT1],
                'BILL_AMT2' : [self.BILL_AMT2],
                'BILL_AMT3' : [self.BILL_AMT3],
                'BILL_AMT4' : [self.BILL_AMT4],
                'BILL_AMT5' : [self.BILL_AMT5],
                'BILL_AMT6' : [self.BILL_AMT6],
                'PAY_AMT1' : [self.PAY_AMT1],
                'PAY_AMT2' : [self.PAY_AMT2],
                'PAY_AMT3' : [self.PAY_AMT3],
                'PAY_AMT4' : [self.PAY_AMT4],
                'PAY_AMT5' : [self.PAY_AMT5],
                'PAY_AMT6' : [self.PAY_AMT6]
            }
            return input_data   
        except Exception as e:
            raise CreditCardException(e, sys) from e
        

class Credit_Card_Predictor:
    
    def __init__(self, model_dir : str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            filename = os.listdir(latest_model_dir)[0]
            latest_model_file_path = os.path.join(latest_model_dir, filename)
            return latest_model_file_path
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def predict(self, X):
        try:
            model_file_path = self.get_latest_model_path()
            
            model = load_object(model_file_path)
            
            columns = ['LIMIT_BAL', 'PAY_0','PAY_2','PAY_3','BILL_AMT1','BILL_AMT2',
                       'BILL_AMT3','PAY_AMT1','PAY_AMT2','PAY_AMT3']
            
            X = X[columns]
            
            default_payment_next_month = model.predict(X)
            
            default_payment_next_month = [True if i == 1 else False for i in default_payment_next_month]
            
            return default_payment_next_month
        except Exception as e:
            raise CreditCardException(e, sys) from e  