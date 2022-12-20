import os
import sys


class CreditCardException(Exception):
    
    def __init__(self, error_message : Exception, error_detail : sys):
        super().__init__(error_message)
        self.error_message = CreditCardException.get_detailed_error_message(
            error_message=error_message,
            error_detail=error_detail)
        
        
    
    @staticmethod
    def get_detailed_error_message(error_message:Exception, error_detail:sys):
        '''
        error_message : Exception object
        error_details : object of sys module
        '''
        _, _, exc_tab = error_detail.exc_info()
        
        except_block_line_number = exc_tab.tb_frame.f_lineno
        
        try_block_line_number = exc_tab.tb_lineno
        
        filename = exc_tab.tb_frame.f_code.co_filename
        
        error_message = f"""
        Error has occured in script :[
            {filename}
            ],
        at try block line number [{try_block_line_number}] and 
        except block line number [{except_block_line_number}]
        error message : [{error_message}]
        """
        return error_message
    
    def __str__(self)->str:
        return self.error_message
    
    def __repr__(self) -> str:
        return CreditCardException.__name__.str()