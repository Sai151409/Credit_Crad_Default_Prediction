from setuptools import setup, find_packages
from typing import List

#Declaring Variables for setup functions
PROJECT_NAME = 'credit-card-default-prediction'
VERSION = '0.0.1'
AUTHOR = 'Sai Ram'
DESCRIPTION = 'This is credit card default prediction'
REQUIREMENTS_LIST = "requirements.txt"
HYPHEN_E_DOT = '-e .'

def get_requirements() -> List[str]:
    '''
    Description : This function to going ot return the 
    list of requirements mention in requirements.txt'''
    with open(REQUIREMENTS_LIST) as file:
        requirement_list = [i.replace('\n', '') for i in file.readlines()]
        if HYPHEN_E_DOT in requirement_list:
            requirement_list.remove(HYPHEN_E_DOT)
    
        return requirement_list



setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=get_requirements()
)