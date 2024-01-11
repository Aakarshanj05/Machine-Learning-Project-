from setuptools import setup, find_packages
from typing import List

PROJECT_NAME = "Machine Learning Project"
VERSION = "0.0.1"
DESCRIPTION = "This is our machine learning project in modular coding"
AUTHOR_NAME = "Aakarshan Jha"
AUTHOR_EMAIL = "aakarshan5june@gmail.com"  # Corrected variable name

REQUIREMENTS_FILE_NAME = "requirements.txt"

HYPHEN_E_DOT = "-e ."

def get_requirements_list() -> List[str]:
    try:
        with open(REQUIREMENTS_FILE_NAME) as requirement_file:
            requirement_list = requirement_file.readlines()
            requirement_list = [requirement_name.replace("\n", "") for requirement_name in requirement_list if requirement_name.strip()]
            if HYPHEN_E_DOT in requirement_list:
                requirement_list.remove(HYPHEN_E_DOT)
            return requirement_list
    except FileNotFoundError:
        print(f"Error: The '{REQUIREMENTS_FILE_NAME}' file was not found.")
        return []

setup(
    name=PROJECT_NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,  
    packages=find_packages(),
    install_requires=get_requirements_list()  
)
