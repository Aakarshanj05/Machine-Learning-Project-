import os
import logging
from pathlib import Path

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

project_name = input("Enter your project name: ")

# Corrected the typo in the directory name
list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/constant/__init__.py",  
    f"{project_name}/entity/__init__.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/utils/__init__.py",
    "config/config.yaml",
    "schema.yaml",  
    "app.py",
    "main.py",
    "logs.py",
    "exception.py",
    "setup.py"
]

for filepath in list_of_files:
    path = Path(filepath)
    filedir, filename = os.path.split(path)

    if filedir:
        os.makedirs(filedir, exist_ok=True)

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f:
            pass
    else:
        logging.info(f"File is already present at: {filepath}")
