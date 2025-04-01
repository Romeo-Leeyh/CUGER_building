import logging
import os
import sys
from datetime import datetime

log_dir = "BuildingConvex/data/logs"
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(log_dir, f"{timestamp}.log")

log_format = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(log_file),  
        logging.StreamHandler(sys.stdout),  
    ]
)


data_dir = "E:/DATA/Moosasbuildingdatasets_02/output"


