import logging
import os
from datetime import datetime

BASE_DIR = r'F:\DOCUMENTS\Project\Stores-Sales-Prediction'

CURRENT_TIME_STAMP=  f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

LOG_FILE_NAME=f"log_{CURRENT_TIME_STAMP}.log"

LOGS = 'logs'
os.makedirs(LOGS,exist_ok=True)

LOG_FILE_PATH = os.path.join(BASE_DIR,LOGS,LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,

)
