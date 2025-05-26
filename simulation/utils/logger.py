import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger("EnergyTradingLogger")
logger.setLevel(logging.DEBUG)

log_filename = "energy_trading.log"

file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logger.handlers.clear()
logger.addHandler(file_handler)

df_logger: pd.DataFrame = pd.DataFrame()
def log_df(new_records: pd.DataFrame):
    global df_logger
    df_logger = pd.concat([df_logger, pd.DataFrame(new_records)])
    df_logger.to_csv("simulation_result.csv")

