import logging
import pandas as pd

logger = logging.getLogger("EnergyTradingLogger")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

df_logger: pd.DataFrame = pd.DataFrame(columns=["iteration", "id", "production", "consumption", "stored_kwh", "trades_to"])
