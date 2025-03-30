import pandas as pd
from household import Household
from tqdm import tqdm
from utils.logger import logger


class Community:

    def __init__(self, db_path: str):
        """
        db_path: str: Path to the database file.
        """

        df = pd.read_parquet(db_path)
        logger.info("Database loaded successfully.")

        household_ids = df["id"].unique()
        self.households: list[Household] = []
        for id in tqdm(household_ids, desc="Loading households"):
            self.households.append(Household(id, df[df["id"] == id]))

        logger.info("Households loaded successfully.")
