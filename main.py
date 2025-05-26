from simulation.simulation import Simulation
import pandas as pd
from simulation.config import Config
from simulation.utils.logger import logger
import os

if __name__ == "__main__":
    logger.info("Loading db for households")
    household_db_path = os.path.join(Config.DB_PATH, Config.DB)
    household_data = pd.read_parquet(path=household_db_path)
    logger.info("Household db loaded")

    sim = Simulation(household_data)
    logger.info("Start simulation...")
    sim.run(steps=30)
    logger.info("Simulation ended...")
