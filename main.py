from simulation.simulation import Simulation
import pandas as pd
from simulation.config import Config
import os

if __name__ == "__main__":
    household_db_path = os.path.join(Config.DB_PATH, Config.DB)
    household_data = pd.read_parquet(path=household_db_path)
    
    sim = Simulation(household_data)
    sim.run(steps=3)
