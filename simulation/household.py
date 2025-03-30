import numpy as np


class Household:
    def __init__(self, household_id: int, db):
        self.household_id = household_id
        self.db = db

    def make_forecasts(self, days: int):
        demand = np.random.uniform(1, 5, days)
        generation = np.random.uniform(0, 4, days)
        return demand, generation

    def update_battery_level(self, charge: float, discharge: float):
        pass
