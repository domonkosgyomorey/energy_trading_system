import numpy as np


class Household:
    def __init__(self, household_id: int, battery_capacity: float):
        self.household_id = household_id
        self.battery_capacity = battery_capacity
        self.battery_level = battery_capacity / 2

    def make_forecasts(self, days: int):
        demand = np.random.uniform(1, 5, days)
        generation = np.random.uniform(0, 4, days)
        return demand, generation

    def update_battery_level(self, charge: float, discharge: float):
        self.battery_level = max(
            0, min(self.battery_capacity, self.battery_level + charge - discharge)
        )
