import random
from simulation.battery.battery import Battery
from simulation.household_resource_handler.resource_handler import ResourceHandler
from typing import Literal

class Household:
    def __init__(self, id: str, data:dict, battery: Battery, resource_handler: ResourceHandler):
        self.id: str = id
        self.data: dict = data
        self.battery: Battery = battery
        self.resource_handler: ResourceHandler = resource_handler

    def update(self, iteration: int, updated_token: float):
        current_production: float = self.data["production"][iteration]
        current_consumption: float = self.data["consumption"][iteration]
        self.resource_handler.handle(current_production, current_consumption, self.battery, updated_token)            

    def get_current_data(self, iteration: int) -> dict[Literal["production", "consumption", "stored_kwh"], float]:
        return {
            "production": self.data["production"][iteration],
            "consumption": self.data["consumption"][iteration],
            "stored_kwh": self.battery.get_stored_kwh()
        }
