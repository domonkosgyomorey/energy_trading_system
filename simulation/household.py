from simulation.battery.battery import Battery
from typing import Literal

class Household:
    def __init__(self, id: str, data:dict, battery: Battery):
        self.id: str = id
        self.data: dict = data
        self.battery: Battery = battery
        self.wallet: float = 0 

    def get_sensor_data(self, iteration: int) -> dict[Literal["production", "consumption", "stored_kwh", "wallet"], float]:
        return {
            "production": self.data["production"][iteration],
            "consumption": self.data["consumption"][iteration],
            "stored_kwh": self.battery.get_stored_kwh(),
            "wallet": self.wallet
        }
