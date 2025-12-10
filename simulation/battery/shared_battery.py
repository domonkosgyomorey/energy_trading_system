from simulation.battery.battery import Battery
from simulation.battery.central_battery import CentralBattery

class SharedBattery(Battery):
    
    def __init__(self, household_id: str, central_battery: CentralBattery):
        self.household_id: str = household_id
        self.central_battery: CentralBattery = central_battery

    def store_energy(self, amount: float) -> None:
        self.central_battery.store_energy(household_id=self.household_id, amount=amount)

    def retrieve_energy(self, amount: float) -> float:
        return self.central_battery.retrieve_energy(household_id=self.household_id, amount=amount)

    def get_stored_kwh(self) -> float:
        return self.central_battery.get_stored_kwh(household_id=self.household_id)

    def get_capacity_in_kwh(self) -> float:
        return self.central_battery.get_capacity_in_kwh(household_id=self.household_id)

    def get_fields(self) -> dict:
        return {
            "discharge_efficiency": self.central_battery.discharge_efficiency,
            "charge_efficiency": self.central_battery.charge_efficiency,
            "capacity_in_kwh": self.central_battery.capacity_in_kwh,
            "stored_kwh": self.get_stored_kwh()
        }

