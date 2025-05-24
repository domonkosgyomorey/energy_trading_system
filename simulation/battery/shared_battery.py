from simulation.battery.battery import Battery
from simulation.battery.central_battery import CentralBattery

class SharedBattery(Battery):
    
    def __init__(self, household_id: str, central_battery: CentralBattery):
        self.household_id: str = household_id
        self.central_battery: CentralBattery = central_battery

    def store_energy(self, amount: float) -> None:
        self.central_battery.store_energy_for_house(household_id=self.household_id, amount=amount)

    def retrieve_energy(self, amount: float) -> float:
        return self.central_battery.retrieve_energy(household_id=self.household_id, amount=amount)

    def update(self) -> None:
        self.central_battery.update(household_id=self.household_id)

    def get_stored_kwh(self) -> float:
        return self.central_battery.get_stored_kwh(household_id=self.household_id)

    def get_capacity_in_kwh(self) -> float:
        return self.central_battery.get_capacity_in_kwh()

    def get_efficiency(self) -> float:
        return self.central_battery.get_efficiency()

