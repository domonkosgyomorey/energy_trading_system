from simulation.battery.battery import Battery

class SimpleBattery(Battery):
    def __init__(self, capacity_in_kwh: float, charge_efficiency: float, discharge_efficiency: float, initial_charge_kwh: float = 5.0):
        self.capacity_in_kwh: float = capacity_in_kwh
        self.charge_efficiency: float = charge_efficiency
        self.discharge_efficiency: float = discharge_efficiency
        self.stored_kwh: float = min(initial_charge_kwh, capacity_in_kwh)

    def store_energy(self, amount: float) -> None:
        amount *= self.charge_efficiency
        self.stored_kwh = min(self.capacity_in_kwh, self.stored_kwh+amount)

    def retrieve_energy(self, amount: float) -> float:
        amount += (1-self.discharge_efficiency)*amount
        if self.stored_kwh > amount:
            self.stored_kwh -= amount
            return amount
        else:
            return_kwh = self.stored_kwh
            self.stored_kwh = 0
            return return_kwh

    def get_stored_kwh(self) -> float:
        return self.stored_kwh*self.discharge_efficiency

    def get_capacity_in_kwh(self) -> float:
        return self.capacity_in_kwh

    def get_fields(self) -> dict:
        return {
            "discharge_efficiency": self.discharge_efficiency,
            "charge_efficiency": self.charge_efficiency,
            "capacity_in_kwh": self.capacity_in_kwh,
            "stored_kwh": self.stored_kwh
        }
