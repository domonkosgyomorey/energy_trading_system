from simulation.battery.battery import Battery

class SimpleBattery(Battery):
    def __init__(self, capacity_in_kwh: float, efficiency: float):
        self.capacity_in_kwh: float = capacity_in_kwh
        self.efficiency: float = efficiency
        self.stored_kwh: float = 0.0

    def store_energy(self, amount: float) -> None:
        self.store_kwh = min(self.capacity_in_kwh, self.stored_kwh+amount)

    def retrieve_energy(self, amount: float) -> float:
        if self.stored_kwh > amount:
            self.stored_kwh -= amount
            return amount
        else:
            return_kwh = self.stored_kwh
            self.stored_kwh = 0
            return return_kwh

    def update(self):
        self.stored_kwh *= self.efficiency

    def get_stored_kwh(self) -> float:
        return self.stored_kwh

    def get_capacity_in_kwh(self) -> float:
        return self.capacity_in_kwh

    def get_efficiency(self) -> float:
        return self.efficiency
