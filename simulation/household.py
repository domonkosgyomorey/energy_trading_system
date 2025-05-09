import random
from typing import Optional
from simulation.battery.battery import Battery
from simulation.battery.central_battery import CentralBattery

class Household:
    def __init__(self, id, battery: Optional[Battery] = None):
        self.id = id
        self.production = 0.0
        self.consumption = 0.0
        self.battery = battery
        self.tokens = 0.0

    def simulate(self):
        self.production = round(random.uniform(0.0, 10.0), 2)
        self.consumption = round(random.uniform(0.0, 10.0), 2)
        return self.production - self.consumption

    def handle_energy(self, net_energy: float, central_battery: Optional[CentralBattery] = None):
        if self.battery is not None:
            if net_energy > 0:
                stored = self.battery.store_energy(net_energy)
                net_energy -= stored
            else:
                retrieved = self.battery.retrieve_energy(-net_energy)
                net_energy += retrieved
            self.battery.degrade()
        elif central_battery:
            if net_energy > 0:
                stored = central_battery.store_energy(net_energy)
                net_energy -= stored
            else:
                retrieved = central_battery.retrieve_energy(-net_energy)
                net_energy += retrieved
            central_battery.degrade()

        self.tokens += net_energy
        return {
            'id': self.id,
            'production': self.production,
            'consumption': self.consumption,
            'battery': self.battery.status() if self.battery else None,
            'tokens': round(self.tokens, 2)
        }

    def receive_energy(self, amount: float):
        if self.battery:
            remaining = amount - self.battery.store_energy(amount)
            self.tokens += remaining
        else:
            self.tokens += amount

    def provide_energy(self, amount: float) -> float:
        if self.battery:
            available = self.battery.retrieve_energy(amount)
            self.tokens -= available
            return available
        else:
            self.tokens -= amount
            return amount