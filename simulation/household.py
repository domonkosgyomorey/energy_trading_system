import random
from typing import Optional
from simulation.battery import BatteryInterface

class Household:
    def __init__(self, id, battery: Optional[BatteryInterface] = None):
        self.id = id
        self.production = 0.0
        self.consumption = 0.0
        self.battery = battery
        self.tokens = 0.0

    def simulate(self):
        self.production = round(random.uniform(0.0, 10.0), 2)
        self.consumption = round(random.uniform(0.0, 10.0), 2)
        net_energy = self.production - self.consumption

        if self.battery is not None:
            if net_energy > 0:
                stored = self.battery.store_energy(net_energy)
                net_energy -= stored
            else:
                retrieved = self.battery.retrieve_energy(-net_energy)
                net_energy += retrieved
            self.battery.degrade()

        self.tokens += net_energy
        return {
            'id': self.id,
            'production': self.production,
            'consumption': self.consumption,
            'battery': self.battery.status() if self.battery else None,
            'tokens': round(self.tokens, 2)
        }