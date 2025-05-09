from typing import Protocol

class BatteryInterface(Protocol):
    def store_energy(self, amount: float):
        ...

    def retrieve_energy(self, amount: float) -> float:
        ...

    def degrade(self):
        ...

    def status(self) -> dict:
        ...

class SimpleBattery:
    def __init__(self, capacity: float, efficiency: float):
        self.capacity = capacity
        self.efficiency = efficiency  # 0 to 1
        self.level = 0.0

    def store_energy(self, amount: float):
        actual = min(amount * self.efficiency, self.capacity - self.level)
        self.level += actual
        return actual

    def retrieve_energy(self, amount: float) -> float:
        available = min(amount, self.level)
        self.level -= available
        return available * self.efficiency

    def degrade(self):
        self.level *= 0.995  # 0.5% daily loss

    def status(self):
        return {
            "capacity": self.capacity,
            "efficiency": self.efficiency,
            "level": round(self.level, 2)
        }