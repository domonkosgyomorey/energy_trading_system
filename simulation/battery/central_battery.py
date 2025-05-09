class CentralBattery:
    def __init__(self, capacity: float, efficiency: float, tax_per_kwh: float):
        self.capacity = capacity
        self.efficiency = efficiency
        self.level = 0.0
        self.tax_per_kwh = tax_per_kwh

    def store_energy(self, amount: float):
        taxed = max(0, amount - self.tax_per_kwh)
        actual = min(taxed * self.efficiency, self.capacity - self.level)
        self.level += actual
        return actual

    def retrieve_energy(self, amount: float) -> float:
        available = min(amount, self.level)
        self.level -= available
        return available * self.efficiency

    def degrade(self):
        self.level *= 0.998  # slower loss than home batteries

    def status(self):
        return {
            "capacity": self.capacity,
            "efficiency": self.efficiency,
            "level": round(self.level, 2),
            "tax_per_kwh": self.tax_per_kwh
        }