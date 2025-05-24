class CentralBattery:
    def __init__(self, capacity_in_kwh: float, efficiency: float, tax_per_kwh: float):
        self.capacity_in_kwh: float = capacity_in_kwh
        self.efficiency: float = efficiency
        self.tax_per_kwh: float = tax_per_kwh

        # For every households stores the stored kwh for tax calculation
        self.households_shared_battery: dict[str, float] = {}

    def retrieve_energy(self, household_id: str, amount: float) -> float:
        if self.households_shared_battery.get(household_id) is None:
            return 0
        elif self.households_shared_battery[household_id] >= amount:
            self.households_shared_battery[household_id] -= amount
            return amount
        else:
            return_kwh = self.households_shared_battery[household_id]
            self.households_shared_battery[household_id] = 0
            return return_kwh

    def store_energy_for_house(self, household_id: str, amount: float) -> None:
        if self.households_shared_battery.get(household_id) is None:
            self.households_shared_battery[household_id] = 0
        self.households_shared_battery[household_id] += amount

    def update(self, household_id: str):
        if self.households_shared_battery.get(household_id) is None:
            return

        self.households_shared_battery[household_id] *= self.efficiency
        amount = self.households_shared_battery[household_id]
        tax = amount*self.tax_per_kwh
        if amount > 0:
            self.households_shared_battery[household_id] -= tax

    def get_stored_kwh(self, household_id: str) -> float:
        return self.households_shared_battery[household_id]

    def get_efficiency(self) -> float:
        return self.efficiency

    def get_capacity_in_kwh(self) -> float:
        return self.capacity_in_kwh

    def get_tax_per_kwh(self, household_id: str) -> float:
        return self.tax_per_kwh * self.households_shared_battery[household_id]

