class CentralBattery:
    def __init__(self, capacity_in_kwh: float, charge_efficiency: float, discharge_efficiency: float, tax_per_kwh: float):
        self.capacity_in_kwh: float = capacity_in_kwh
        self.charge_efficiency: float = charge_efficiency
        self.discharge_efficiency: float = discharge_efficiency
        self.tax_per_kwh: float = tax_per_kwh

        # For every households stores the stored kwh for tax calculation
        self.households_shared_battery: dict[str, float] = {}

    def retrieve_energy(self, household_id: str, amount: float) -> float:
        new_amount = amount + amount*(1-self.discharge_efficiency)
        if self.households_shared_battery.get(household_id) is None:
            return 0
        elif self.households_shared_battery[household_id] >= new_amount:
            self.households_shared_battery[household_id] -= new_amount
            return amount
        else:
            return_kwh = self.households_shared_battery[household_id]
            self.households_shared_battery[household_id] = 0
            return return_kwh

    def store_energy(self, household_id: str, amount: float) -> None:
        amount *= self.charge_efficiency
        if self.households_shared_battery.get(household_id) is None:
            self.households_shared_battery[household_id] = 0
        self.households_shared_battery[household_id] += amount

    def update(self):
        for household_id, stored_kwh in self.households_shared_battery.items():
            tax = stored_kwh*self.tax_per_kwh
            if stored_kwh > 0:
                self.households_shared_battery[household_id] -= tax

    def get_stored_kwh(self, household_id: str) -> float:
        return self.households_shared_battery[household_id]

    def get_capacity_in_kwh(self) -> float:
        return self.capacity_in_kwh

    def get_tax_per_kwh(self, household_id: str) -> float:
        return self.tax_per_kwh * self.households_shared_battery[household_id]

