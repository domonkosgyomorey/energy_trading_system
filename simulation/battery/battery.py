from typing import Protocol

class Battery(Protocol):
    def store_energy(self, amount: float) -> None:
        ...

    def retrieve_energy(self, amount: float) -> float:
        ...

    def get_stored_kwh(self) -> float:
        ...

    def get_capacity_in_kwh(self) -> float:
        ...

    def get_fields(self) -> dict:
        ...
