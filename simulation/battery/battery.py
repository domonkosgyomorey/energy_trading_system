from typing import Protocol

class Battery(Protocol):
    def store_energy(self, amount: float):
        ...

    def retrieve_energy(self, amount: float) -> float:
        ...

    def degrade(self):
        ...

    def status(self) -> dict:
        ...