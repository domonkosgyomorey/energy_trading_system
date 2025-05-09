from typing import Protocol

from simulation.household import Household

class OptimizerStrategy(Protocol):
    def optimize(self, households: list[Household], forecasts: dict[int, list[dict]]) -> list[dict]:
        ...