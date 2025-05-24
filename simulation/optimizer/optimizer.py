from typing import Literal, Protocol

from simulation.household import Household

class OptimizerStrategy(Protocol):

    def optimize(self, households: list[Household], forecasts: dict[str, dict[Literal["production", "consumption"], list[float]]], city_grid_prices_forecast: list[float]) -> list[dict]:
        ...
