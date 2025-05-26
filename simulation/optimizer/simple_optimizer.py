from simulation.optimizer.optimizer import OptimizerStrategy
from simulation.household import Household
from typing import Literal

class SimpleOptimizer(OptimizerStrategy):

    def optimize(self, households: list[Household], 
                 forecasts: dict[str, dict[Literal["production", "consumption"], list[float]]], 
                 city_grid_prices_forecast: list[float],
                 current_consumption: dict[str, float]) -> list[dict]:
        return []
