from typing import Literal, Protocol

from simulation.household import Household
from simulation.local_price_estimator.price_estimator import PriceEstimator
class OptimizerStrategy(Protocol):

     def optimize(self,
                households: list[Household],
                forecasts: dict[str, dict[Literal["production", "consumption"], list[float]]],
                city_grid_prices_forecast: list[float],
                city_grid_sell_prices_forecast: list[float],
                current_consumption: dict[str, float],
                current_production: dict[str, float],
                local_energy_price_estimator: PriceEstimator) -> list[dict]:
        ...
