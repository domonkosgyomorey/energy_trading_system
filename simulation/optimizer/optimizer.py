from typing import Literal, Protocol

from simulation.household import Household
from simulation.local_price_estimator.price_estimator import PriceEstimator


class OptimizerStrategy(Protocol):
    """
    Protocol for optimization strategies.
    
    The optimizer decides optimal energy trading actions for each timestep:
    - How much to buy/sell from city grid
    - P2P trades between households
    - Battery charge/discharge scheduling
    """

    def optimize(
        self,
        households: list[Household],
        forecasts: dict[str, dict[Literal["production", "consumption"], list[float]]],
        city_grid_prices_forecast: list[float],
        city_grid_sell_prices_forecast: list[float],
        current_consumption: dict[str, float],
        current_production: dict[str, float],
        local_energy_price_estimator: PriceEstimator,
        grid_import_capacity_forecast: list[float] | None = None,
        grid_export_capacity_forecast: list[float] | None = None,
    ) -> list[dict]:
        """
        Optimize energy trading decisions.
        
        Args:
            households: List of household objects
            forecasts: Production/consumption forecasts per household
            city_grid_prices_forecast: Forecasted buy prices from grid (per timestep)
            city_grid_sell_prices_forecast: Forecasted sell prices to grid (per timestep)
            current_consumption: Current consumption per household
            current_production: Current production per household
            local_energy_price_estimator: P2P price calculator
            grid_import_capacity_forecast: Max kW that can be imported from grid (per timestep)
            grid_export_capacity_forecast: Max kW that can be exported to grid (per timestep)
            
        Returns:
            List of trade decisions per household
        """
        ...
