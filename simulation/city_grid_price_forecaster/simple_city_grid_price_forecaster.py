from simulation.city_grid_price_forecaster.city_gird_price_forecaster import CityGridPriceForecaster
from typing import Literal

class SimpleCityGridPriceForecaster(CityGridPriceForecaster):

    def forecast(self, price_history: list[float], forecast_size: int) -> dict[Literal["sell", "buy"], list[float]]:
        return {
            "sell": [0.01 for _ in range(forecast_size)],
            "buy": [100 for _ in range(forecast_size)]
        }
