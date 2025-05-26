from simulation.city_grid_price_forecaster.city_gird_price_forecaster import CityGridPriceForecaster
from typing import Literal
import numpy as np

class SimpleCityGridPriceForecaster(CityGridPriceForecaster):
    def __init__(self):
        super().__init__()
        self._min_sell: float = 0.01
        self._max_sell: float = 15
        self._min_buy: float = 9
        self._max_buy: float = 50

    def forecast(self, price_history: list[float], forecast_size: int) -> dict[Literal["sell", "buy"], list[float]]:
        buy_prices: list[float] = [np.random.uniform(self._min_buy, self._max_buy) for _ in range(forecast_size)]

        max_allowed_sell: int = min(buy_prices) - self._min_sell  # sell < min_buy
        effective_max_sell = min(self._max_sell, max_allowed_sell)

        sell_prices = [np.random.uniform(self._min_sell, effective_max_sell) for _ in range(forecast_size)]

        return {
            "sell": sell_prices,
            "buy": buy_prices
        }
