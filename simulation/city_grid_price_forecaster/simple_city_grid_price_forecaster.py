from simulation.city_grid_price_forecaster.city_gird_price_forecaster import CityGridPriceForecaster
from simulation.params import GridPriceParams
from typing import Literal
import numpy as np


class SimpleCityGridPriceForecaster(CityGridPriceForecaster):
    def __init__(self, price_params: GridPriceParams | None = None):
        super().__init__()

        params = price_params or GridPriceParams()
        self._min_sell: float = params.min_sell_price
        self._max_sell: float = params.max_sell_price
        self._min_buy: float = params.min_buy_price
        self._max_buy: float = params.max_buy_price

    def forecast(self, price_history: list[float], forecast_size: int) -> dict[Literal["sell", "buy"], list[float]]:
        buy_prices: list[float] = [np.random.uniform(self._min_buy, self._max_buy) for _ in range(forecast_size)]

        max_allowed_sell: float = min(buy_prices) - self._min_sell  # sell < min_buy
        effective_max_sell: float = min(self._max_sell, max_allowed_sell)

        sell_prices: list[float] = [np.random.uniform(self._min_sell, effective_max_sell) for _ in range(forecast_size)]

        return {
            "sell": sell_prices,
            "buy": buy_prices
        }
