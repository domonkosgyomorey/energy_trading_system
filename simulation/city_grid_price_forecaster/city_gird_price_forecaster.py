from abc import ABC, abstractmethod
from typing import Literal

class CityGridPriceForecaster(ABC):

    @abstractmethod
    def forecast(self, price_history: list[float], forecast_size: int) -> dict[Literal["sell", "buy"], list[float]]:
        ...
