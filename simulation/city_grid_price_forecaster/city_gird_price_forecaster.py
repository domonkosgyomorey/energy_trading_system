from abc import ABC, abstractmethod

class CityGridPriceForecaster(ABC):

    @abstractmethod
    def forecast(self, price_history: list[float], forecast_size: int) -> list[float]:
        ...
