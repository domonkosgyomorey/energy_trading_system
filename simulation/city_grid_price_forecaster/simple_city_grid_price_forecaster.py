from simulation.city_grid_price_forecaster.city_gird_price_forecaster import CityGridPriceForecaster

class SimpleCityGridPriceForecaster(CityGridPriceForecaster):

    def forecast(self, price_history: list[float], forecast_size: int) -> list[float]:
        return [1 for _ in range(forecast_size)]
