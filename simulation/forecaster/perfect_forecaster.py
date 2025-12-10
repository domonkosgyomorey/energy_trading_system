from simulation.forecaster.forecaster import Forecaster
from simulation.household import Household
from typing import Literal

class PerfectForecaster(Forecaster):

    def forecast(self, household: Household, iteration: int, forecast_size: int) -> dict[Literal["production", "consumption"], list[float]]:
        return {
            "production": household.data["production"][iteration:iteration+forecast_size],
            "consumption": household.data["consumption"][iteration:iteration+forecast_size],
        } 
