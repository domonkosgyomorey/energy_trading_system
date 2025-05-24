from simulation.forecaster.forecaster import Forecaster
from simulation.household import Household
from typing import Literal

class PerfectForecaster(Forecaster):

    def forecast(self, household: Household, iteration: int, prediction_range: int) -> dict[Literal["production", "consumption"], list[float]]:
        return {
            "production": household.data["production"][iteration:iteration+prediction_range],
            "consumption": household.data["consumption"][iteration:iteration+prediction_range]
        } 
