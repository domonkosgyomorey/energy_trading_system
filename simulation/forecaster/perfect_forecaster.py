from simulation.forecaster.forecaster import Forecaster
from simulation.household import Household
from typing import Literal
from simulation.config import Config

class PerfectForecaster(Forecaster):

    def forecast(self, household: Household, iteration: int) -> dict[Literal["production", "consumption"], list[float]]:
        return {
            "production": household.data["production"][iteration:iteration+Config.FORECASTER_PRED_SIZE],
            "consumption": household.data["consumption"][iteration:iteration+Config.FORECASTER_PRED_SIZE]
        } 
