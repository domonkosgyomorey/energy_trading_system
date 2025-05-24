from simulation.household import Household
from abc import ABC, abstractmethod
from typing import Literal

class Forecaster(ABC):

    @abstractmethod 
    def forecast(self, household: Household, iteration: int, prediction_range: int) -> dict[Literal["production", "consumption"], list[float]]:
        ...
