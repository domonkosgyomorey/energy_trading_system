from household import Household

from blockchain import Blockchain


class OptimizationModel:
    def __init__(self, blockchain: Blockchain, household: Household):
        self.blockchain = blockchain
        self.household = household

    def optimize(
        self, current_day: int, days: int, forecast_demand, forecast_generation
    ):
        pass
