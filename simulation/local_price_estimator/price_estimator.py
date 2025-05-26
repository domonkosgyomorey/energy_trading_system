class PriceEstimator():

    def calculate_price(self, total_production: float, total_consumption: float) -> float:
        return total_consumption / total_production
