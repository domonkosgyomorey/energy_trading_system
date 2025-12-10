class PriceEstimator():

    def calculate_price(self, total_production: float, total_consumption: float) -> float:
        if total_production <= 0:
            return 50.0  # Return high price when no production available
        return max(0.01, total_consumption / total_production)  # Minimum price floor
