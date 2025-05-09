from simulation.household import Household
import random

class Forecaster:
    def forecast(self, household: Household, days=5):
        forecast = []
        for day in range(days):
            prod = round(random.uniform(0.0, 10.0), 2)
            cons = round(random.uniform(0.0, 10.0), 2)
            forecast.append({"day": day, "production": prod, "consumption": cons})
        return forecast