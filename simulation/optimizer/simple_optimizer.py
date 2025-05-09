from simulation.battery.central_battery import CentralBattery
from simulation.household import Household

class SimpleRuleBasedOptimizer:
    def __init__(self, central_battery: CentralBattery):
        self.central_battery = central_battery

    def optimize(self, households: list[Household], forecasts: dict[int, list[dict]]) -> list[dict]:
        offers = []
        for hh in households:
            hh_forecast = forecasts[hh.id]
            for day_data in hh_forecast:
                surplus = day_data["production"] - day_data["consumption"]

                if hh.battery:
                    future_net = sum(d['production'] - d['consumption'] for d in hh_forecast[day_data['day']+1:])
                    if surplus > 0 and future_net < 0:
                        continue
                    if surplus < 0 and hh.battery.level > 0:
                        continue
                elif self.central_battery:
                    if surplus > self.central_battery.tax_per_kwh:
                        continue  # érdemesebb eltárolni

                if surplus > 1.0:
                    offers.append({
                        "type": "offer",
                        "household_id": hh.id,
                        "day": day_data["day"],
                        "amount": round(surplus, 2)
                    })
                elif surplus < -1.0:
                    offers.append({
                        "type": "request",
                        "household_id": hh.id,
                        "day": day_data["day"],
                        "amount": round(-surplus, 2)
                    })
        return offers