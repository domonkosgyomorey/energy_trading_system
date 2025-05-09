from simulation.household import Household
from typing import Protocol

# --- Optimizer Interface ---
class OptimizerStrategy(Protocol):
    def optimize(self, households: list[Household], forecasts: dict[int, list[dict]]) -> list[dict]:
        ...

# --- Simple Optimizer ---
class SimpleRuleBasedOptimizer:
    def optimize(self, households: list[Household], forecasts: dict[int, list[dict]]) -> list[dict]:
        offers = []
        for hh in households:
            hh_forecast = forecasts[hh.id]
            for day_data in hh_forecast:
                surplus = day_data["production"] - day_data["consumption"]

                # If battery exists, keep surplus if future days are forecasted to be negative
                if hh.battery:
                    future_net = sum(d['production'] - d['consumption'] for d in hh_forecast[day_data['day']+1:])
                    if surplus > 0 and future_net < 0:
                        continue  # better to store energy
                    if surplus < 0 and hh.battery.level > 0:
                        continue  # use own battery later

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