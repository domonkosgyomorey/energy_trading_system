import cvxpy as cp
import numpy as np
from blockchain import Blockchain
from household import Household


class OptimizationModel:
    def __init__(self, blockchain: Blockchain, household: Household):
        self.blockchain = blockchain
        self.household = household

    def optimize(
        self, current_day: int, days: int, forecast_demand, forecast_generation
    ):
        battery_charge = cp.Variable(days)
        battery_discharge = cp.Variable(days)
        buy = cp.Variable(days)
        sell = cp.Variable(days)

        cost_buy = np.ones(days) * 0.15
        cost_sell = np.ones(days) * 0.1

        available_offers = self.blockchain.get_offers(current_day, days)
        reservations = self.blockchain.get_reservations(current_day)
        household_reservations = [
            offer_id for day, offer_id in reservations.items() if day >= current_day
        ]

        for i, (offer_id, offer) in enumerate(available_offers):
            if offer_id in household_reservations:
                offer_day_index = offer["day"] - current_day
                cost_buy[offer_day_index] = offer["price"]
                self.blockchain.mark_offer_as_used(offer["day"], offer_id)

        objective = cp.Minimize(cp.sum(cost_buy * buy - cost_sell * sell))

        constraints = [
            battery_charge >= 0,
            battery_discharge >= 0,
            buy >= 0,
            sell >= 0,
            cp.cumsum(battery_charge - battery_discharge)
            <= self.household.battery_capacity,
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status == cp.OPTIMAL:
            self.household.update_battery_level(
                charge=sum(battery_charge.value), discharge=sum(battery_discharge.value)
            )

        return {
            "cost": problem.value,
            "buy": buy.value,
            "sell": sell.value,
            "battery_charge": battery_charge.value,
            "battery_discharge": battery_discharge.value,
        }
