import numpy as np
import cvxpy as cp

from simulation.optimizer.optimizer import OptimizerStrategy
from simulation.household import Household
from simulation.local_price_estimator.price_estimator import PriceEstimator
from typing import Literal

class ConvexOptimizer(OptimizerStrategy):

    def optimize(self,
                households: list[Household],
                forecasts: dict[str, dict[Literal["production", "consumption"], list[float]]],
                city_grid_prices_forecast: list[float],
                city_grid_sell_prices_forecast: list[float],
                current_consumption: dict[str, float],
                current_production: dict[str, float],
                local_energy_price_estimator: PriceEstimator) -> list[dict]:

        N = len(households)
        T = len(city_grid_prices_forecast)

        # Forecasted production and consumption: shape (N, T)
        P = np.array([[current_production[hh.id]] + forecasts[hh.id]["production"] for hh in households])
        C = np.array([[current_consumption[hh.id]] + forecasts[hh.id]["consumption"] for hh in households])

        B0 = np.array([hh.battery.get_fields().get("stored_kwh", 0.0) for hh in households])
        B_max = np.array([hh.battery.get_capacity_in_kwh() for hh in households])
        eta_charge = np.array([hh.battery.get_fields().get("charge_efficiency", 1.0) for hh in households])
        eta_discharge = np.array([hh.battery.get_fields().get("discharge_efficiency", 1.0) for hh in households])
        wallets_0 = np.array([hh.wallet for hh in households])

        # Decision variables
        G_buy = cp.Variable((N, T), nonneg=True)
        G_sell = cp.Variable((N, T), nonneg=True)
        B = cp.Variable((N, T + 1), nonneg=True)
        E = cp.Variable((N, N, T), nonneg=True)
        B_charge = cp.Variable((N, T), nonneg=True)
        B_discharge = cp.Variable((N, T), nonneg=True)
        wallet = cp.Variable((N, T + 1))

        constraints = [B[:, 0] == B0, wallet[:, 0] == wallets_0]

        for t in range(T):
            prod_t = P[:, t]
            cons_t = C[:, t]

            total_prod_t = float(np.sum(prod_t))
            total_cons_t = float(np.sum(cons_t))
            p2p_price_t = local_energy_price_estimator.calculate_price(total_prod_t, total_cons_t)

            for i in range(N):
                constraints += [B[i, t + 1] <= B_max[i]]
                constraints += [E[i, i, t] == 0]

                constraints += [
                    B[i, t + 1] == B[i, t] + B_charge[i, t] * eta_charge[i] - B_discharge[i, t] / eta_discharge[i]
                ]

                max_rate = B_max[i] * 0.2
                constraints += [
                    B_charge[i, t] <= max_rate,
                    B_discharge[i, t] <= max_rate,
                    B_discharge[i, t] <= B[i, t]
                ]

                constraints += [
                    G_sell[i, t] <= prod_t[i] + B_discharge[i, t] - cp.sum(E[i, :, t])
                ]

                constraints += [
                    cp.sum(E[i, :, t]) <= prod_t[i] + B_discharge[i, t]
                ]

                constraints += [
                    G_buy[i, t] <= cons_t[i] + max_rate
                ]

                energy_available = prod_t[i] + cp.sum(E[:, i, t]) + G_buy[i, t] + B_discharge[i, t]
                energy_needed = cons_t[i] + B_charge[i, t] + cp.sum(E[i, :, t]) + G_sell[i, t]

                constraints += [energy_available >= energy_needed]

                p2p_sell_income = cp.sum(E[i, :, t]) * p2p_price_t
                p2p_buy_cost = cp.sum(E[:, i, t]) * p2p_price_t
                city_buy_cost = G_buy[i, t] * city_grid_prices_forecast[t]
                city_sell_income = G_sell[i, t] * city_grid_sell_prices_forecast[t]

                constraints += [
                    wallet[i, t + 1] == wallet[i, t]
                                    - city_buy_cost
                                    + city_sell_income
                                    - p2p_buy_cost
                                    + p2p_sell_income
                ]

        # Objective
        objective = cp.Minimize(
            cp.sum(cp.multiply(G_buy, city_grid_prices_forecast)) +
            cp.sum(cp.multiply(G_sell, city_grid_sell_prices_forecast)) +
            cp.sum(E) +
            100 * cp.sum(cp.pos(-wallet))
        )

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.COPT, verbose=False)
            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                problem.solve(solver=cp.SCS, verbose=False)
        except:
            problem.solve(solver=cp.SCS, verbose=False)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"Optimization failed: {problem.status}")

        # Only return t = 0 decisions
        output = []
        for i in range(N):
            p2p_buys = []

            for j in range(N):
                if i != j:
                    amount_bought = E[j, i, 0].value or 0.0
                    if amount_bought > 1e-6:
                        p2p_buys.append({
                            "from": households[j].id,
                            "amount": float(amount_bought)
                        })

            output.append({
                "id": households[i].id,
                "buy_from_city": float(G_buy[i, 0].value or 0),
                "sell_to_city": float(G_sell[i, 0].value or 0),
                "buys": p2p_buys,
            })

        return output

