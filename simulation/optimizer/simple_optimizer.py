import numpy as np
import cvxpy as cp
from simulation.optimizer.optimizer import OptimizerStrategy
from simulation.household import Household
from typing import Literal

class SimpleRuleBasedOptimizer(OptimizerStrategy):
    
    def optimize(self, households: list[Household], forecasts: dict[str, dict[Literal["production", "consumption"], list[float]]], city_grid_prices_forecast: list[float]) -> list[dict]:
        N = len(households)
        T = len(city_grid_prices_forecast)

        P = np.array([forecasts[hh.id]["production"] for hh in households])  # (N, T)
        C = np.array([forecasts[hh.id]["consumption"] for hh in households])  # (N, T)
        B0 = np.array([hh.battery.get_fields().get("stored_kwh", 0.0) for hh in households])
        B_max = np.array([hh.battery.get_capacity_in_kwh() for hh in households])
        eta_plus = np.array([hh.battery.get_fields().get("discharge_efficiency", 1.0) for hh in households])
        eta_minus = np.array([hh.battery.get_fields().get("charge_efficiency", 1.0) for hh in households])

        G = cp.Variable((N, T))  # grid usage
        B = cp.Variable((N, T + 1))  # battery state
        E = cp.Variable((N, N, T))  # energy sent from i to j at time t

        constraints = [B[:, 0] == B0]

        for t in range(T):
            for i in range(N):
                constraints += [
                    G[i, t] >= 0,
                    B[i, t + 1] >= 0,
                    B[i, t + 1] <= B_max[i]
                ]
                for j in range(N):
                    if i != j:
                        constraints += [E[i, j, t] >= 0]

            for i in range(N):
                inflow = cp.sum(E[:, i, t]) - cp.sum(E[i, :, t])
                constraints += [
                    B[i, t + 1] == B[i, t] + P[i, t] + inflow * eta_plus[i] - C[i, t] * eta_minus[i] + G[i, t]
                ]

        objective = cp.Minimize(cp.sum(cp.multiply(G, city_grid_prices_forecast)))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS_BB)

        if problem.status != cp.OPTIMAL:
            raise RuntimeError(f"Optimalizálás sikertelen: {problem.status}")

        output = []
        P2P_out = E.sum(axis=1)  # shape (N, T)
        P2P_in = E.sum(axis=0)   # shape (N, T)

        for i in range(N):
            p2p_transactions = []
            for j in range(N):
                if i != j:
                    total_energy = sum(E[j, i, t].value for t in range(T))
                    if total_energy > 1e-4:
                        p2p_transactions.append({
                            "from": households[j].id,
                            "amount": float(total_energy)
                        })

            discharge = np.maximum(0, C[i, :] - P[i, :])  # közelítő becslés
            output.append({
                    "id": households[i].id,
                    "sells": float(P2P_out[i, :].value.sum()),
                    "buys": p2p_transactions,
                    "buy_from_city": float(G[i, :].value.sum()),
                    "used_accumulator": float(np.sum(discharge)),
                    "remaining_accumulator": float(B[i, -1].value)
            })
        return output
