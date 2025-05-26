import numpy as np
import cvxpy as cp
from simulation.optimizer.optimizer import OptimizerStrategy
from simulation.household import Household
from typing import Literal

class SimpleRuleBasedOptimizer(OptimizerStrategy):

    def optimize(self, households: list[Household], 
                 forecasts: dict[str, dict[Literal["production", "consumption"], list[float]]], 
                 city_grid_prices_forecast: list[float],
                 current_consumption: dict[str, float]) -> list[dict]:
        N = len(households)
        T = len(city_grid_prices_forecast)

        P = np.array([forecasts[hh.id]["production"] for hh in households])  # (N, T)
        C = np.array([forecasts[hh.id]["consumption"] for hh in households])  # (N, T)
        B0 = np.array([hh.battery.get_fields().get("stored_kwh", 0.0) for hh in households])
        B_max = np.array([hh.battery.get_capacity_in_kwh() for hh in households])
        eta_charge = np.array([hh.battery.get_fields().get("charge_efficiency", 1.0) for hh in households])
        eta_discharge = np.array([hh.battery.get_fields().get("discharge_efficiency", 1.0) for hh in households])

        # Jelenlegi fogyasztás tömb (N,)
        current_C = np.array([current_consumption.get(hh.id, 0.0) for hh in households])

        # Változók
        G_buy = cp.Variable((N, T), nonneg=True)   
        B = cp.Variable((N, T + 1), nonneg=True)   
        E = cp.Variable((N, N, T), nonneg=True)    
        B_charge = cp.Variable((N, T), nonneg=True)  
        B_discharge = cp.Variable((N, T), nonneg=True)  

        constraints = [B[:, 0] == B0]

        for t in range(T):
            for i in range(N):
                # Akkumulátor kapacitás korlát
                constraints += [B[i, t + 1] <= B_max[i]]

                # Saját magának nem küldhet P2P energiát
                constraints += [E[i, i, t] == 0]

                # P2P energia korlát: nem lehet több mint termelés + akkumulátor kisütés + grid eladás
                constraints += [
                    cp.sum(E[i, :, t]) <= P[i, t] + B_discharge[i, t]
                ]

            for i in range(N):
                # Akkumulátor állapot frissítése
                constraints += [
                    B[i, t + 1] == B[i, t] + B_charge[i, t] * eta_charge[i] - B_discharge[i, t] / eta_discharge[i]
                ]

                # Energia egyensúly: bejövő energia = kimenő energia
                energy_in = P[i, t] + cp.sum(E[:, i, t]) + G_buy[i, t] + B_discharge[i, t]

                # Az első iterációnál (t=0) a jelenlegi fogyasztást használjuk,
                # a későbbi időlépéseknél a forecastból vett fogyasztást.
                consumption = current_C[i] if t == 0 else C[i, t]

                energy_out = consumption + cp.sum(E[i, :, t]) + B_charge[i, t]

                constraints += [energy_in == energy_out]

                # Akkumulátor töltési/kisütési korlátok (max 20% kapacitás óránként)
                max_charge_rate = B_max[i] * 0.90
                constraints += [
                    B_charge[i, t] <= max_charge_rate,
                    B_discharge[i, t] <= max_charge_rate,
                    B_discharge[i, t] <= B[i, t]  # nem lehet többet kisütni mint ami van
                ]


        early_grid_penalty = np.array(city_grid_prices_forecast)
        early_grid_penalty[0] *= 10
        objective = cp.Minimize(
            cp.sum(cp.multiply(G_buy, early_grid_penalty)) +   # városi vásárlás költsége
            1 * cp.sum(E)                                          # P2P tranzakciók büntetése
        )

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.ECOS)
            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                problem.solve(solver=cp.SCS)
        except:
            try:
                problem.solve(solver=cp.SCS)
            except:
                problem.solve(solver=cp.OSQP)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"Optimalizálás sikertelen: {problem.status}")

        output = []

        for i in range(N):
            p2p_transactions = []
            total_p2p_sells = 0

            for j in range(N):
                if i != j:
                    total_energy_bought = sum(E[j, i, t].value if E[j, i, t].value is not None else 0 for t in range(T))
                    if total_energy_bought > 1e-6:
                        p2p_transactions.append({
                            "from": households[j].id,
                            "amount": float(total_energy_bought)
                        })

                    total_energy_sold = sum(E[i, j, t].value if E[i, j, t].value is not None else 0 for t in range(T))
                    total_p2p_sells += total_energy_sold

            grid_buy = sum(G_buy[i, t].value if G_buy[i, t].value is not None else 0 for t in range(T))

            output.append({
                "id": households[i].id,
                "sells": float(total_p2p_sells),
                "buys": p2p_transactions,
                "buy_from_city": float(grid_buy),
            })

        return output

