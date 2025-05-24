from typing import Literal
from simulation.household import Household

import numpy as np
import cvxpy as cp

from simulation.config import Config
from simulation.optimizer.optimizer import OptimizerStrategy

class SimpleRuleBasedOptimizer(OptimizerStrategy):
    
    def optimize(self, households: list[Household], forecasts: dict[str, dict[Literal["production", "consumption"], list[float]]], city_grid_prices_forecast: list[float]) -> list[dict]:
        N = len(households)
        T = Config.FORECASTER_PRED_SIZE
        
        E = cp.Variable((N, N, T), nonneg=True)  # E[i,j,t] = i → j energia
        G = cp.Variable((N, T), nonneg=True)     # Városi áram
        
        charge = cp.Variable((N, T), nonneg=True)
        discharge = cp.Variable((N, T), nonneg=True)
        
        B = cp.Variable((N, T+1), nonneg=True)   # Akkumulátor szintek

        # Célfüggvény
        objective = cp.sum(cp.multiply(G, np.array(city_grid_prices_forecast)))
        
        # Korlátozások
        constraints = []
        
        # Kezdeti akkumulátor
        for i in range(N):
            constraints += [B[i,0] == households[i].battery.get_stored_kwh()]
        
        for t in range(T):
            for i in range(N):
                # Energiamérleg
                charge_efficiency: float = households[i].battery.get_fields().get("charge_efficiency") or 1.0
                discharge_efficiency: float = households[i].battery.get_fields().get("discharge_efficiency") or 1.0
                energy_in = forecasts[households[i].id]["production"][t] + cp.sum(E[:,i,t]) + discharge[i,t]*discharge_efficiency
                energy_out = forecasts[households[i].id]["consumption"][t] + cp.sum(E[i,:,t]) + G[i,t] + charge[i,t]/charge_efficiency
                constraints += [energy_in == energy_out]
                
                # Akkumulátor dinamika
                constraints += [B[i,t+1] == B[i,t] - discharge[i,t] + charge[i,t]]
                capacity_in_kwh: float = households[i].battery.get_fields().get("capacity_in_kwh") or 0
                constraints += [B[i,t+1] <= capacity_in_kwh]
                constraints += [B[i,t+1] >= 0]
                
                # Töltés/kisütés korlátok
                constraints += [charge[i,t] <= capacity_in_kwh - B[i,t]]
                constraints += [discharge[i,t] <= B[i,t]]

        # Optimalizáció
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.ECOS)
        
        # Eredmények formázása
        output = []
        for i in range(N):
            output.append({
                "id": i,
                "sells": sum(E[i,:,:].value.flatten()),
                "buys": [
                    {"from": j, "amount": E[j,i,t].value}
                    for j in range(N) for t in range(T) if E[j,i,t].value > 1e-4
                ],
                "buy_from_city": G[i,:].value.sum(),
                "used_accumulator": discharge[i,:].value.sum() * (households[i].battery.get_fields().get("discharge_efficiency") or 1.0),
                "remaining_accumulator": B[i,-1].value
            })
        
        return output
