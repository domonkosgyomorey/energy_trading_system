import numpy as np
import cvxpy as cp
from simulation.config import Config
from simulation.optimizer.optimizer import OptimizerStrategy
from simulation.household import Household
from typing import Literal

class SimpleRuleBasedOptimizer(OptimizerStrategy):
    
    def optimize(self, households: list[Household], forecasts: dict[str, dict[Literal["production", "consumption"], list[float]]], city_grid_prices_forecast: list[float]) -> list[dict]:
        N = len(households)
        T = Config.FORECASTER_PRED_SIZE
        
        # Változók definiálása
        G = cp.Variable((N, T), nonneg=True)  # Gridből vett energia
        B = cp.Variable((N, T+1), nonneg=True)  # Akkumulátor állapot
        charge = cp.Variable((N, T), nonneg=True)
        discharge = cp.Variable((N, T), nonneg=True)
        P2P_in = cp.Variable((N, T), nonneg=True)  # P2P-ből vett energia
        P2P_out = cp.Variable((N, T), nonneg=True)  # P2P-be adott energia
        
        # Célfüggvény: Minimális grid költség
        objective = cp.sum(cp.multiply(G, city_grid_prices_forecast))
        
        constraints = []
        
        # Kezdeti akkumulátor feltöltés
        for i in range(N):
            constraints += [B[i,0] == households[i].battery.get_stored_kwh()]
        
        # Pool egyensúly: Összes P2P_in = Összes P2P_out
        for t in range(T):
            constraints += [cp.sum(P2P_in[:,t]) == cp.sum(P2P_out[:,t])]
        
        for t in range(T):
            for i in range(N):
                prod = forecasts[households[i].id]["production"][t]
                cons = forecasts[households[i].id]["consumption"][t]
                capacity = households[i].battery.get_fields().get("capacity_in_kwh", 0)
                
                # Energiamérleg újratervézve P2P-vel
                constraints += [
                    G[i,t] + discharge[i,t] + P2P_in[i,t] == cons - prod + charge[i,t] + P2P_out[i,t],
                    B[i,t+1] == B[i,t] + charge[i,t] - discharge[i,t],
                    B[i,t+1] <= capacity,
                    charge[i,t] <= capacity - B[i,t],
                    discharge[i,t] <= B[i,t]
                ]
        
        # Optimalizáció végrehajtása
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.ECOS, verbose=False)
        
        # Eredmények formázása P2P kereskedelem részletezésével
        output = []
        for i in range(N):
            p2p_transactions = []
            for j in range(N):
                if i != j:
                    total_energy = sum(P2P_out[j,t].value * (P2P_in[i,t].value/(cp.sum(P2P_in[:,t]).value + 1e-6)) 
                                     for t in range(T) if cp.sum(P2P_in[:,t]).value > 1e-6)
                    if total_energy > 1e-4:
                        p2p_transactions.append({"from": households[j].id, "amount": total_energy})
            
            output.append({
                "id": households[i].id,
                "sells": np.sum(P2P_out[i,:].value),
                "buys": p2p_transactions,
                "buy_from_city": np.sum(G[i,:].value),
                "used_accumulator": np.sum(discharge[i,:].value),
                "remaining_accumulator": B[i,-1].value
            })
        return output
