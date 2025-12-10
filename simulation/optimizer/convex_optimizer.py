import numpy as np
import cvxpy as cp

from simulation.optimizer.optimizer import OptimizerStrategy
from simulation.household import Household
from simulation.local_price_estimator.price_estimator import PriceEstimator
from typing import Literal


SOLVER_MAP = {
    "CLARABEL": cp.CLARABEL,
    "ECOS": cp.ECOS,
    "OSQP": cp.OSQP,
    "SCS": cp.SCS,
    "COPT": cp.COPT,
}


class ConvexOptimizer(OptimizerStrategy):
    """
    Convex optimization-based energy trading optimizer.
    
    Uses CVXPY to solve a multi-timestep optimization problem that minimizes
    total community energy costs while respecting grid capacity constraints.
    """
    
    def __init__(
        self, 
        p2p_transaction_cost: float = 0.5, 
        min_trade_threshold: float = 0.1,
        solver: str = "CLARABEL",
        warm_start: bool = True
    ):
        """
        Initialize the optimizer.
        
        Args:
            p2p_transaction_cost: Fixed cost per P2P transaction pair (discourages small trades)
            min_trade_threshold: Minimum kWh for a trade to be included in output
            solver: Solver to use (CLARABEL, ECOS, OSQP, SCS, COPT)
            warm_start: Whether to warm-start from previous solution
        """
        self.p2p_transaction_cost = p2p_transaction_cost
        self.min_trade_threshold = min_trade_threshold
        self.solver_name = solver.upper()
        self.warm_start = warm_start
        
        # Cache for warm starting
        self._prev_solution: dict | None = None

    def optimize(
        self,
        households: list[Household],
        forecasts: dict[str, dict[Literal["production", "consumption"], list[float]]],
        city_grid_prices_forecast: list[float],
        city_grid_sell_prices_forecast: list[float],
        current_consumption: dict[str, float],
        current_production: dict[str, float],
        local_energy_price_estimator: PriceEstimator,
        grid_import_capacity_forecast: list[float] | None = None,
        grid_export_capacity_forecast: list[float] | None = None,
    ) -> list[dict]:

        N = len(households)
        T = len(city_grid_prices_forecast)

        # Default to unlimited capacity if not provided
        if grid_import_capacity_forecast is None:
            grid_import_capacity_forecast = [float('inf')] * T
        if grid_export_capacity_forecast is None:
            grid_export_capacity_forecast = [float('inf')] * T

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

            # Grid capacity constraints: total community import/export cannot exceed grid limits
            import_cap_t = grid_import_capacity_forecast[t]
            export_cap_t = grid_export_capacity_forecast[t]
            
            if import_cap_t != float('inf'):
                constraints += [cp.sum(G_buy[:, t]) <= import_cap_t]
            if export_cap_t != float('inf'):
                constraints += [cp.sum(G_sell[:, t]) <= export_cap_t]

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
                    # Discharge limited by available energy (accounting for efficiency loss)
                    B_discharge[i, t] / eta_discharge[i] <= B[i, t]
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

        # Objective: Minimize total cost (buying cost - selling revenue)
        # Terms:
        #   1. Grid buy cost: pay city_grid_prices to buy from grid
        #   2. Grid sell revenue: receive city_grid_sell_prices for selling (negative cost)
        #   3. P2P preference: small bonus for local trading vs grid (encourages community trading)
        #   4. Wallet penalty: heavily penalize negative wallet balances
        #   5. Transaction cost: penalty per P2P trade to discourage many small trades
        
        # Small incentive to prefer P2P over grid when prices are similar
        # (P2P is typically priced between grid buy and sell prices)
        p2p_incentive_weight = 0.01
        
        # Transaction cost penalty: use a smooth approximation to count non-zero trades
        # We penalize the sum of trades weighted by transaction cost
        # This encourages fewer, larger trades instead of many small ones
        # We use a soft-threshold approach: trades < threshold get proportionally less penalty
        transaction_cost_penalty = 0.0
        if self.p2p_transaction_cost > 0:
            # Penalize each trade proportionally, which discourages small trades
            # since the fixed cost makes them not worth it
            transaction_cost_penalty = self.p2p_transaction_cost * cp.sum(E) / (N * T + 1)
        
        objective = cp.Minimize(
            cp.sum(cp.multiply(G_buy, city_grid_prices_forecast))      # Cost of buying from grid
            - cp.sum(cp.multiply(G_sell, city_grid_sell_prices_forecast))  # Revenue from selling to grid
            - p2p_incentive_weight * cp.sum(E)                         # Small incentive for P2P trading
            + 100 * cp.sum(cp.pos(-wallet))                            # Penalty for negative balance
            + transaction_cost_penalty                                  # Transaction cost for P2P trades
        )

        problem = cp.Problem(objective, constraints)
        
        # Apply warm start if available and enabled
        if self.warm_start and self._prev_solution is not None:
            try:
                prev = self._prev_solution
                # Shift previous solution by 1 timestep for warm starting
                if prev.get("G_buy") is not None and prev["G_buy"].shape == G_buy.shape:
                    G_buy.value = prev["G_buy"]
                    G_sell.value = prev["G_sell"]
                    B.value = prev["B"]
                    E.value = prev["E"]
                    B_charge.value = prev["B_charge"]
                    B_discharge.value = prev["B_discharge"]
                    wallet.value = prev["wallet"]
            except Exception:
                pass  # Warm start failed, continue with cold start
        
        # Try the configured solver first, fall back to SCS if it fails
        solver = SOLVER_MAP.get(self.solver_name, cp.CLARABEL)
        solve_kwargs = {"verbose": False}
        
        # Add warm_start flag for solvers that support it
        if self.warm_start and self._prev_solution is not None:
            solve_kwargs["warm_start"] = True
        
        try:
            problem.solve(solver=solver, **solve_kwargs)
            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # Fall back to SCS (most robust)
                problem.solve(solver=cp.SCS, verbose=False)
        except Exception:
            # Fall back to SCS on any error
            problem.solve(solver=cp.SCS, verbose=False)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"Optimization failed: {problem.status}")
        
        # Save solution for warm-starting next iteration
        if self.warm_start:
            self._prev_solution = {
                "G_buy": G_buy.value,
                "G_sell": G_sell.value,
                "B": B.value,
                "E": E.value,
                "B_charge": B_charge.value,
                "B_discharge": B_discharge.value,
                "wallet": wallet.value,
            }

        # Only return t = 0 decisions
        output = []
        for i in range(N):
            p2p_buys = []

            for j in range(N):
                if i != j:
                    amount_bought = E[j, i, 0].value or 0.0
                    # Only include trades above the minimum threshold
                    if amount_bought > self.min_trade_threshold:
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