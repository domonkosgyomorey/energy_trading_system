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
        warm_start: bool = True,
        max_neighbors: int | None = None
    ):
        """
        Initialize the optimizer.
        
        Args:
            p2p_transaction_cost: Fixed cost per P2P transaction pair (discourages small trades)
            min_trade_threshold: Minimum kWh for a trade to be included in output
            solver: Solver to use (CLARABEL, ECOS, OSQP, SCS, COPT)
            warm_start: Whether to warm-start from previous solution
            max_neighbors: Maximum neighbors each household can trade with (None = all households)
        """
        self.p2p_transaction_cost = p2p_transaction_cost
        self.min_trade_threshold = min_trade_threshold
        self.solver_name = solver.upper()
        self.warm_start = warm_start
        self.max_neighbors = max_neighbors
        
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

        # Decision variables - SIMPLIFIED MODEL
        # Instead of E[i,j,t] matrix (N×N×T), use net_P2P[i,t] (N×T)
        # net_P2P > 0 means household is a SELLER
        # net_P2P < 0 means household is a BUYER
        # This naturally prevents simultaneous buy/sell!
        
        G_buy = cp.Variable((N, T), nonneg=True)      # Grid purchases
        G_sell = cp.Variable((N, T), nonneg=True)     # Grid sales
        B = cp.Variable((N, T + 1), nonneg=True)      # Battery state
        B_charge = cp.Variable((N, T), nonneg=True)   # Battery charging
        B_discharge = cp.Variable((N, T), nonneg=True) # Battery discharging
        net_P2P = cp.Variable((N, T))                  # Net P2P position (+ = sell, - = buy)
        wallet = cp.Variable((N, T + 1))

        constraints = [B[:, 0] == B0, wallet[:, 0] == wallets_0]

        for t in range(T):
            prod_t = P[:, t]
            cons_t = C[:, t]

            total_prod_t = float(np.sum(prod_t))
            total_cons_t = float(np.sum(cons_t))
            grid_buy_price_t = city_grid_prices_forecast[t]
            p2p_price_t = local_energy_price_estimator.calculate_price(
                total_prod_t, total_cons_t, grid_buy_price_t
            )

            # P2P MARKET BALANCE: Total sells == Total buys
            # Since net_P2P > 0 is sell and net_P2P < 0 is buy, sum must be 0
            constraints += [cp.sum(net_P2P[:, t]) == 0]

            # Grid capacity constraints
            import_cap_t = grid_import_capacity_forecast[t]
            export_cap_t = grid_export_capacity_forecast[t]
            
            if import_cap_t != float('inf'):
                constraints += [cp.sum(G_buy[:, t]) <= import_cap_t]
            if export_cap_t != float('inf'):
                constraints += [cp.sum(G_sell[:, t]) <= export_cap_t]

            for i in range(N):
                # Battery constraints
                constraints += [B[i, t + 1] <= B_max[i]]
                
                constraints += [
                    B[i, t + 1] == B[i, t] + B_charge[i, t] * eta_charge[i] - B_discharge[i, t] / eta_discharge[i]
                ]

                max_rate = B_max[i] * 0.5  # 50% max charge/discharge rate
                constraints += [
                    B_charge[i, t] <= max_rate,
                    B_discharge[i, t] <= max_rate,
                    B_discharge[i, t] / eta_discharge[i] <= B[i, t]
                ]

                # P2P sell limit: can't sell more than you produce + discharge
                constraints += [net_P2P[i, t] <= prod_t[i] + B_discharge[i, t]]
                
                # P2P buy limit: can't buy more than you need + can charge
                constraints += [-net_P2P[i, t] <= cons_t[i] + max_rate]

                # Grid sell limit: can only sell when not doing P2P sell
                # Use auxiliary variable to avoid cp.pos() in subtraction (DCP violation)
                # Simplified: just limit grid sell by production + discharge
                constraints += [G_sell[i, t] <= prod_t[i] + B_discharge[i, t]]

                # Energy balance (DCP-compliant formulation):
                # production + grid_buy + discharge = consumption + grid_sell + charge + net_P2P
                # (net_P2P > 0 means selling P2P, < 0 means buying P2P)
                constraints += [
                    prod_t[i] + G_buy[i, t] + B_discharge[i, t] == 
                    cons_t[i] + G_sell[i, t] + B_charge[i, t] + net_P2P[i, t]
                ]

                # Wallet update
                # P2P: If net_P2P > 0, we sell and earn. If net_P2P < 0, we buy and pay.
                # net_P2P * p2p_price gives us income if positive, cost if negative
                p2p_cashflow = net_P2P[i, t] * p2p_price_t
                city_buy_cost = G_buy[i, t] * city_grid_prices_forecast[t]
                city_sell_income = G_sell[i, t] * city_grid_sell_prices_forecast[t]

                constraints += [
                    wallet[i, t + 1] == wallet[i, t]
                                    - city_buy_cost
                                    + city_sell_income
                                    + p2p_cashflow  # + if selling, - if buying
                ]

        # Objective: Minimize costs
        # - Grid buy cost
        # - Grid sell revenue (negative cost)
        # - Small P2P incentive to prefer local trading
        # - Penalty for negative wallet
        # - Transaction cost for P2P (penalize volume to discourage tiny trades)
        
        p2p_incentive_weight = 0.01
        transaction_cost_penalty = 0.0
        if self.p2p_transaction_cost > 0:
            # Penalize absolute P2P volume - this discourages small trades
            transaction_cost_penalty = self.p2p_transaction_cost * cp.sum(cp.abs(net_P2P)) / (N * T + 1)
        
        objective = cp.Minimize(
            cp.sum(cp.multiply(G_buy, city_grid_prices_forecast))      # Cost of buying from grid
            - cp.sum(cp.multiply(G_sell, city_grid_sell_prices_forecast))  # Revenue from selling to grid
            + p2p_incentive_weight * cp.sum(cp.abs(net_P2P))           # Small incentive for P2P trading
            + 100 * cp.sum(cp.pos(-wallet))                            # Penalty for negative balance
            + transaction_cost_penalty                                  # Transaction cost for P2P trades
        )

        problem = cp.Problem(objective, constraints)
        
        # Apply warm start if available and enabled
        if self.warm_start and self._prev_solution is not None:
            try:
                prev = self._prev_solution
                if prev.get("G_buy") is not None and prev["G_buy"].shape == G_buy.shape:
                    G_buy.value = prev["G_buy"]
                    G_sell.value = prev["G_sell"]
                    B.value = prev["B"]
                    net_P2P.value = prev["net_P2P"]
                    B_charge.value = prev["B_charge"]
                    B_discharge.value = prev["B_discharge"]
                    wallet.value = prev["wallet"]
            except Exception:
                pass  # Warm start failed, continue with cold start
        
        # Try the configured solver first, fall back to SCS if it fails
        solver = SOLVER_MAP.get(self.solver_name, cp.CLARABEL)
        solve_kwargs = {"verbose": False}
        
        if self.warm_start and self._prev_solution is not None:
            solve_kwargs["warm_start"] = True
        
        try:
            problem.solve(solver=solver, **solve_kwargs)
            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                problem.solve(solver=cp.SCS, verbose=False)
        except Exception:
            problem.solve(solver=cp.SCS, verbose=False)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"Optimization failed: {problem.status}")
        
        # Save solution for warm-starting next iteration
        if self.warm_start:
            self._prev_solution = {
                "G_buy": G_buy.value,
                "G_sell": G_sell.value,
                "B": B.value,
                "net_P2P": net_P2P.value,
                "B_charge": B_charge.value,
                "B_discharge": B_discharge.value,
                "wallet": wallet.value,
            }

        # Build output from net_P2P values at t=0
        # Match sellers with buyers using greedy algorithm
        net_p2p_t0 = net_P2P[:, 0].value if net_P2P.value is not None else np.zeros(N)
        
        # Separate sellers and buyers
        sellers = [(i, net_p2p_t0[i]) for i in range(N) if net_p2p_t0[i] > self.min_trade_threshold]
        buyers = [(i, -net_p2p_t0[i]) for i in range(N) if net_p2p_t0[i] < -self.min_trade_threshold]
        
        # Sort by amount (largest first)
        sellers.sort(key=lambda x: -x[1])
        buyers.sort(key=lambda x: -x[1])
        
        # Greedy matching: build trades dict
        trades: dict[int, list[tuple[int, float]]] = {i: [] for i in range(N)}  # buyer_idx -> [(seller_idx, amount)]
        remaining_supply = {i: amt for i, amt in sellers}
        remaining_demand = {i: amt for i, amt in buyers}
        
        # Create neighbor map for topology constraints
        neighbor_map = self._create_neighbor_map(N)
        
        for buyer_idx, _ in buyers:
            demand = remaining_demand.get(buyer_idx, 0)
            if demand < self.min_trade_threshold:
                continue
            
            for seller_idx, _ in sellers:
                supply = remaining_supply.get(seller_idx, 0)
                if supply < self.min_trade_threshold:
                    continue
                
                # Check if they're neighbors (if max_neighbors is set)
                if self.max_neighbors is not None and not neighbor_map[buyer_idx, seller_idx]:
                    continue
                
                trade_amount = min(supply, demand)
                if trade_amount >= self.min_trade_threshold:
                    trades[buyer_idx].append((seller_idx, trade_amount))
                    remaining_supply[seller_idx] -= trade_amount
                    remaining_demand[buyer_idx] -= trade_amount
                    demand = remaining_demand[buyer_idx]
                
                if demand < self.min_trade_threshold:
                    break
        
        # Build output
        output = []
        for i in range(N):
            p2p_buys = [
                {"from": households[seller_idx].id, "amount": float(amount)}
                for seller_idx, amount in trades[i]
            ]
            
            output.append({
                "id": households[i].id,
                "buy_from_city": float(G_buy[i, 0].value or 0),
                "sell_to_city": float(G_sell[i, 0].value or 0),
                "buys": p2p_buys,
            })

        return output
    
    def _create_neighbor_map(self, N: int) -> np.ndarray:
        """
        Create a boolean NxN matrix indicating which households can trade with each other.
        
        If max_neighbors is None, all households can trade (full connectivity).
        If max_neighbors is set, each household can only trade with K nearest neighbors.
        
        Uses a circular/ring topology for simplicity: household i can trade with
        households (i-K//2) to (i+K//2) in the list, wrapping around.
        
        Args:
            N: Number of households
            
        Returns:
            neighbor_map: Boolean array of shape (N, N) where neighbor_map[i,j] = True
                         means household i can trade with household j
        """
        neighbor_map = np.zeros((N, N), dtype=bool)
        
        if self.max_neighbors is None or self.max_neighbors >= N - 1:
            # Full connectivity: everyone can trade with everyone
            neighbor_map = np.ones((N, N), dtype=bool)
            np.fill_diagonal(neighbor_map, False)  # Can't trade with self
        else:
            # Limited connectivity: each household trades with K neighbors
            K = min(self.max_neighbors, N - 1)
            half_K = K // 2
            
            for i in range(N):
                # Add neighbors on both sides in circular fashion
                for offset in range(1, half_K + 1):
                    # Left neighbors
                    left_neighbor = (i - offset) % N
                    neighbor_map[i, left_neighbor] = True
                    
                    # Right neighbors
                    right_neighbor = (i + offset) % N
                    neighbor_map[i, right_neighbor] = True
                
                # If K is odd, add one more neighbor to reach exactly K neighbors
                if K % 2 == 1 and K < N - 1:
                    extra_neighbor = (i + half_K + 1) % N
                    neighbor_map[i, extra_neighbor] = True
        
        return neighbor_map