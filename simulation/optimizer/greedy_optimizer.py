"""
Greedy/Heuristic optimizer for P2P energy trading.

This optimizer uses a fast greedy algorithm instead of convex optimization.
It's much faster (O(N log N) vs O(N²T)) and naturally enforces realistic constraints
like "a household cannot simultaneously buy and sell".

Algorithm:
1. For each timestep, calculate net energy position for each household
2. Separate into sellers (surplus) and buyers (deficit)
3. Match sellers with buyers greedily, prioritizing neighbors
4. Handle remaining deficit/surplus through grid or battery
"""

import numpy as np
from typing import Literal

from simulation.optimizer.optimizer import OptimizerStrategy
from simulation.household import Household
from simulation.local_price_estimator.price_estimator import PriceEstimator


class GreedyOptimizer(OptimizerStrategy):
    """
    Fast greedy optimizer for P2P energy trading.
    
    Advantages over ConvexOptimizer:
    - Much faster: O(N log N) per timestep
    - Naturally prevents simultaneous buy/sell (each household is either seller OR buyer)
    - Simpler to understand and debug
    - Works well for real-time/large-scale simulations
    
    Trade-offs:
    - May not find globally optimal solution
    - Doesn't look ahead at future timesteps
    - Simpler battery strategy
    """
    
    def __init__(
        self,
        p2p_transaction_cost: float = 0.25,
        min_trade_threshold: float = 0.5,
        max_neighbors: int | None = None,
        battery_target_pct: float = 0.5,  # Target battery level (50%)
        p2p_price_factor: float = 0.80,
    ):
        """
        Initialize greedy optimizer.
        
        Args:
            p2p_transaction_cost: Minimum savings required for P2P trade to be worthwhile
            min_trade_threshold: Minimum kWh for a trade to count
            max_neighbors: Max neighbors to trade with (None = all)
            battery_target_pct: Target battery level to maintain (0-1)
            p2p_price_factor: P2P price as fraction of grid buy price
        """
        self.p2p_transaction_cost = p2p_transaction_cost
        self.min_trade_threshold = min_trade_threshold
        self.max_neighbors = max_neighbors
        self.battery_target_pct = battery_target_pct
        self.p2p_price_factor = p2p_price_factor
    
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
        """
        Run greedy optimization for current timestep.
        
        Returns decisions only for t=0 (current timestep).
        """
        N = len(households)
        
        if N == 0:
            return []
        
        grid_buy_price = city_grid_prices_forecast[0]
        grid_sell_price = city_grid_sell_prices_forecast[0]
        
        # Calculate P2P price
        total_prod = sum(current_production.values())
        total_cons = sum(current_consumption.values())
        p2p_price = local_energy_price_estimator.calculate_price(
            total_prod, total_cons, grid_buy_price
        )
        
        # Create neighbor map for topology constraints
        neighbor_map = self._create_neighbor_map(N)
        
        # Build index mapping
        id_to_idx = {hh.id: i for i, hh in enumerate(households)}
        
        # Step 1: Calculate net position for each household
        household_data: list[dict[str, any]] = []
        for i, hh in enumerate(households):
            prod = current_production[hh.id]
            cons = current_consumption[hh.id]
            
            battery_stored = hh.battery.get_fields().get("stored_kwh", 0.0)
            battery_capacity = hh.battery.get_capacity_in_kwh()
            battery_target = battery_capacity * self.battery_target_pct
            max_charge_rate = battery_capacity * 0.5  # 50% per hour max
            
            # Net energy = production - consumption
            net_energy = prod - cons
            
            # Adjust for battery strategy
            # If net positive and battery below target, charge first
            # If net negative and battery above minimum, discharge first
            battery_action = 0.0  # positive = charge, negative = discharge
            
            if net_energy > 0:
                # Surplus energy - consider charging battery
                charge_room = battery_capacity - battery_stored
                desired_charge = min(net_energy, charge_room, max_charge_rate)
                if battery_stored < battery_target:
                    # Below target, charge up to target
                    desired_charge = min(desired_charge, battery_target - battery_stored)
                else:
                    # Above target, don't charge aggressively
                    desired_charge = 0.0
                battery_action = desired_charge
                net_energy -= desired_charge
            else:
                # Deficit energy - consider discharging battery
                deficit = -net_energy
                # Only discharge if above minimum reserve (20%)
                min_reserve = battery_capacity * 0.2
                available_discharge = max(0, battery_stored - min_reserve)
                discharge = min(deficit, available_discharge, max_charge_rate)
                battery_action = -discharge
                net_energy += discharge
            
            household_data.append({
                "idx": i,
                "id": hh.id,
                "net_energy": net_energy,  # positive = seller, negative = buyer
                "battery_action": battery_action,
                "wallet": hh.wallet,
            })
        
        # Step 2: Separate into sellers and buyers
        sellers = [h for h in household_data if float(h["net_energy"]) > self.min_trade_threshold]
        buyers = [h for h in household_data if float(h["net_energy"]) < -self.min_trade_threshold]
        
        # Sort sellers by surplus (largest first) and buyers by deficit (largest first)
        sellers.sort(key=lambda x: -float(x["net_energy"]))
        buyers.sort(key=lambda x: float(x["net_energy"]))  # Most negative first
        
        # Step 3: Match sellers with buyers (greedy matching)
        trades: dict[str, list[tuple[str, float]]] = {hh.id: [] for hh in households}
        
        remaining_supply: dict[str, float] = {s["id"]: float(s["net_energy"]) for s in sellers}
        remaining_demand: dict[str, float] = {b["id"]: float(-b["net_energy"]) for b in buyers}
        
        # Check if P2P is worthwhile (P2P price should be between grid sell and buy)
        p2p_worthwhile = grid_sell_price < p2p_price < grid_buy_price
        
        if p2p_worthwhile:
            for buyer in buyers:
                buyer_id = str(buyer["id"])
                buyer_idx = int(buyer["idx"])
                demand = remaining_demand[buyer_id]
                
                if demand < self.min_trade_threshold:
                    continue
                
                # Find sellers this buyer can trade with (respecting topology)
                for seller in sellers:
                    seller_id = str(seller["id"])
                    seller_idx = int(seller["idx"])
                    supply = remaining_supply[seller_id]
                    
                    if supply < self.min_trade_threshold:
                        continue
                    
                    # Check if they're neighbors
                    if not neighbor_map[buyer_idx, seller_idx]:
                        continue
                    
                    # Calculate trade amount
                    trade_amount = min(supply, demand)
                    
                    if trade_amount >= self.min_trade_threshold:
                        trades[buyer_id].append((seller_id, trade_amount))
                        remaining_supply[seller_id] -= trade_amount
                        remaining_demand[buyer_id] -= trade_amount
                        demand = remaining_demand[buyer_id]
                    
                    if demand < self.min_trade_threshold:
                        break
        
        # Step 4: Build output - remaining deficit/surplus goes to grid
        output = []
        
        for hh_data in household_data:
            hh_id = str(hh_data["id"])
            net_energy = float(hh_data["net_energy"])
            
            # Calculate P2P amounts
            p2p_buys = []
            p2p_buy_total = 0.0
            for seller_id, amount in trades[hh_id]:
                p2p_buys.append({"from": seller_id, "amount": float(amount)})
                p2p_buy_total += amount
            
            # Calculate grid transactions
            if net_energy > 0:
                # Seller - sell remaining to grid
                remaining_surplus = remaining_supply.get(hh_id, net_energy)
                sell_to_city = max(0.0, remaining_surplus)
                buy_from_city = 0.0
            else:
                # Buyer - buy remaining from grid
                remaining_deficit = remaining_demand.get(hh_id, -net_energy)
                buy_from_city = max(0.0, remaining_deficit)
                sell_to_city = 0.0
            
            output.append({
                "id": hh_id,
                "buy_from_city": float(buy_from_city),
                "sell_to_city": float(sell_to_city),
                "buys": p2p_buys,
            })
        
        return output
    
    def _create_neighbor_map(self, N: int) -> np.ndarray:
        """Create neighbor connectivity map (same as ConvexOptimizer)."""
        neighbor_map = np.zeros((N, N), dtype=bool)
        
        if self.max_neighbors is None or self.max_neighbors >= N - 1:
            neighbor_map = np.ones((N, N), dtype=bool)
            np.fill_diagonal(neighbor_map, False)
        else:
            K = min(self.max_neighbors, N - 1)
            half_K = K // 2
            
            for i in range(N):
                for offset in range(1, half_K + 1):
                    left_neighbor = (i - offset) % N
                    neighbor_map[i, left_neighbor] = True
                    right_neighbor = (i + offset) % N
                    neighbor_map[i, right_neighbor] = True
                
                if K % 2 == 1 and K < N - 1:
                    extra_neighbor = (i + half_K + 1) % N
                    neighbor_map[i, extra_neighbor] = True
        
        return neighbor_map
