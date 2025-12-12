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
        smart_battery_strategy: bool = True,  # Use smart battery strategy with lookahead
        prefer_p2p_over_grid: bool = True,  # Prefer P2P trades to reduce grid dependency
    ):
        """
        Initialize greedy optimizer.
        
        Args:
            p2p_transaction_cost: Minimum savings required for P2P trade to be worthwhile
            min_trade_threshold: Minimum kWh for a trade to count
            max_neighbors: Max neighbors to trade with (None = all)
            battery_target_pct: Target battery level to maintain (0-1)
            p2p_price_factor: P2P price as fraction of grid buy price
            smart_battery_strategy: Use lookahead to optimize battery usage
            prefer_p2p_over_grid: Prioritize P2P trades over grid transactions
        """
        self.p2p_transaction_cost = p2p_transaction_cost
        self.min_trade_threshold = min_trade_threshold
        self.max_neighbors = max_neighbors
        self.battery_target_pct = battery_target_pct
        self.p2p_price_factor = p2p_price_factor
        self.smart_battery_strategy = smart_battery_strategy
        self.prefer_p2p_over_grid = prefer_p2p_over_grid
    
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
        
        # Analyze future prices to make smarter battery decisions
        avg_future_buy_price = np.mean(city_grid_prices_forecast[:min(3, len(city_grid_prices_forecast))])
        avg_future_sell_price = np.mean(city_grid_sell_prices_forecast[:min(3, len(city_grid_sell_prices_forecast))])
        current_buy_is_cheap = grid_buy_price < avg_future_buy_price * 0.95
        current_sell_is_expensive = grid_sell_price > avg_future_sell_price * 1.05
        
        # Step 1: Calculate net position for each household with smart battery strategy
        household_data: list[dict[str, any]] = []
        for i, hh in enumerate(households):
            prod = current_production[hh.id]
            cons = current_consumption[hh.id]
            
            battery_stored = hh.battery.get_fields().get("stored_kwh", 0.0)
            battery_capacity = hh.battery.get_capacity_in_kwh()
            battery_charge_eff = hh.battery.get_fields().get("charge_efficiency", 0.9)
            battery_discharge_eff = hh.battery.get_fields().get("discharge_efficiency", 0.9)
            max_charge_rate = battery_capacity * 0.5  # 50% per hour max
            
            # Net energy = production - consumption
            net_energy = prod - cons
            
            # Smart battery strategy based on lookahead and current conditions
            battery_action = 0.0  # positive = charge, negative = discharge
            
            if self.smart_battery_strategy and len(forecasts[hh.id]["production"]) > 0:
                # Look ahead to next few timesteps
                future_prod = forecasts[hh.id]["production"][:3]
                future_cons = forecasts[hh.id]["consumption"][:3]
                future_net = [p - c for p, c in zip(future_prod, future_cons)]
                avg_future_net = np.mean(future_net) if future_net else 0.0
                
                # Smart battery decision based on current and future conditions
                if net_energy > 0:
                    # Surplus energy
                    charge_room = battery_capacity - battery_stored
                    max_possible_charge = min(net_energy, charge_room, max_charge_rate)
                    
                    # Charge more aggressively if:
                    # 1. Future looks like we'll need it (future_net < 0)
                    # 2. Current prices are good (cheap to buy or expensive to sell)
                    # 3. Battery is below target
                    charge_aggressiveness = 0.5  # Default: moderate charging
                    
                    if avg_future_net < -2.0:  # Will need energy soon
                        charge_aggressiveness = 1.0  # Charge as much as possible
                    elif battery_stored < battery_capacity * 0.3:  # Battery very low
                        charge_aggressiveness = 0.8
                    elif current_sell_is_expensive and grid_sell_price > p2p_price * 1.1:
                        # Selling to grid is very profitable now, don't charge much
                        charge_aggressiveness = 0.2
                    elif battery_stored < battery_capacity * self.battery_target_pct:
                        charge_aggressiveness = 0.7
                    else:
                        charge_aggressiveness = 0.3  # Battery above target, charge less
                    
                    battery_action = max_possible_charge * charge_aggressiveness
                    net_energy -= battery_action
                    
                else:
                    # Deficit energy
                    deficit = -net_energy
                    min_reserve = battery_capacity * 0.15  # Keep 15% minimum
                    available_discharge = max(0, battery_stored - min_reserve)
                    max_possible_discharge = min(deficit, available_discharge, max_charge_rate)
                    
                    # Discharge more aggressively if:
                    # 1. Future looks good (will produce surplus)
                    # 2. Current grid prices are high
                    # 3. P2P buying is expensive
                    discharge_aggressiveness = 0.7  # Default: moderate discharge
                    
                    if avg_future_net > 2.0:  # Will have surplus soon
                        discharge_aggressiveness = 1.0  # Use battery fully
                    elif grid_buy_price > avg_future_buy_price * 1.1:  # Grid expensive now
                        discharge_aggressiveness = 1.0
                    elif p2p_price > grid_buy_price * 0.9:  # P2P not much cheaper
                        discharge_aggressiveness = 0.9
                    elif battery_stored > battery_capacity * 0.7:  # Battery very full
                        discharge_aggressiveness = 0.8
                    else:
                        discharge_aggressiveness = 0.6
                    
                    battery_action = -max_possible_discharge * discharge_aggressiveness
                    net_energy += max_possible_discharge * discharge_aggressiveness
                    
            else:
                # Fallback to simple target-based strategy
                battery_target = battery_capacity * self.battery_target_pct
                
                if net_energy > 0:
                    charge_room = battery_capacity - battery_stored
                    desired_charge = min(net_energy, charge_room, max_charge_rate)
                    if battery_stored < battery_target:
                        desired_charge = min(desired_charge, battery_target - battery_stored)
                    else:
                        desired_charge = 0.0
                    battery_action = desired_charge
                    net_energy -= desired_charge
                else:
                    deficit = -net_energy
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
        
        # Step 3: Match sellers with buyers (smart greedy matching)
        trades: dict[str, list[tuple[str, float]]] = {hh.id: [] for hh in households}
        
        remaining_supply: dict[str, float] = {s["id"]: float(s["net_energy"]) for s in sellers}
        remaining_demand: dict[str, float] = {b["id"]: float(-b["net_energy"]) for b in buyers}
        
        # Calculate P2P savings vs grid
        p2p_buyer_savings = grid_buy_price - p2p_price  # How much buyer saves
        p2p_seller_gain = p2p_price - grid_sell_price  # How much seller gains vs selling to grid
        total_p2p_benefit = p2p_buyer_savings + p2p_seller_gain
        
        # P2P is worthwhile if both parties benefit and total benefit > transaction cost
        p2p_worthwhile = (
            total_p2p_benefit > self.p2p_transaction_cost and
            p2p_buyer_savings > 0.001 and  # Buyer must save at least 0.1 cents
            p2p_seller_gain > 0.001  # Seller must gain at least 0.1 cents
        )
        
        # If prefer_p2p_over_grid, relax the constraints slightly
        if self.prefer_p2p_over_grid and not p2p_worthwhile:
            p2p_worthwhile = (
                grid_sell_price < p2p_price < grid_buy_price and
                total_p2p_benefit > 0
            )
        
        if p2p_worthwhile:
            # Smart matching: prioritize larger trades first (more efficient)
            # Create list of all possible trades with their benefits
            potential_trades = []
            for buyer in buyers:
                buyer_id = str(buyer["id"])
                buyer_idx = int(buyer["idx"])
                demand = remaining_demand[buyer_id]
                
                if demand < self.min_trade_threshold:
                    continue
                
                for seller in sellers:
                    seller_id = str(seller["id"])
                    seller_idx = int(seller["idx"])
                    supply = remaining_supply[seller_id]
                    
                    if supply < self.min_trade_threshold:
                        continue
                    
                    # Check if they're neighbors
                    if not neighbor_map[buyer_idx, seller_idx]:
                        continue
                    
                    # Calculate potential trade amount and benefit
                    trade_amount = min(supply, demand)
                    if trade_amount >= self.min_trade_threshold:
                        trade_benefit = trade_amount * total_p2p_benefit
                        potential_trades.append({
                            "buyer_id": buyer_id,
                            "seller_id": seller_id,
                            "amount": trade_amount,
                            "benefit": trade_benefit,
                        })
            
            # Sort by benefit (largest first) to prioritize most beneficial trades
            potential_trades.sort(key=lambda x: -x["benefit"])
            
            # Execute trades in order of benefit
            for trade in potential_trades:
                buyer_id = trade["buyer_id"]
                seller_id = trade["seller_id"]
                
                # Recalculate actual available amounts
                current_supply = remaining_supply.get(seller_id, 0)
                current_demand = remaining_demand.get(buyer_id, 0)
                
                if current_supply < self.min_trade_threshold or current_demand < self.min_trade_threshold:
                    continue
                
                # Execute trade with updated amounts
                trade_amount = min(current_supply, current_demand)
                
                if trade_amount >= self.min_trade_threshold:
                    trades[buyer_id].append((seller_id, trade_amount))
                    remaining_supply[seller_id] -= trade_amount
                    remaining_demand[buyer_id] -= trade_amount
        
        # Step 4: Build output - remaining deficit/surplus goes to grid (respecting capacity)
        output = []
        
        # Track total grid usage to respect capacity constraints
        total_grid_buy = 0.0
        total_grid_sell = 0.0
        grid_import_capacity = grid_import_capacity_forecast[0] if grid_import_capacity_forecast else float('inf')
        grid_export_capacity = grid_export_capacity_forecast[0] if grid_export_capacity_forecast else float('inf')
        
        # First pass: calculate all grid transactions
        grid_transactions = []
        for hh_data in household_data:
            hh_id = str(hh_data["id"])
            net_energy = float(hh_data["net_energy"])
            
            # Calculate P2P amounts
            p2p_buy_total = sum(amount for _, amount in trades[hh_id])
            
            # Calculate grid transactions needed
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
            
            grid_transactions.append({
                "hh_id": hh_id,
                "buy_from_city": buy_from_city,
                "sell_to_city": sell_to_city,
            })
            
            total_grid_buy += buy_from_city
            total_grid_sell += sell_to_city
        
        # Second pass: scale down if exceeding capacity
        if total_grid_buy > grid_import_capacity:
            scale_factor = grid_import_capacity / total_grid_buy
            for txn in grid_transactions:
                txn["buy_from_city"] *= scale_factor
        
        if total_grid_sell > grid_export_capacity:
            scale_factor = grid_export_capacity / total_grid_sell
            for txn in grid_transactions:
                txn["sell_to_city"] *= scale_factor
        
        # Build final output
        for hh_data, txn in zip(household_data, grid_transactions):
            hh_id = str(hh_data["id"])
            
            # Calculate P2P buys
            p2p_buys = []
            for seller_id, amount in trades[hh_id]:
                p2p_buys.append({"from": seller_id, "amount": float(amount)})
            
            output.append({
                "id": hh_id,
                "buy_from_city": float(txn["buy_from_city"]),
                "sell_to_city": float(txn["sell_to_city"]),
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
