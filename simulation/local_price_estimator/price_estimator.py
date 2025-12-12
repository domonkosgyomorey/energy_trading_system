class PriceEstimator():
    """
    Estimates P2P energy price based on grid prices and supply/demand dynamics.
    
    P2P price is calculated as:
    - Base price: grid_buy_price * p2p_price_factor (e.g., 80% of grid buy price)
    - Adjusted by supply/demand ratio to incentivize trades when needed
    """
    
    def __init__(self, p2p_price_factor: float = 0.75):
        """
        Initialize price estimator.
        
        Args:
            p2p_price_factor: P2P base price as fraction of grid buy price (0.5-0.95 recommended)
                             Lower values make P2P more attractive vs buying from grid
        """
        self.p2p_price_factor = p2p_price_factor

    def calculate_price(
        self, 
        total_production: float, 
        total_consumption: float,
        grid_buy_price: float
    ) -> float:
        """
        Calculate P2P energy price.
        
        Args:
            total_production: Total community production (kWh)
            total_consumption: Total community consumption (kWh)
            grid_buy_price: Current grid purchase price ($/kWh)
            
        Returns:
            P2P price ($/kWh) that's attractive compared to grid, adjusted by supply/demand
        """
        # Base P2P price as percentage of grid buy price
        base_p2p_price = grid_buy_price * self.p2p_price_factor
        
        # Adjust based on supply/demand balance
        if total_production <= 0:
            # No local production - P2P price approaches grid price
            return grid_buy_price * 0.95
        
        # Supply/demand ratio: >1 means surplus, <1 means deficit
        supply_demand_ratio = total_production / max(total_consumption, 0.01)
        
        # Adjustment factor: 
        # - Lots of supply (ratio > 2): decrease price by 20% (0.8x)
        # - Balanced (ratio = 1): no adjustment (1.0x)
        # - Shortage (ratio < 0.5): increase price by 20% (1.2x)
        if supply_demand_ratio > 2.0:
            adjustment = 0.8
        elif supply_demand_ratio > 1.5:
            adjustment = 0.9
        elif supply_demand_ratio > 0.75:
            adjustment = 1.0
        elif supply_demand_ratio > 0.5:
            adjustment = 1.1
        else:
            adjustment = 1.2
        
        final_price = base_p2p_price * adjustment
        
        # Ensure P2P price stays reasonable (between 50% and 95% of grid buy price)
        min_price = grid_buy_price * 0.5
        max_price = grid_buy_price * 0.95
        
        return max(min_price, min(final_price, max_price))
