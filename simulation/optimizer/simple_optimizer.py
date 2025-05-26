from typing import Literal
from simulation.optimizer.optimizer import OptimizerStrategy
from simulation.household import Household

class RuleBasedOptimizer(OptimizerStrategy):
    def optimize(self, households: list[Household], 
                forecasts: dict[str, dict[Literal["production", "consumption"], list[float]]], 
                city_grid_prices_forecast: list[float],
                current_consumption: dict[str, float]) -> list[dict]:
       
        
        current_step = 0
        output = []
        
        # First calculate net energy for each household
        household_energy = {}
        for hh in households:
            current_prod = forecasts[hh.id]["production"][current_step]
            current_cons = current_consumption[hh.id]
            battery_kwh = hh.battery.get_stored_kwh()
            battery_capacity = hh.battery.get_capacity_in_kwh()
            
            # Max that can be discharged (20% of capacity)
            max_discharge = min(battery_kwh, battery_capacity * 0.2)
            net_energy = current_prod + max_discharge - current_cons
            household_energy[hh.id] = net_energy
        
        # Create producer and consumer lists
        producers = []
        consumers = []
        
        for hh in households:
            energy = household_energy[hh.id]
            if energy > 0:
                producers.append((hh.id, energy))
            elif energy < 0:
                consumers.append((hh.id, -energy))
        
        # Sort producers by highest energy first
        producers.sort(key=lambda x: -x[1])
        
        # Process each consumer
        for consumer_id, deficit in consumers:
            buys = []
            remaining_deficit = deficit
            
            # Try to buy from producers
            for i, (producer_id, surplus) in enumerate(producers):
                if remaining_deficit <= 0:
                    break
                
                if surplus > 0:
                    trade_amount = min(surplus, remaining_deficit)
                    buys.append({
                        "from": producer_id,
                        "amount": trade_amount
                    })
                    
                    # Update remaining amounts
                    remaining_deficit -= trade_amount
                    producers[i] = (producer_id, surplus - trade_amount)
            
            # Buy remaining from city
            buy_from_city = max(remaining_deficit, 0)
            
            output.append({
                "id": consumer_id,
                "sells": 0,  # Will be updated later
                "buys": buys,
                "buy_from_city": buy_from_city
            })
        
        # Calculate sales for each producer
        sales = {producer_id: 0 for producer_id, _ in producers}
        for hh_output in output:
            for buy in hh_output["buys"]:
                sales[buy["from"]] += buy["amount"]
        
        # Add producers to output
        for producer_id, remaining_surplus in producers:
            output.append({
                "id": producer_id,
                "sells": sales.get(producer_id, 0),
                "buys": [],
                "buy_from_city": 0
            })
        
        return output