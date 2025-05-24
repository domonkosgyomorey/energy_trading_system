from collections import defaultdict
from simulation.household import Household
from simulation.blockchain import Blockchain
from simulation.forecaster.perfect_forecaster import PerfectForecaster
from simulation.forecaster.forecaster import Forecaster
from simulation.optimizer.simple_optimizer import SimpleRuleBasedOptimizer
from simulation.optimizer.optimizer import OptimizerStrategy
from simulation.battery.simple_battery import SimpleBattery
from simulation.battery.central_battery import CentralBattery
from simulation.battery.shared_battery import SharedBattery 
from simulation.token_to_battery import TokenToBattery
from simulation.config import Config
from simulation.city_grid_price_forecaster.simple_city_grid_price_forecaster import SimpleCityGridPriceForecaster
from simulation.city_grid_price_forecaster.city_gird_price_forecaster import CityGridPriceForecaster
import json
import pandas as pd
from typing import Literal

class Simulation: 
    def __init__(self, household_data: pd.DataFrame):
        """
        Description: Simulates the automatic energy trading with blockchain.

        Args:
            household_data (pd.DataFrame): Database for the household production, consumption data
            household_without_battery_prob (float): From this, the simulation calculates how many household will have battery
            forecaster_pred_range (int): The forecaster will predict prod and cons this much further
        """

        self.central_battery: CentralBattery = CentralBattery(capacity_in_kwh=10000000.0, charge_efficiency=0.999, discharge_efficiency=0.999, tax_per_kwh=0.01)
        self.households: list[Household] = []
         
        grouped = household_data.groupby("id")
        iteration_counter: int = 0
        no_battery_iter: int = int(grouped.ngroups * Config.HOUSEHOLD_WITHOUT_BATTERY_PROB)

        for household_id, group in grouped:
            group = group.drop(columns="id")
            data_dict = {col: group[col].to_list() for col in group.columns}
            
            if iteration_counter % no_battery_iter == 0:
                self.households.append(Household(id=str(household_id), data=data_dict, battery=SharedBattery(central_battery=self.central_battery,household_id=str(household_id))))
            else:
                self.households.append(Household(id=str(household_id), data=data_dict, battery=SimpleBattery(capacity_in_kwh=100, charge_efficiency=0.99, discharge_efficiency=0.99)))
            iteration_counter += 1

        self.blockchain: Blockchain = Blockchain()
        self.forecaster: Forecaster = PerfectForecaster()
        self.optimizer: OptimizerStrategy = SimpleRuleBasedOptimizer()
        self.city_grid_price_forcaster: CityGridPriceForecaster = SimpleCityGridPriceForecaster()
        self.token_to_battery: TokenToBattery = TokenToBattery()

    def run(self, steps):
        for step in range(steps):
            households_forecasts: dict[str, dict[Literal["production", "consumption"], list[float]]] = {}
            for hh in self.households:

                current_info: dict[Literal["production", "consumption", "stored_kwh"], float] = hh.get_current_data(iteration=step)
                print(f"Household({hh.id}): production:{current_info["production"]}, consumption:{current_info["consumption"]}, battery:{current_info["stored_kwh"]}KWh")
                households_forecasts[hh.id] = self.forecaster.forecast(household=hh, iteration=step, forecast_size=Config.FORECASTER_PRED_SIZE) 
                self.blockchain.add_household_data(id=hh.id, production=current_info["production"], consumption=current_info["consumption"], stored_kwh=current_info["stored_kwh"])
            self.blockchain.add_block(f"['type': 'forecast_data', 'data': {json.dumps(households_forecasts)}]")

            city_grid_price_forcast = self.city_grid_price_forcaster.forecast(price_history=[], forecast_size=Config.CITY_GRID_PRICE_PRED_SIZE)

            trade_plan = self.optimizer.optimize(self.households, households_forecasts, city_grid_price_forcast)
            finalized_offers = self.finalize_offers(trade_plan)

            self.blockchain.add_offer(finalized_offers=finalized_offers)
            self.blockchain.trade_event()

    def finalize_offers(self, optimized_offers: list[dict]) -> dict[str, list[dict]]:
        finalized_offers: dict[str, list[dict]] = defaultdict(list)
        for offers in optimized_offers:
            for offer in offers["buys"]:
                finalized_offers[offer["from"]].append({
                    "seller": offer["from"],
                    "buyer": offers["id"],
                    "amount": offer["amount"]
                })
        return finalized_offers

