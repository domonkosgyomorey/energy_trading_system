from simulation.household import Household
from simulation.blockchain import Blockchain
from simulation.forecaster.perfect_forecaster import PerfectForecaster
from simulation.forecaster.forecaster import Forecaster
from simulation.optimizer.simple_optimizer import SimpleRuleBasedOptimizer
from simulation.battery.simple_battery import SimpleBattery
from simulation.battery.central_battery import CentralBattery
from simulation.battery.shared_battery import SharedBattery 
from simulation.household_resource_handler.simple_resource_handler import SimpleResourceHandler
from simulation.config import Config

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

        self.central_battery: CentralBattery = CentralBattery(capacity_in_kwh=10000000.0, efficiency=0.999, tax_per_kwh=0.01)
        self.households: list[Household] = []
         
        grouped = household_data.groupby("id")
        iteration_counter: int = 0
        no_battery_iter: int = int(grouped.count() * Config.HOUSEHOLD_WITHOUT_BATTERY_PROB)

        for household_id, group in grouped:
            group = group.drop(columns="id")
            data_dict = {col: group[col].to_list() for col in group.columns}
            
            if iteration_counter % no_battery_iter == 0:
                self.households.append(Household(id=str(household_id), data=data_dict, battery=SharedBattery(central_battery=self.central_battery,household_id=str(household_id)), resource_handler=SimpleResourceHandler()))
            else:
                self.households.append(Household(id=str(household_id), data=data_dict, battery=SimpleBattery(capacity_in_kwh=100, efficiency=0.98), resource_handler=SimpleResourceHandler()))
            iteration_counter += 1

        self.blockchain = Blockchain()
        self.forecaster: Forecaster = PerfectForecaster()
        self.optimizer = SimpleRuleBasedOptimizer(self.central_battery)

    def run(self, steps):
        for step in range(steps):
            forecasts: dict[str, dict[Literal["production", "consumption"], list[float]]] = {}
            for hh in self.households:
                hh.update(iteration=step, updated_token=hh.battery.get_stored_kwh())
                current_info: dict[Literal["production", "consumption", "stored_kwh"], float] = hh.get_current_data(iteration=step)
                print(f"Household({hh.id}): production:{current_info["production"]}, consumption:{current_info["consumption"]}, battery:{current_info["stored_kwh"]}KWh")
                forecasts[hh.id] = self.forecaster.forecast(household=hh, iteration=step) 
                self.blockchain.add_household_data(id=hh.id, production=current_info["production"], consumption=current_info["consumption"], stored_kwh=current_info["stored_kwh"])
            self.blockchain.add_block(f"['type': 'forecast_data', 'data': {json.dumps(forecasts)}]")

            trade_plan = self.optimizer.optimize(self.households, forecasts)
            finalized_offers = self.finalize_offers(trade_plan)

            self.blockchain.add_offer(offers=finalized_offers)
            self.blockchain.trade_event()

    def finalize_offers(self, optimized_offers: list[dict]) -> dict[str, list[dict]]:
       return optimized_offers 
