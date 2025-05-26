from collections import defaultdict
from simulation.household import Household
from simulation.blockchain import Blockchain
from simulation.forecaster.perfect_forecaster import PerfectForecaster
from simulation.forecaster.forecaster import Forecaster
from simulation.optimizer.convex_optimizer import ConvexOptimizer
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
from simulation.utils.logger import logger, log_df

class Simulation: 
    def __init__(self, household_data: pd.DataFrame):
        """
        Description: Simulates the automatic energy trading with blockchain.

        Args:
            household_data (pd.DataFrame): Database for the household production, consumption data
            household_without_battery_prob (float): From this, the simulation calculates how many household will have battery
            forecaster_pred_range (int): The forecaster will predict prod and cons this much further
        """

        # Makes a global central battery park for the energy community
        self.central_battery: CentralBattery = CentralBattery(capacity_in_kwh=10000000.0, charge_efficiency=1.0, discharge_efficiency=1.0, tax_per_kwh=0.01)
        
        # Data storage for the households
        self.households: list[Household] = []
        
        # group each record according to their household id
        grouped = household_data.groupby("id")

        # iteration counter for the household limit and battery management
        iteration_counter: int = 0

        # Tells how ofter do we need to use the central battery
        no_battery_iter: int = int(grouped.ngroups * Config.HOUSEHOLD_WITHOUT_BATTERY_PROB)

        # Currently limiting the number of household in our simulation
        limit = 5

        # initilizing the households
        for household_id, group in grouped:
            
            # Drop columns whith non-numeric values
            group = group.drop(columns=["id", "timestamp", "category", "season", "period"])
            
            # Data storage fot the new database for the current household
            raw_dict = {col: group[col].to_list() for col in group.columns}
            
            # Data storege for the accumulated database
            data_dict = defaultdict(list)

            # Accumulates 96*7*(15 minutes) so accumulated a whole week for testing
            for key, val in raw_dict.items():
                for i in range(0, len(val)-96*7, 96*7):
                    sum = 0
                    for j in range(96*7):
                        sum += abs(val[i+j])
                    data_dict[key].append(sum)                

            # Every so often we will use the central battery
            if iteration_counter % no_battery_iter == 0:
                self.households.append(Household(id=str(household_id), data=data_dict, battery=SharedBattery(central_battery=self.central_battery,household_id=str(household_id))))
            else:
                self.households.append(Household(id=str(household_id), data=data_dict, battery=SimpleBattery(capacity_in_kwh=100, charge_efficiency=1.0, discharge_efficiency=1.0)))
            
            # If the household limit is reached we stoping the household initializatio
            if iteration_counter > limit:
                break

            # Increase the iteration number
            iteration_counter += 1

        # Initializing the simulated blockchain
        self.blockchain: Blockchain = Blockchain()
        
        # Initializing the perfect production/consumption forecaster
        self.forecaster: Forecaster = PerfectForecaster()

        # Initializing the optimizer based on Convex Optimization Problem
        self.optimizer: OptimizerStrategy = ConvexOptimizer()

        # Initializing the city's electricity grid price forecaster
        self.city_grid_price_forcaster: CityGridPriceForecaster = SimpleCityGridPriceForecaster()
        
        # Initializing the battery updater with the tokens
        self.token_to_battery: TokenToBattery = TokenToBattery()

    def run(self, steps: int):
        """
        Description: Runs the simulation for the requested steps amount
        
        Args:
            step (int): Tells the simulation that, how many step should make 
        """
        
        # Simulation core
        for step in range(steps):
            
            # A data structure thats helps the dataframe logger to collect data
            df_logger_helper: list[dict] = []

            # Helps to collect data for the logger
            logger_helper: dict[str, dict] = {}
           
            # Logging the start of a new iteration
            logger.info(f"------Iteration: {step+1}-------")
            print(f"----{step+1}. Iteration-----")
            
            # Data storeage for the households' production, consumption forecasts
            households_forecasts: dict[str, dict[Literal["production", "consumption"], list[float]]] = {}
            
            # Data storage for the current energy consumption for each household
            currect_household_consumption: dict[str, float] = {} 

            # Update each household and collect information about them for this iteration
            for hh in self.households:

                # Update the battery with the currect production and gives information about
                # the current prodcution, consumpotion and the battery state
                current_info: dict[Literal["production", "consumption", "stored_kwh"], float] = hh.update_battery_with_production_and_get_sensor_data(iteration=step)
                
                # Save the current consumption data for the optimizer
                currect_household_consumption[hh.id] = current_info["consumption"]

                # Logging the household state
                df_logger_helper.append({
                    "iteration": step,
                    "id": hh.id,
                    "production": round(current_info["production"], ndigits=4),
                    "consumption": round(current_info["consumption"], ndigits=4),
                    "stored_kwh": round(current_info["stored_kwh"], ndigits=4)
                })

                # Logging the household state
                logger_helper[hh.id] = {
                    "iteration": step,
                    "production": round(current_info["production"], ndigits=4),
                    "consumption": round(current_info["consumption"], ndigits=4),
                    "stored_kwh": round(current_info["stored_kwh"], ndigits=4)
                }

                # Store the current household's state in the blockchain
                self.blockchain.add_household_data(id=hh.id, production=current_info["production"], consumption=current_info["consumption"], stored_kwh=current_info["stored_kwh"])
                
                # Make production/consumption forecast for each household
                households_forecasts[hh.id] = self.forecaster.forecast(household=hh, iteration=step, forecast_size=Config.FORECASTER_PRED_SIZE) 
                          
            # Store each forecast in the blockchain
            self.blockchain.add_block(f"['type': 'forecast_data', 'data': {json.dumps(households_forecasts)}]")

            # Predicts the future city's electricity gird prices
            city_grid_price_forcast = self.city_grid_price_forcaster.forecast(price_history=[], forecast_size=Config.CITY_GRID_PRICE_PRED_SIZE)

            # Make trades and actions what the household should take to create an optimal energy community
            trade_plan = self.optimizer.optimize(self.households, households_forecasts, city_grid_price_forcast, currect_household_consumption)

            # Re-structures the trade plan into a more useful format
            finalized_offers = self.finalize_offers(trade_plan)
            
            # Add trade plan logs into the current household state
            for logged_hhd in df_logger_helper:
                trades_from:list[str] = []
                for offer in finalized_offers[logged_hhd["id"]]["trades_from"]:
                    trades_from.append(f"{offer['seller']}:{round(offer['amount'],ndigits=4)}")
                
                # Logs the trades
                logged_hhd["trades_from"] = "-".join(trades_from)
                logged_hhd["amount_from_city"] = round(finalized_offers[logged_hhd["id"]]["amount_from_city"],ndigits=4)
                logged_hhd["central_battery_tax"] = round(self.central_battery.get_tax_per_kwh(logged_hhd["id"]),ndigits=4)
                
                logger_helper[logged_hhd["id"]]["trades_from"] = "-".join(trades_from)
                logger_helper[logged_hhd["id"]]["amount_from_city"] = round(finalized_offers[logged_hhd["id"]]["amount_from_city"],ndigits=4)
                logger_helper[logged_hhd["id"]]["central_battery_tax"] = round(self.central_battery.get_tax_per_kwh(logged_hhd["id"]),ndigits=4)
                

            # Updates the logs
            log_df(pd.DataFrame(df_logger_helper))
            logger.info(json.dumps(logger_helper, indent=4))
                       
            # Gets the taxer for each household, which stores energy in the central battery park
            taxes_for_household: dict[str, float] = self.central_battery.get_households_tax_in_kwh()
            
            # Add taxes for the offers
            for id, _ in finalized_offers.items():
                finalized_offers[id]["central_battery_tax"] = taxes_for_household.get(id, 0.0)
            
            # Stores the trade offers in blockchain
            self.blockchain.add_trades(finalized_offers=finalized_offers)
            
            # Executes all of the trades
            self.blockchain.trade_event()

            # After the trades updates batteries for each household
            for hh in self.households:
                self.token_to_battery.handle(hh.battery, self.blockchain.households[hh.id].token)

    def finalize_offers(self, optimized_offers: list[dict]) -> dict[str, dict]:
        """
        Description: Transforms the trade plan into a more useful formate

        Args:
            optimized_offers (list[dict]): the trade plan
        """

        finalized_offers: dict[str, dict] = {}
        for offers in optimized_offers:
            hh_stats: dict = {
                "trades_from": [],
                "amount_from_city": offers["buy_from_city"],
            }
            for offer in offers["buys"]:
                hh_stats["trades_from"].append({
                    "seller": offer["from"],
                    "amount": offer["amount"],
                })
            finalized_offers[offers["id"]] = hh_stats
        return finalized_offers

