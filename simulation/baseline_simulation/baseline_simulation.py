import pandas as pd
from typing import Literal
from collections import defaultdict
from simulation.blockchain import Blockchain
from simulation.battery.battery import Battery
from simulation.battery.simple_battery import SimpleBattery
from simulation.city_grid_price_forecaster.city_gird_price_forecaster import CityGridPriceForecaster
from simulation.city_grid_price_forecaster.simple_city_grid_price_forecaster import SimpleCityGridPriceForecaster
from simulation.config import Config
from simulation.household import Household

class BaselineSimulator:
    def __init__(self, household_data: pd.DataFrame):
                
        # Data storage for the households
        self.households: list[Household] = []
        
        # Currently limiting the number of household in our simulation
        limit = 5      
        
        # iteration counter for the household limit and battery management
        iteration_counter: int = 0
        
        # group each record according to their household id
        grouped = household_data.groupby("id")

        for household_id, group in grouped:
            
            # Drop columns whith non-numeric values
            group = group.drop(columns=["id", "timestamp", "category", "season", "period"])
            
            # Data storage for the new database for the current household
            raw_dict = {col: group[col].to_list() for col in group.columns}
            
            # Data storege for the accumulated database
            data_dict = defaultdict(list)

            # Accumulates 96*7*(15 minutes) so accumulated a whole week for testing
            for key, val in raw_dict.items():
                for i in range(0, len(val)-96, 96):
                    sum = 0
                    for j in range(96):
                        sum += val[i+j]
                    data_dict[key].append(sum)   

            self.households.append(Household(id=str(household_id), data=data_dict, battery=SimpleBattery(capacity_in_kwh=0, charge_efficiency=0, discharge_efficiency=0)))

            # If the household limit is reached we stoping the household initializatio
            if iteration_counter > limit:
                break

            # Increase the iteration number
            iteration_counter += 1

        self.iteration = 0
        self.city_grid_price_forcaster: CityGridPriceForecaster = SimpleCityGridPriceForecaster()

        # Initializing the simulated blockchain
        self.blockchain: Blockchain = Blockchain()

    def step(self):
        # Predicts the future city's electricity grid prices
        city_grid_price_forcast = self.city_grid_price_forcaster.forecast(price_history=[], forecast_size=1)
        buy_price = city_grid_price_forcast["buy"][0]

        for household in self.households:
            data = household.get_sensor_data(self.iteration)
            production = data["production"]
            consumption = data["consumption"]

            net_energy = production - consumption

            if net_energy >= 0:
                energy_bought = 0
            else:
                energy_bought = abs(net_energy)
                household.wallet -= energy_bought * buy_price

            self.blockchain.add_household_data(
                id=household.id,
                production=production,
                consumption=consumption,
                wallet=household.wallet,
                stored_kwh=0.0
            )

    def run(self, iterations: int):
        for i in range(iterations):
            self.iteration = i
            self.step()
    