import pandas as pd
from collections import defaultdict

# Szimuláció komponensek
from simulation.battery.central_battery import CentralBattery
from simulation.battery.shared_battery import SharedBattery
from simulation.battery.simple_battery import SimpleBattery
from simulation.city_grid_price_forecaster.simple_city_grid_price_forecaster import SimpleCityGridPriceForecaster
from simulation.household import Household
from simulation.config import Config
from simulation.utils.logger import logger, log_df

class BaselineSimulator:
    def __init__(self, household_data: pd.DataFrame):
        """
        Egyszerűsített baseline szimulátor, amely nem használ előrejelzést, optimalizációt vagy központi akkumulátort.

        Paraméter:
            household_data (pd.DataFrame): A háztartások termelés/fogyasztás adatai (id, timestamp stb.)
        """

        # Egyszerű városi áramár előrejelző (rögzített vagy szimpla logikával dolgozik)
        self.city_grid_price_forecaster = SimpleCityGridPriceForecaster()

        # Háztartások listája
        self.households: list[Household] = []

        # Makes a global central battery park for the energy community
        self.central_battery: CentralBattery = CentralBattery(capacity_in_kwh=10000000.0, charge_efficiency=1.0, discharge_efficiency=1.0, tax_per_kwh=0.01)
        
        # Háztartások csoportosítása azonosító alapján
        grouped = household_data.groupby("id")

       # Tells how ofter do we need to use the central battery
        no_battery_iter: int = int(grouped.ngroups * Config.HOUSEHOLD_WITHOUT_BATTERY_PROB)

        # Maximum háztartás a szimulációban (teszteléshez limitált)
        limit = 5
        iteration_counter = 0

        # Háztartások inicializálása
        for household_id, group in grouped:
            # Elhagyjuk a nem numerikus oszlopokat
            group = group.drop(columns=["id", "timestamp", "category", "season", "period"])

            # Nyers értékek kigyűjtése
            raw_dict = {col: group[col].to_list() for col in group.columns}

            # Heti összevont adatok készítése (96 x 15 perc = 1 nap, így 96 x 7 = 1 hét)
            data_dict = defaultdict(list)
            for key, val in raw_dict.items():
                for i in range(0, len(val) - 96, 96):
                    week_sum = sum(val[i:i+96])
                    data_dict[key].append(week_sum)

            # Every so often we will use the central battery
            if iteration_counter % no_battery_iter == 0:
                self.households.append(Household(id=str(household_id), data=data_dict, battery=SharedBattery(central_battery=self.central_battery,household_id=str(household_id))))
            else:
                self.households.append(Household(id=str(household_id), data=data_dict, battery=SimpleBattery(capacity_in_kwh=100, charge_efficiency=1.0, discharge_efficiency=1.0)))
            
            # Csak az első 'limit' számú háztartást vesszük figyelembe
            if iteration_counter >= limit:
                break
            iteration_counter += 1

    def run(self, steps: int):
        """
        A baseline szimuláció futtatása adott számú lépésre.

        Paraméter:
            steps (int): Iterációk száma
        """
        for step in range(steps):
            logger.info(f"------Baseline Iteration: {step + 1}-------")
            print(f"----Baseline Iteration {step + 1}-----")

            # Adatok naplózásához
            df_logger_helper = []

            # Egyszerű előrejelzés (itt csak egy időpillanatra)
            price_forecast = self.city_grid_price_forecaster.forecast(price_history=[], forecast_size=1)
            buy_price = price_forecast["buy"][0]  # Ár, amennyiért a várostól vásárolhatunk

            for hh in self.households:
                # Szenzoradatok lekérdezése (aktuális termelés, fogyasztás, akku állapot)
                data = hh.get_sensor_data(step)
                production = data["production"]
                consumption = data["consumption"]
                stored_kwh = data["stored_kwh"]

                # Nettó energia: ha negatív, akkor vásárolnunk kell
                net_energy = production + hh.battery.retrieve_energy(hh.battery.get_stored_kwh()) - consumption
                if net_energy < 0:
                    bought_energy = abs(net_energy)
                    hh.wallet -= bought_energy * buy_price  # Vásárolt energia levonása a pénztárcából
                else:
                    bought_energy = 0  # Nem kell vásárolni, van elég energia
                    hh.battery.store_energy(net_energy)

                # Eredmények hozzáadása a naplózáshoz
                df_logger_helper.append({
                    "iteration": step,
                    "id": hh.id,
                    "production": round(production, 4),
                    "consumption": round(consumption, 4),
                    "stored_kwh": round(hh.battery.get_stored_kwh(), 4),
                    "wallet": round(hh.wallet, 4),
                    "grid_energy": round(bought_energy, 4)  # Mennyi energiát vett a várostól
                })

            # Eredmények naplózása (CSV, stdout stb.)
            log_df(pd.DataFrame(df_logger_helper))
