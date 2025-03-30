import os

from community import Community
from config import Config
from optimizer import OptimizationModel

from blockchain import Blockchain

if __name__ == "__main__":

    db_path = os.path.join(Config.DB_PATH, Config.DB)
    community = Community(db_path)

    blockchain = Blockchain()
    models = [OptimizationModel(blockchain, h) for h in community.households]

    days = 7
    total_days = 30

    for current_day in range(total_days):
        for model in models:
            forecast_demand, forecast_generation = model.household.make_forecasts(days)
            results = model.optimize(
                current_day, days, forecast_demand, forecast_generation
            )
            print(
                f"\nDay {current_day} - Household {model.household.household_id} Optimization results:"
            )
            print("Cost:", results["cost"])
            print("Buy:", results["buy"])
            print("Sell:", results["sell"])

        blockchain.finalize_day(current_day)

        reservations = blockchain.get_reservations(current_day)
        print("Reservations:", reservations)
        print("Reservations:", reservations)
        print("Reservations:", reservations)
        print("Reservations:", reservations)
