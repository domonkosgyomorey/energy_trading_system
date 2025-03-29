from argparse import ArgumentParser

from blockchain import Blockchain
from household import Household
from optimizer import OptimizationModel

if __name__ == "__main__":

    parser = ArgumentParser(
        description="Simulation of a blockchain-based energy trading system."
    )

    parser.add_argument(
        "--days", type=int, default=30, help="Number of days to simulate."
    )

    parser.add_argument(
        "--households",
        type=int,
        default=5,
        help="Number of households in the simulation.",
    )

    args = parser.parse_args()
    days = args.days
    households_count = args.households

    blockchain = Blockchain()
    households = [Household(household_id=i, battery_capacity=10) for i in range(5)]
    models = [OptimizationModel(blockchain, h) for h in households]

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
            print("Battery Level:", model.household.battery_level)

        blockchain.finalize_day(current_day)

        reservations = blockchain.get_reservations(current_day)
        print("Reservations:", reservations)
