from simulation.household import Household
from simulation.blockchain import Blockchain
from simulation.forecaster import Forecaster
from simulation.optimizer import SimpleRuleBasedOptimizer
from simulation.battery import SimpleBattery

class Simulation:
    def __init__(self, num_households):
        self.households = [
            Household(id=i, battery=SimpleBattery(capacity=20.0 + i*5, efficiency=0.9)) if i % 2 == 0 else Household(id=i)
            for i in range(num_households)
        ]
        self.blockchain = Blockchain()
        self.forecaster = Forecaster()
        self.optimizer = SimpleRuleBasedOptimizer()

    def run(self, steps):
        for step in range(steps):
            print(f"\n--- Step {step+1} ---")
            data = [hh.simulate() for hh in self.households]
            self.blockchain.add_block({"type": "daily_data", "data": data})
            for d in data:
                print(f"Household {d['id']}: Prod={d['production']}, Cons={d['consumption']}, Bat={d['battery']}, Tokens={d['tokens']}")

            # Forecast
            forecasts = {hh.id: self.forecaster.forecast(hh) for hh in self.households}
            self.blockchain.add_block({"type": "forecast_data", "data": forecasts})

            # Optimization
            trade_plan = self.optimizer.optimize(self.households, forecasts)

            # Offers on blockchain
            offers = [trade for trade in trade_plan if trade['type'] == 'offer']
            requests = [trade for trade in trade_plan if trade['type'] == 'request']
            self.blockchain.add_block({"type": "offers", "data": offers})
            self.blockchain.add_block({"type": "requests", "data": requests})

            print("Offers:")
            for offer in offers:
                print(offer)
            print("Requests:")
            for req in requests:
                print(req)