from simulation.household import Household
from simulation.blockchain import Blockchain
from simulation.forecaster import Forecaster
from simulation.optimizer.simple_optimizer import SimpleRuleBasedOptimizer
from simulation.battery.simple_battery import SimpleBattery
from simulation.battery.central_battery import CentralBattery

class Simulation:
    def __init__(self, num_households):
        self.central_battery = CentralBattery(capacity=100.0, efficiency=0.9, tax_per_kwh=1.0)
        self.households = [
            Household(id=i, battery=SimpleBattery(capacity=20.0 + i*5, efficiency=0.9)) if i % 2 == 0 else Household(id=i)
            for i in range(num_households)
        ]
        self.blockchain = Blockchain()
        self.forecaster = Forecaster()
        self.optimizer = SimpleRuleBasedOptimizer(self.central_battery)

    def run(self, steps):
        for step in range(steps):
            print(f"\n--- Step {step+1} ---")
            data = []
            for hh in self.households:
                net = hh.simulate()
                result = hh.handle_energy(net, self.central_battery)
                data.append(result)

            self.blockchain.add_block({"type": "daily_data", "data": data})
            for d in data:
                print(f"Household {d['id']}: Prod={d['production']}, Cons={d['consumption']}, Bat={d['battery']}, Tokens={d['tokens']}")

            forecasts = {hh.id: self.forecaster.forecast(hh) for hh in self.households}
            self.blockchain.add_block({"type": "forecast_data", "data": forecasts})

            trade_plan = self.optimizer.optimize(self.households, forecasts)
            offers = [trade for trade in trade_plan if trade['type'] == 'offer']
            requests = [trade for trade in trade_plan if trade['type'] == 'request']

            matched = []
            for req in requests:
                for offer in offers:
                    if offer['day'] == req['day'] and offer['amount'] > 0:
                        offered_hh = self.households[offer['household_id']]
                        requested_hh = self.households[req['household_id']]
                        transfer = min(req['amount'], offer['amount'])
                        actual_transfer = offered_hh.provide_energy(transfer)
                        requested_hh.receive_energy(actual_transfer)
                        offer['amount'] -= transfer
                        req['amount'] -= transfer
                        matched.append({
                            "from": offer['household_id'],
                            "to": req['household_id'],
                            "day": offer['day'],
                            "amount": round(actual_transfer, 2)
                        })
                        if req['amount'] <= 0:
                            break

            self.blockchain.add_block({"type": "offers", "data": offers})
            self.blockchain.add_block({"type": "requests", "data": requests})
            self.blockchain.add_block({"type": "trades", "data": matched})

            print("Trades:")
            for trade in matched:
                print(trade)