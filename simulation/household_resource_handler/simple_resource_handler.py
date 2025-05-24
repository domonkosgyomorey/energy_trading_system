from simulation.household_resource_handler.resource_handler import ResourceHandler
from simulation.battery.battery import Battery

class SimpleResourceHandler(ResourceHandler):

    def handle(self, production: float, consumption: float, battery: Battery, token: float) -> None:
        current_kwh:float = battery.get_stored_kwh()
        if current_kwh > token:
            net = current_kwh - token
            battery.retrieve_energy(net)
        else:
            net = token - current_kwh
            battery.store_energy(net)
            
        current_net = production - consumption
        if current_net > 0:
            battery.store_energy(current_net)
        elif battery.get_stored_kwh()+current_net >= 0:
            _ = battery.retrieve_energy(current_net)
        else:
            _ = battery.retrieve_energy(battery.get_stored_kwh())
            print("No electisity for today! :(")
