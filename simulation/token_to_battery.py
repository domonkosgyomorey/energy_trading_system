from simulation.battery.battery import Battery

class TokenToBattery:

    def handle(self, battery: Battery, token: float) -> None:
        current_kwh:float = battery.get_stored_kwh()
        if current_kwh > token:
            net = current_kwh - token
            battery.retrieve_energy(net)
        else:
            net = token - current_kwh
            battery.store_energy(net)
