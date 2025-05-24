from typing import Literal
from simulation.battery.central_battery import CentralBattery
from simulation.household import Household

class SimpleRuleBasedOptimizer:
    def __init__(self, central_battery: CentralBattery):
        self.central_battery = central_battery

    def optimize(self, households: list[Household], forecasts: dict[str, dict[Literal["production", "consumption"], list[float]]]) -> list[dict]:
        offers = []

        # Csoportosítjuk a háztartásokat id alapján
        hh_map = {hh.id: hh for hh in households}
        days = len(next(iter(forecasts.values())))  # feltételezzük minden háztartásra ugyanannyi nap van

        for day in range(days):
            daily_balances = {}
            surplus_hhs = []
            deficit_hhs = []

            # 1. lépés: kiszámítjuk az egyenleget minden háztartásra
            for hh_id, hh in hh_map.items():
                production: float = forecasts[hh_id]["production"][day]
                consumption: float = forecasts[hh_id]["consumption"][day]

                net = production - consumption
                daily_balances[hh_id] = net

                if net > 0:
                    surplus_hhs.append((hh_id, net))
                elif net < 0:
                    deficit_hhs.append((hh_id, -net))

            # 2. lépés: fedezzük a hiányokat a feleslegekből
            for def_id, need in deficit_hhs:
                for i, (sur_id, available) in enumerate(surplus_hhs):
                    if available <= 0:
                        continue
                    used = min(available, need)
                    offers.append({
                        "type": "transfer",
                        "from": sur_id,
                        "to": def_id,
                        "day": day,
                        "amount": round(used, 2)
                    })
                    surplus_hhs[i] = (sur_id, available - used)
                    need -= used
                    if need <= 0:
                        break
                if need > 0:
                    hh = hh_map[def_id]
                    # 3. lépés: próbáljuk akkumulátorból fedezni
                    if hh.battery and hh.battery.level >= need:
                        offers.append({
                            "type": "battery_discharge",
                            "household_id": def_id,
                            "day": day,
                            "amount": round(need, 2)
                        })
                        hh.battery.level -= need
                        need = 0
                    elif self.central_battery and self.central_battery.level >= need:
                        offers.append({
                            "type": "central_discharge",
                            "household_id": def_id,
                            "day": day,
                            "amount": round(need, 2)
                        })
                        self.central_battery.level -= need
                        need = 0

                    if need > 0:
                        # 4. lépés: városból kell kérni
                        offers.append({
                            "type": "request",
                            "household_id": def_id,
                            "day": day,
                            "amount": round(need, 2)
                        })

            # 5. lépés: a megmaradt felesleget eltároljuk (ha lehet)
            for hh_id, remain in surplus_hhs:
                if remain <= 0:
                    continue
                hh = hh_map[hh_id]
                if hh.battery and hh.battery.capacity - hh.battery.level > 0:
                    store_amount = min(remain, hh.battery.capacity - hh.battery.level)
                    offers.append({
                        "type": "battery_charge",
                        "household_id": hh_id,
                        "day": day,
                        "amount": round(store_amount, 2)
                    })
                    hh.battery.level += store_amount
                    remain -= store_amount
                if remain > 0 and self.central_battery:
                    central_free = self.central_battery.capacity - self.central_battery.level
                    store_amount = min(remain, central_free)
                    if store_amount > 0:
                        offers.append({
                            "type": "central_charge",
                            "household_id": hh_id,
                            "day": day,
                            "amount": round(store_amount, 2)
                        })
                        self.central_battery.level += store_amount
                        remain -= store_amount
                if remain > 0:
                    # még mindig maradt felesleg: küldjük ki
                    offers.append({
                        "type": "offer",
                        "household_id": hh_id,
                        "day": day,
                        "amount": round(remain, 2)
                    })

        return offers
