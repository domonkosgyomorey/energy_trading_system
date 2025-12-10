import hashlib
from collections import deque
import json
import time

from dataclasses import dataclass, asdict

# Default history size for household data tracking
DEFAULT_HISTORY_SIZE = 10

@dataclass()
class HouseholdTrades:
    trades_from: list[dict]
    amount_from_city: float
    amount_to_city: float
    central_battery_tax: float

class HouseholdData:
    def __init__(self,  id: str, history_size: int = DEFAULT_HISTORY_SIZE):
        self.id:str = id
        self.production: deque[float] = deque(maxlen=history_size) 
        self.consumption: deque[float] = deque(maxlen=history_size)
        self.wallet: float = 0 
        self.trades: HouseholdTrades 
        self.token: float = 0

    def __str__(self) -> str:
        return f"{{id:{self.id},production:{self.production[-1]},consumption:{self.consumption[-1]},wallet:{self.wallet},token:{self.token}}}"


    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "production": self.production,
            "consumption": self.consumption,
            "wallet": self.wallet,
            "trades": self.trades,
            "token": self.token
        }

class Block:
    def __init__(self, index: int, timestamp: float, data: str, previous_hash: str):
        self.index = index
        self.timestamp = timestamp
        self.data: str = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        block_string = f"{self.index}{self.timestamp}{self.data}{self.previous_hash}"
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain: list[Block] = [self.create_genesis_block()]
        self.households: dict[str, HouseholdData] = {}
        
    def create_genesis_block(self) -> Block:
        return Block(0, time.time(), "Genesis Block", "0")

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def add_block(self, data: str) -> None:
        previous_block = self.get_latest_block()
        new_block = Block(len(self.chain), time.time(), data, previous_block.hash)
        self.chain.append(new_block)

    def add_household_data(self, id: str, production: float, consumption: float, wallet: float, stored_kwh: float) -> None:
        if self.households.get(id) is None:
            self.households[id] = HouseholdData(id=id)
        
        self.households[id].production.append(production)
        self.households[id].consumption.append(consumption)
        self.households[id].wallet = wallet
        self.households[id].token = stored_kwh
        self.add_block(self.households[id].__str__())

    def get_households_data(self) -> list[dict]:
        return [household.to_dict() for _, household in self.households.items()]


    def add_trades(self, finalized_offers: dict[str, dict]) -> None:
        for household_id, _ in finalized_offers.items():
            self.households[household_id].trades = HouseholdTrades(**finalized_offers[household_id])

    def trade_event(self, local_energy_price: float, city_buy_price: float, city_sell_price: float) -> None:
        trading_strs: list[str] = []

        for address, household in self.households.items():
            if household.trades is None:
                continue

            # --- Execute P2P trading ---
            p2p_total_sold = 0
            for trade in household.trades.trades_from:
                seller = self.households[trade["seller"]]
                amount = trade["amount"]

                seller.token -= amount
                seller.wallet += amount * local_energy_price

                household.token += amount
                household.wallet -= amount * local_energy_price

                p2p_total_sold += amount

            # --- Net production (gross - p2p sold) ---
            net_production = household.production[-1] - p2p_total_sold
            household.token += net_production

            # --- Purchase from city ---
            household.token += household.trades.amount_from_city
            household.wallet -= household.trades.amount_from_city * city_buy_price

            # --- Sell to city ---
            household.token -= household.trades.amount_to_city
            household.wallet += household.trades.amount_to_city * city_sell_price

            # --- Central battery tax ---
            household.token -= household.trades.central_battery_tax

            # --- Consumption ---
            household.token -= household.consumption[-1]

            # --- Transaction log ---
            trading_strs.append(address + ":" + json.dumps(asdict(household.trades), ensure_ascii=False))

        self.add_block(";".join(trading_strs))

    
