import hashlib
from collections import deque
import json
import time

from simulation.config import Config
from dataclasses import dataclass, asdict

@dataclass()
class HouseholdTrades:
    trades_from: list[dict]
    amount_from_city: float
    central_battery_tax: float

class HouseholdData:
    def __init__(self,  id: str):
        self.id:str = id
        self.production: deque[float] = deque(maxlen=Config.FORECASTER_HIST_SIZE) 
        self.consumption: deque[float] = deque(maxlen=Config.FORECASTER_HIST_SIZE)
        self.trades: HouseholdTrades
        self.token: float = 0

    def __str__(self) -> str:
        return f"{{id:{self.id},production:{self.production[-1]},consumption:{self.consumption[-1]},token:{self.token}}}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "production": self.production,
            "consumption": self.consumption,
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

    def add_household_data(self, id: str, production: float, consumption: float, stored_kwh: float) -> None:
        if self.households.get(id) is None:
            self.households[id] = HouseholdData(id=id)
        
        self.households[id].production.append(production)
        self.households[id].consumption.append(consumption)
        self.households[id].token = stored_kwh
        self.add_block(self.households[id].__str__())

    def get_households_data(self) -> list[dict]:
        return [household.to_dict() for _, household in self.households.items()]


    def add_trades(self, finalized_offers: dict[str, dict]) -> None:
        for household_id, _ in finalized_offers.items():
            self.households[household_id].trades = HouseholdTrades(**finalized_offers[household_id])

    def trade_event(self) -> None:
        trading_strs: list[str] = []
        for address, household in self.households.items():
            if household.trades is None:
                continue
            
            for trade in household.trades.trades_from:
                self.households[trade["seller"]].token -= trade["amount"]
                household.token += trade["amount"]
            
            household.token += household.trades.amount_from_city
            household.token -= household.trades.central_battery_tax
            household.token -= household.consumption[-1]

            trading_strs.append(address+":"+json.dumps(asdict(household.trades), ensure_ascii=False))

        self.add_block(";".join(trading_strs))

    
