import hashlib
from collections import deque
import time
from typing import Literal
class Offer:
    def __init__(self, seller_id: str, buyer_id: str, amount: float):
        self.seller_address = seller_id
        self.buyer_address = buyer_id
        self.amount = amount

    def to_string(self) -> str:
        return f"{self.seller_address};{self.buyer_address};{self.amount};"

    @staticmethod
    def from_dict(data: dict[Literal["seller_id", "buyer_id", "amount", "iter_index"], str | float | int]) -> None:
       Offer(seller_id=str(data["seller_id"]), buyer_id=str(data["buyer_id"]), amount=float(data["amount"])) 
    
class HouseholdData:
    def __init__(self,  id: str, forecast_history_size):
        self.id:str = id
        self.production: deque[float] = deque(maxlen=forecast_history_size) 
        self.consumption: deque[float] = deque(maxlen=forecast_history_size)
        self.token: float = 0

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
    def __init__(self, forecast_history_size: int, forecast_prediction_size: int):
        self.chain: list[Block] = [self.create_genesis_block()]
        self.households: dict[str, HouseholdData] = {}
        self.offers: list[list[Offer]] = []
        
        self.forecast_history_size = forecast_history_size
        self.forecast_prediction_size = forecast_prediction_size

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
            self.households[id] = HouseholdData(id=id,forecast_history_size=self.forecast_prediction_size)
        
        self.households[id].production.append(production)
        self.households[id].consumption.append(consumption)
        self.households[id].token = stored_kwh

    def get_households_data(self) -> list[dict]:
        return [household.to_dict() for _, household in self.households.items()]


    def update_households(self) -> None:
        # Itt majd implementáld az update logikát
        pass

    def add_offer(self, offers: list[dict]) -> None:
        self.offers = [[] for _ in range(self.forecast_prediction_size)]
        for offer in offers:
            self.offers[offer["iter_index"]].append(Offer(buyer_id=offer["buyer_id"], seller_id=offer["seller_id"], amount=offer["amount"]))

    def trade_on_every_15(self) -> None:
        # A blokk adat egy string, az első pending_offers lista összes ajánlatának string összefűzve, soronként:
        data = "\n".join([offer.to_string() for offer in self.pending_offers[0]])
        self.add_block(data)

        # Shifteljük a pending_offers listát eggyel előrébb
        for i in range(len(self.pending_offers) - 1):
            self.pending_offers[i] = self.pending_offers[i + 1]
        self.pending_offers.pop()


    
