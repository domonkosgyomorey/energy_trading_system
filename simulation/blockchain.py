import hashlib
from queue import Queue
import time
from simulation.household import Household
from typing import Optional, List

class Offer:
    def __init__(self, seller_id: str, buyer_id: str, amount: float, timestamp: Optional[float] = None):
        self.seller_id = seller_id
        self.buyer_id = buyer_id
        self.amount = amount
        self.timestamp = timestamp or time.time()

    def to_string(self) -> str:
        offer=f"{self.seller_id};{self.buyer_id};{self.amount};{self.timestamp}"
        return offer
    
class HouseholdData:
    def __init__(self,  id: str, production: Queue, consumption:Queue, battery: float):
        self.id:str = id
        self.production = production
        self.consumption = consumption
        self.stored_kwh: float = battery

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "production": self.production,
            "consumption": self.consumption,
            "stored_kwh": self.stored_kwh
        }

class Block:
    def __init__(self, index: int, timestamp: float, data: str, previous_hash: str):
        self.index = index
        self.timestamp = timestamp
        self.data: str = data #deal info
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        block_string = f"{self.index}{self.timestamp}{self.data}{self.previous_hash}"
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self, forecast_history_size: int, forecast_prediction_size: int):
        self.chain: list[Block] = [self.create_genesis_block()]
        self.household_data_list: list[HouseholdData] = []
        self.pending_offers: list[list[Offer]] = []
        #self.done_deals: list[Offer] = []
        self.forecast_history_size = forecast_history_size
        self.forecast_history_prediction = forecast_prediction_size

    def create_genesis_block(self) -> Block:
        return Block(0, time.time(), "Genesis Block", "0")

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def add_block(self, data: str) -> None:
        previous_block = self.get_latest_block()
        new_block = Block(len(self.chain), time.time(), data, previous_block.hash)
        self.chain.append(new_block)

    def upsert_household_data(self, id: str, production: float, consumption: float, battery) -> None:

        self.household_data_list.append(HouseholdData(id, production, consumption, battery))

    def get_households_data(self) -> list[dict]:
        return [household.to_dict() for household in self.household_data_list]

    def update_households(self) -> None:
        # Itt majd implementáld az update logikát
        pass

    def add_offer(self, offers: list[Offer]) -> None:
        self.pending_offers.append(offers)

    def trade_on_every_15(self) -> None:
        # A blokk adat egy string, az első pending_offers lista összes ajánlatának string összefűzve, soronként:
        data = "\n".join([offer.to_string() for offer in self.pending_offers[0]])
        self.add_block(data)

        # Shifteljük a pending_offers listát eggyel előrébb
        for i in range(len(self.pending_offers) - 1):
            self.pending_offers[i] = self.pending_offers[i + 1]
        self.pending_offers.pop()


    