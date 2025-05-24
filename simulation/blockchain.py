import hashlib
from collections import deque
import time

from simulation.config import Config

class Offer:
    def __init__(self, seller_id: str, buyer_id: str, amount: float):
        self.seller_address = seller_id
        self.buyer_address = buyer_id
        self.amount = amount

class HouseholdData:
    def __init__(self,  id: str):
        self.id:str = id
        self.production: deque[float] = deque(maxlen=Config.FORECASTER_HIST_SIZE) 
        self.consumption: deque[float] = deque(maxlen=Config.FORECASTER_HIST_SIZE)
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
        self.offers: list[Offer] = []
        
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


    def add_offer(self, offers: list[dict]) -> None:
        self.offers: list[Offer] = []
        for offer in offers:
            self.offers.append(Offer(buyer_id=offer["buyer_id"], seller_id=offer["seller_id"], amount=offer["amount"]))

    def trade_event(self) -> None:
        trading_str: str = ""
        for offer in self.offers:
            self.households[offer.seller_address].token -= offer.amount
            self.households[offer.buyer_address].token += offer.amount
            trading_str += f"{offer.seller_address}->{offer.buyer_address}:{offer.amount};"
        
        self.add_block(trading_str)

    
