from simulation.battery.battery import Battery
from abc import ABC, abstractmethod

class ResourceHandler(ABC):
    
    @abstractmethod
    def handle(self, production: float, consumption: float, battery: Battery, token: float) -> None:
        ...

