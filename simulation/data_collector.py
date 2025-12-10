"""
Data collector for simulation visualization.
Uses observer pattern to collect data from both baseline and optimized simulations.
"""
from dataclasses import dataclass, field
from typing import Literal, Callable, Protocol
import pandas as pd
from threading import Lock


SimulationType = Literal["baseline", "optimized"]


@dataclass
class StepData:
    """Data captured at each simulation step for a single household."""
    step: int
    household_id: str
    production: float
    consumption: float
    stored_kwh: float
    wallet: float
    grid_buy: float = 0.0
    grid_sell: float = 0.0
    p2p_trades: str = ""
    central_battery_tax: float = 0.0
    grid_import_capacity: float = -1.0  # -1 means unlimited
    grid_export_capacity: float = -1.0  # -1 means unlimited
    battery_pct: float = 0.0  # Battery charge percentage (0-100)
    p2p_buy_amount: float = 0.0  # Total energy bought via P2P
    p2p_sell_amount: float = 0.0  # Total energy sold via P2P


@dataclass
class AggregatedStepData:
    """Aggregated data for all households at a simulation step."""
    step: int
    total_production: float
    total_consumption: float
    total_stored_kwh: float
    total_wallet: float
    total_grid_buy: float
    total_grid_sell: float
    household_count: int


class SimulationObserver(Protocol):
    """Protocol for simulation data observers."""
    def on_step_complete(self, sim_type: SimulationType, step: int, data: list[StepData]) -> None:
        ...
    
    def on_simulation_complete(self, sim_type: SimulationType) -> None:
        ...


class SimulationDataCollector:
    """
    Collects and stores simulation data for visualization.
    Thread-safe for concurrent access from GUI.
    """
    
    def __init__(self):
        self._lock = Lock()
        self._baseline_data: list[StepData] = []
        self._optimized_data: list[StepData] = []
        self._observers: list[SimulationObserver] = []
        self._current_step: dict[SimulationType, int] = {"baseline": 0, "optimized": 0}
        
    def add_observer(self, observer: SimulationObserver) -> None:
        """Register an observer to receive data updates."""
        with self._lock:
            self._observers.append(observer)
    
    def remove_observer(self, observer: SimulationObserver) -> None:
        """Remove a registered observer."""
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)
    
    def record_step(self, sim_type: SimulationType, step: int, data: list[StepData]) -> None:
        """Record data from a simulation step."""
        with self._lock:
            storage = self._baseline_data if sim_type == "baseline" else self._optimized_data
            storage.extend(data)
            self._current_step[sim_type] = step
            
            # Notify observers
            for observer in self._observers:
                observer.on_step_complete(sim_type, step, data)
    
    def notify_complete(self, sim_type: SimulationType) -> None:
        """Notify that simulation has completed."""
        with self._lock:
            for observer in self._observers:
                observer.on_simulation_complete(sim_type)
    
    def get_dataframe(self, sim_type: SimulationType) -> pd.DataFrame:
        """Get all recorded data as DataFrame."""
        with self._lock:
            data = self._baseline_data if sim_type == "baseline" else self._optimized_data
            if not data:
                return pd.DataFrame()
            
            return pd.DataFrame([
                {
                    "step": d.step,
                    "household_id": d.household_id,
                    "production": d.production,
                    "consumption": d.consumption,
                    "stored_kwh": d.stored_kwh,
                    "battery_pct": d.battery_pct,
                    "wallet": d.wallet,
                    "grid_buy": d.grid_buy,
                    "grid_sell": d.grid_sell,
                    "p2p_trades": d.p2p_trades,
                    "p2p_buy_amount": d.p2p_buy_amount,
                    "p2p_sell_amount": d.p2p_sell_amount,
                    "central_battery_tax": d.central_battery_tax,
                    "grid_import_capacity": d.grid_import_capacity if d.grid_import_capacity != float('inf') else -1,
                    "grid_export_capacity": d.grid_export_capacity if d.grid_export_capacity != float('inf') else -1,
                }
                for d in data
            ])
    
    def get_aggregated_by_step(self, sim_type: SimulationType) -> pd.DataFrame:
        """Get data aggregated by step (totals across all households)."""
        df = self.get_dataframe(sim_type)
        if df.empty:
            return df
        
        return df.groupby("step").agg({
            "production": "sum",
            "consumption": "sum",
            "stored_kwh": "sum",
            "wallet": "sum",
            "grid_buy": "sum",
            "grid_sell": "sum",
            "p2p_buy_amount": "sum",
            "p2p_sell_amount": "sum",
            "household_id": "count"
        }).rename(columns={"household_id": "household_count"}).reset_index()
    
    def get_stats_by_step(self, sim_type: SimulationType) -> pd.DataFrame:
        """Get statistics (mean, std) by step for confidence intervals."""
        df = self.get_dataframe(sim_type)
        if df.empty:
            return df
        
        agg_funcs = {
            "production": ["mean", "std"],
            "consumption": ["mean", "std"],
            "stored_kwh": ["mean", "std"],
            "battery_pct": ["mean", "std"],
            "wallet": ["mean", "std", "sum"],
            "grid_buy": ["mean", "std", "sum"],
            "grid_sell": ["mean", "std", "sum"],
            "p2p_buy_amount": ["mean", "std", "sum"],
            "p2p_sell_amount": ["mean", "std", "sum"],
        }
        
        result = df.groupby("step").agg(agg_funcs)
        # Flatten column names
        result.columns = ['_'.join(col).strip() for col in result.columns.values]
        return result.reset_index()
    
    def get_current_step(self, sim_type: SimulationType) -> int:
        """Get the current step number for a simulation type."""
        with self._lock:
            return self._current_step.get(sim_type, 0)
    
    def clear(self) -> None:
        """Clear all collected data."""
        with self._lock:
            self._baseline_data.clear()
            self._optimized_data.clear()
            self._current_step = {"baseline": 0, "optimized": 0}
    
    def export_to_csv(self, sim_type: SimulationType, filepath: str) -> None:
        """Export collected data to CSV file."""
        df = self.get_dataframe(sim_type)
        df.to_csv(filepath, index=False)
