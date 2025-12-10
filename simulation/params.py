"""
Unified parameter system for the energy trading simulation.
All configurable values are centralized here with proper typing and defaults.
"""
from dataclasses import dataclass, field, asdict
from typing import Literal
import json
from pathlib import Path


@dataclass
class DataPaths:
    """Paths to data files."""
    household_db_dir: str = "household_dbs"
    household_db_file: str = "merged2024.parquet"
    grid_capacity_file: str = ""  # Optional: CSV/parquet with grid capacity data
    output_dir: str = "results"
    
    @property
    def household_db_path(self) -> str:
        return str(Path(self.household_db_dir) / self.household_db_file)


@dataclass
class BatteryParams:
    """Parameters for battery configuration."""
    simple_capacity_kwh: float = 100.0
    simple_charge_efficiency: float = 1.0
    simple_discharge_efficiency: float = 1.0
    simple_initial_charge_kwh: float = 5.0
    
    central_capacity_kwh: float = 10_000_000.0
    central_charge_efficiency: float = 1.0
    central_discharge_efficiency: float = 1.0
    central_tax_per_kwh: float = 0.01


@dataclass 
class GridPriceParams:
    """Parameters for city grid pricing."""
    min_buy_price: float = 9.0
    max_buy_price: float = 50.0
    min_sell_price: float = 0.01
    max_sell_price: float = 15.0


@dataclass
class GridCapacityParams:
    """Parameters for city grid capacity limits."""
    use_capacity_limits: bool = True  # Enable grid capacity constraints
    default_import_capacity_kw: float = 10000.0  # Max power importable from grid
    default_export_capacity_kw: float = 8000.0   # Max power exportable to grid
    # Synthetic data generation parameters
    peak_reduction_factor: float = 0.5  # Capacity reduction during peak periods
    noise_std: float = 0.1  # Random variation in capacity (fraction of base)


@dataclass
class ForecasterParams:
    """Parameters for forecasting."""
    history_size: int = 10
    prediction_size: int = 5  # Reduced from 10 for faster optimization
    city_grid_price_prediction_size: int = 5  # Should match prediction_size


@dataclass
class HouseholdParams:
    """Parameters for household configuration."""
    max_households: int = 5
    shared_battery_probability: float = 0.2  # Probability household uses shared battery
    production_scale_factor: float = 0.65    # Scale factor for production data
    initial_wallet: float = 0.0


@dataclass
class OptimizerParams:
    """Parameters for the convex optimizer."""
    battery_charge_rate_factor: float = 0.2  # Max charge/discharge rate as fraction of capacity
    wallet_penalty_weight: float = 100.0     # Penalty for negative wallet balance
    p2p_transaction_cost: float = 0.5        # Fixed cost per P2P transaction (discourages tiny trades)
    min_trade_threshold: float = 0.1         # Minimum kWh for a trade to be considered
    solver: str = "CLARABEL"                 # Solver: CLARABEL (fast), ECOS, OSQP, SCS (slow but robust)
    warm_start: bool = True                  # Reuse previous solution as starting point


@dataclass
class SimulationParams:
    """Master parameter class containing all simulation configuration."""
    paths: DataPaths = field(default_factory=DataPaths)
    battery: BatteryParams = field(default_factory=BatteryParams)
    grid_price: GridPriceParams = field(default_factory=GridPriceParams)
    grid_capacity: GridCapacityParams = field(default_factory=GridCapacityParams)
    forecaster: ForecasterParams = field(default_factory=ForecasterParams)
    household: HouseholdParams = field(default_factory=HouseholdParams)
    optimizer: OptimizerParams = field(default_factory=OptimizerParams)
    
    simulation_steps: int = 90
    time_step_hours: float = 24.0  # Each step represents this many hours
    
    def save(self, filepath: str) -> None:
        """Save parameters to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "SimulationParams":
        """Load parameters from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(
            paths=DataPaths(**data.get('paths', {})),
            battery=BatteryParams(**data.get('battery', {})),
            grid_price=GridPriceParams(**data.get('grid_price', {})),
            grid_capacity=GridCapacityParams(**data.get('grid_capacity', {})),
            forecaster=ForecasterParams(**data.get('forecaster', {})),
            household=HouseholdParams(**data.get('household', {})),
            optimizer=OptimizerParams(**data.get('optimizer', {})),
            simulation_steps=data.get('simulation_steps', 90),
            time_step_hours=data.get('time_step_hours', 24.0)
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


# Default parameters instance for backward compatibility
DEFAULT_PARAMS = SimulationParams()
