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
    simple_capacity_kwh: float = 13.5  # Tesla Powerwall 2 capacity
    simple_charge_efficiency: float = 0.90  # Typical Li-ion efficiency
    simple_discharge_efficiency: float = 0.90  # Typical Li-ion efficiency
    simple_initial_charge_kwh: float = 3.0  # Start at ~20% charge
    
    central_capacity_kwh: float = 1000.0  # Community battery (realistic for 20-50 households)
    central_charge_efficiency: float = 0.95  # Larger batteries are more efficient
    central_discharge_efficiency: float = 0.95
    central_tax_per_kwh: float = 0.02  # $0.02/kWh service fee for shared battery


@dataclass 
class GridPriceParams:
    """Parameters for city grid pricing ($/kWh)."""
    min_buy_price: float = 0.12  # Off-peak rate (~$0.12/kWh US average)
    max_buy_price: float = 0.35  # Peak rate (~$0.35/kWh during high demand)
    min_sell_price: float = 0.03  # Low feed-in tariff
    max_sell_price: float = 0.10  # Good feed-in tariff (net metering)


@dataclass
class GridCapacityParams:
    """Parameters for city grid capacity limits."""
    use_capacity_limits: bool = True  # Enable grid capacity constraints
    default_import_capacity_kw: float = 500.0  # Realistic for neighborhood transformer (~100 homes)
    default_export_capacity_kw: float = 400.0  # Slightly lower export capacity
    # Synthetic data generation parameters
    peak_reduction_factor: float = 0.6  # 40% capacity reduction during peak (realistic constraint)
    noise_std: float = 0.05  # 5% random variation (more realistic)


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
    initial_wallet: float = 100.0  # Start with $100 credit (realistic prepaid energy account)


@dataclass
class OptimizerParams:
    """Parameters for the optimizer."""
    optimizer_type: str = "greedy"           # "convex" or "greedy" - greedy is faster but less optimal
    battery_charge_rate_factor: float = 0.5  # Max charge rate: 50% of capacity per hour (Powerwall: ~7kW for 13.5kWh)
    wallet_penalty_weight: float = 1000.0    # Strong penalty for debt (realistic credit constraint)
    p2p_transaction_cost: float = 0.25       # $0.25 per transaction (realistic blockchain/platform fee)
    min_trade_threshold: float = 0.5         # Minimum 0.5 kWh per trade (avoid micro-transactions)
    solver: str = "CLARABEL"                 # Solver: CLARABEL (fast), ECOS, OSQP, SCS (slow but robust)
    warm_start: bool = True                  # Reuse previous solution as starting point
    max_neighbors: int | None = 5            # Max neighbors each household can trade with (None = all)
    p2p_price_factor: float = 0.80           # P2P price = 80% of grid buy price (20% savings incentive)
    battery_target_pct: float = 0.50         # Target battery level for greedy optimizer (0-1)


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
    
    simulation_steps: int = 90  # Number of simulation steps to run
    time_step_hours: float = 24.0  # Duration of each step in hours
    
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
