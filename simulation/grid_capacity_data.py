"""
Grid capacity data model and loader.
Provides time-series data for city grid import/export capacity limits (in kW).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Literal


@dataclass
class GridCapacityData:
    """
    Holds grid capacity time-series data.
    
    Columns expected:
        - timestep: int (simulation step index)
        - import_capacity_kw: float (max power that can be bought from grid)
        - export_capacity_kw: float (max power that can be sold to grid)
    """
    data: pd.DataFrame
    
    def get_capacity(self, timestep: int) -> dict[Literal["import", "export"], float]:
        """Get grid capacity for a specific timestep."""
        if timestep >= len(self.data):
            # Use last available value if beyond data range
            row = self.data.iloc[-1]
        else:
            row = self.data.iloc[timestep]
        
        return {
            "import": float(row["import_capacity_kw"]),
            "export": float(row["export_capacity_kw"])
        }
    
    def get_import_capacity(self, timestep: int) -> float:
        """Get max import capacity (kW) at timestep."""
        return self.get_capacity(timestep)["import"]
    
    def get_export_capacity(self, timestep: int) -> float:
        """Get max export capacity (kW) at timestep."""
        return self.get_capacity(timestep)["export"]
    
    @classmethod
    def from_file(cls, filepath: str) -> "GridCapacityData":
        """Load grid capacity data from CSV or Parquet file."""
        path = Path(filepath)
        
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        required_cols = {"timestep", "import_capacity_kw", "export_capacity_kw"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        df = df.sort_values("timestep").reset_index(drop=True)
        return cls(data=df)
    
    @classmethod
    def create_constant(cls, 
                       steps: int,
                       import_capacity_kw: float = 10000.0,
                       export_capacity_kw: float = 8000.0) -> "GridCapacityData":
        """Create constant capacity data (no variation over time)."""
        df = pd.DataFrame({
            "timestep": range(steps),
            "import_capacity_kw": [import_capacity_kw] * steps,
            "export_capacity_kw": [export_capacity_kw] * steps
        })
        return cls(data=df)
    
    def save(self, filepath: str) -> None:
        """Save grid capacity data to file."""
        path = Path(filepath)
        
        if path.suffix == ".parquet":
            self.data.to_parquet(path, index=False)
        elif path.suffix == ".csv":
            self.data.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


def generate_synthetic_grid_capacity(
    steps: int,
    base_import_kw: float = 5000.0,
    base_export_kw: float = 4000.0,
    peak_hour_reduction: float = 0.5,
    noise_std: float = 0.1,
    seed: int | None = None
) -> GridCapacityData:
    """
    Generate synthetic grid capacity data with realistic daily patterns.
    
    Args:
        steps: Number of simulation timesteps
        base_import_kw: Base import capacity during normal hours
        base_export_kw: Base export capacity during normal hours
        peak_hour_reduction: Capacity reduction factor during peak hours (0-1)
        noise_std: Standard deviation of random noise (fraction of base)
        seed: Random seed for reproducibility
    
    Returns:
        GridCapacityData with time-varying capacity
    """
    if seed is not None:
        np.random.seed(seed)
    
    timesteps = np.arange(steps)
    
    # Simulate daily pattern (assuming each step is ~1 day, scaled appropriately)
    # Peak reduction happens periodically
    cycle_length = 7  # Weekly cycle
    cycle_position = timesteps % cycle_length
    
    # Reduce capacity during "peak" days (e.g., weekdays 2-5)
    is_peak = (cycle_position >= 2) & (cycle_position <= 5)
    
    import_multiplier = np.where(is_peak, peak_hour_reduction, 1.0)
    export_multiplier = np.where(is_peak, peak_hour_reduction, 1.0)
    
    # Add noise
    import_noise = 1.0 + np.random.normal(0, noise_std, steps)
    export_noise = 1.0 + np.random.normal(0, noise_std, steps)
    
    import_capacity = base_import_kw * import_multiplier * np.clip(import_noise, 0.5, 1.5)
    export_capacity = base_export_kw * export_multiplier * np.clip(export_noise, 0.5, 1.5)
    
    df = pd.DataFrame({
        "timestep": timesteps,
        "import_capacity_kw": import_capacity,
        "export_capacity_kw": export_capacity
    })
    
    return GridCapacityData(data=df)
