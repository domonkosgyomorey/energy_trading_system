"""
Energy Trading System - Main Entry Point

Usage:
    python main.py          # Run GUI (default)
    python main.py --cli    # Run CLI simulation
    python main.py --help   # Show help
"""
import argparse
import os
import pandas as pd

from simulation.params import SimulationParams
from simulation.utils.logger import logger


def run_cli():
    """Run simulation in CLI mode (legacy behavior)."""
    from simulation.unified_simulator import UnifiedSimulator
    from simulation.data_collector import SimulationDataCollector
    
    params = SimulationParams()
    
    logger.info("Loading db for households")
    household_data = pd.read_parquet(params.paths.household_db_path)
    logger.info("Household db loaded")
    
    collector = SimulationDataCollector()
    sim = UnifiedSimulator(household_data, params, data_collector=collector)
    
    logger.info("Start simulation...")
    baseline_df, optimized_df = sim.run_all()
    logger.info("Simulation ended...")
    
    # Save results
    baseline_df.to_csv("baseline_simulation_result.csv", index=False)
    optimized_df.to_csv("simulation_result.csv", index=False)
    
    print("\n=== Results ===")
    print(f"Baseline final wallet: ${baseline_df.groupby('step')['wallet'].sum().iloc[-1]:.2f}")
    print(f"Optimized final wallet: ${optimized_df.groupby('step')['wallet'].sum().iloc[-1]:.2f}")


def run_gui():
    """Run GUI application."""
    from gui import main as gui_main
    gui_main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy Trading Simulation System")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of GUI")
    args = parser.parse_args()
    
    if args.cli:
        run_cli()
    else:
        run_gui()
