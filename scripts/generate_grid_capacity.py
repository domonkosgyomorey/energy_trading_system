"""
Script to generate synthetic grid capacity data.
Run this to create test data for the simulation.
"""
import argparse
from pathlib import Path
from simulation.grid_capacity_data import generate_synthetic_grid_capacity


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic grid capacity data")
    
    parser.add_argument("-o", "--output", type=str, default="grid_capacity.csv",
                        help="Output file path (supports .csv and .parquet)")
    parser.add_argument("-s", "--steps", type=int, default=90,
                        help="Number of simulation timesteps")
    parser.add_argument("--import-kw", type=float, default=5000.0,
                        help="Base import capacity in kW")
    parser.add_argument("--export-kw", type=float, default=4000.0,
                        help="Base export capacity in kW")
    parser.add_argument("--peak-reduction", type=float, default=0.5,
                        help="Capacity reduction during peak hours (0-1)")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="Random noise standard deviation")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print(f"Generating grid capacity data...")
    print(f"  Steps: {args.steps}")
    print(f"  Base Import: {args.import_kw} kW")
    print(f"  Base Export: {args.export_kw} kW")
    print(f"  Peak Reduction: {args.peak_reduction}")
    
    data = generate_synthetic_grid_capacity(
        steps=args.steps,
        base_import_kw=args.import_kw,
        base_export_kw=args.export_kw,
        peak_hour_reduction=args.peak_reduction,
        noise_std=args.noise,
        seed=args.seed
    )
    
    output_path = Path(args.output)
    data.save(str(output_path))
    
    print(f"\nData saved to: {output_path.absolute()}")
    print(f"\nSample data:")
    print(data.data.head(10).to_string())


if __name__ == "__main__":
    main()
