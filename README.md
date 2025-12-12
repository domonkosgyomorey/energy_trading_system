# Energy Trading System

A multi-agent simulation framework for peer-to-peer (P2P) energy trading among households with solar panels, batteries, and smart optimization algorithms.

## 🎯 Project Goal

This project simulates a local energy community where households can:
- Generate solar energy
- Store energy in batteries (individual or shared)
- Trade surplus energy with neighbors via P2P transactions
- Optimize their energy strategy using convex optimization or greedy algorithms
- Compare baseline (no optimization) vs. optimized scenarios

## ✨ Features

- **Multi-household simulation** with configurable number of homes
- **Solar generation** with realistic patterns
- **Battery systems**: Simple, Central, or Shared battery models
- **P2P Energy Trading** with blockchain-based transaction logging
- **Multiple optimizers**: Greedy and Convex (CVXPY-based)
- **Price forecasting** for city grid buy/sell prices
- **Real-time visualization** with playback controls
- **Detailed analytics**: Per-household metrics, equity scores (Gini coefficient)
- **Collapsible parameter groups** for easy configuration

## 🏗️ Architecture

```
energy_trading_system/
├── gui.py                    # Main GUI application
├── visualizer.py             # Simulation playback & analysis
├── main.py                   # CLI entry point
├── simulation/
│   ├── unified_simulator.py  # Core simulation engine
│   ├── household.py          # Household agent model
│   ├── params.py             # Simulation parameters
│   ├── data_collector.py     # Results collection
│   ├── blockchain.py         # P2P transaction ledger
│   ├── battery/              # Battery implementations
│   ├── optimizer/            # Optimization algorithms
│   ├── forecaster/           # Energy forecasting
│   └── local_price_estimator/ # P2P price calculation
└── simulation_results/       # Saved simulation outputs
```

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/energy_trading_system.git
cd energy_trading_system

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

## 💻 Usage

### GUI Mode (Recommended)

```bash
uv run python gui.py
```

The GUI provides:
- **Parameters Tab**: Configure all simulation settings (collapsible groups)
- **Summary Tab**: View per-household metrics and equity analysis
- **Visualizer**: Real-time playback with Analysis and Households tabs

### CLI Mode

```bash
uv run python main.py
```

## ⚙️ Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_households` | Number of households | 5 |
| `n_steps` | Simulation steps | 48 |
| `battery_capacity` | Battery capacity (kWh) | 10.0 |
| `solar_peak_power` | Peak solar generation (kW) | 5.0 |
| `grid_buy_price` | City grid buy price ($/kWh) | 0.15 |
| `grid_sell_price` | City grid sell price ($/kWh) | 0.05 |
| `optimizer_type` | "greedy" or "convex" | "convex" |

## 📊 Visualization

The visualizer includes three tabs:

1. **📺 Playback**: Step-by-step simulation with network graph and time-series plots
2. **📊 Analysis**: Comparison bar charts (wallet, grid usage, P2P volume, savings)
3. **🏠 Households**: Per-household wallet evolution and savings rankings

## 📈 Metrics

- **Financial**: Total wallet balance, cost savings percentage
- **Energy**: Grid buy/sell amounts, self-consumption ratio
- **P2P Trading**: Transaction volume, participation rate
- **Equity**: Gini coefficient for fair benefit distribution

## 🔧 Technologies

- Python 3.12+
- Tkinter (GUI)
- CVXPY (Convex optimization)
- Pandas (Data handling)
- Matplotlib (Visualization)
- NumPy (Numerical computation)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
