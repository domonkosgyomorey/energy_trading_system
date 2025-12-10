# Energy Trading System

> Self-sustaining energy trading simulator with blockchain technology, P2P trading optimization, and grid capacity constraints.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This simulator models a community of households that:
- **Generate** energy (e.g., solar panels)
- **Consume** energy based on real consumption patterns
- **Trade** energy peer-to-peer using blockchain technology
- **Optimize** trading decisions using convex optimization
- **Store** excess energy in individual or shared batteries

The system compares a **baseline simulation** (no optimization) against an **optimized simulation** (with P2P trading and battery scheduling) to measure efficiency gains.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔋 **Battery Systems** | Simple batteries, shared batteries, and central community battery |
| ⚡ **Grid Capacity Limits** | Realistic grid constraints with time-varying import/export limits |
| 🔗 **Blockchain Trading** | Secure P2P energy trading with transaction logging |
| 📊 **Convex Optimizer** | CVXPY-based optimization with transaction costs for realistic P2P trading |
| 🖥️ **GUI Application** | Parameter configuration with tooltips, real-time progress visualization |
| 🎬 **Visualizer App** | Playback saved simulations with circular network view and P2P arrows |
| 📈 **Data Collection** | Auto-saves results with timestamps for later analysis |
| ⚙️ **Parameter System** | JSON-based configuration with comprehensive tooltips |

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- [uv](https://pypi.org/project/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/domonkosgyomorey/energy_trading_system.git
cd energy_trading_system

# Create and activate virtual environment
uv venv
.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
uv pip install -e .
# or: pip install -e .
```

### Running the Simulation

**Option 1: GUI Application** (Recommended)
```bash
python gui.py
```

**Option 2: Visualizer Playback** (For viewing saved results)
```bash
python visualizer.py
```

**Option 3: Command Line**
```bash
python main.py
```

**Option 3: Generate Grid Capacity Data**
```bash
python scripts/generate_grid_capacity.py --output grid_data.csv --steps 90
```

## 🏗️ Architecture

```
energy_trading_system/
├── gui.py                          # Tkinter GUI application
├── visualizer.py                   # Playback visualizer for saved results
├── main.py                         # CLI entry point
├── simulation/
│   ├── params.py                   # Centralized parameter system
│   ├── unified_simulator.py        # Synchronized baseline + optimized runner
│   ├── data_collector.py           # Observer pattern data collection
│   ├── grid_capacity_data.py       # Grid capacity model & generators
│   ├── household.py                # Household model
│   ├── blockchain.py               # P2P trading blockchain
│   ├── battery/
│   │   ├── simple_battery.py       # Individual household battery
│   │   ├── shared_battery.py       # Community shared battery
│   │   └── central_battery.py      # Central storage facility
│   ├── optimizer/
│   │   ├── optimizer.py            # Optimizer protocol/interface
│   │   └── convex_optimizer.py     # CVXPY implementation
│   ├── forecaster/
│   │   └── perfect_forecaster.py   # Production/consumption forecasting
│   └── utils/
│       └── logger.py               # Logging utilities
├── household_dbs/                  # Household data files
└── scripts/
    └── generate_grid_capacity.py   # Synthetic data generator
```

## ⚙️ Configuration

### Parameter System

All simulation parameters are centralized in `simulation/params.py`. Key configuration groups:

| Group | Parameters |
|-------|-----------|
| **Simulation** | `simulation_steps`, `time_step_hours` |
| **Household** | `max_households`, `shared_battery_probability`, `initial_wallet` |
| **Battery** | `simple_capacity_kwh`, `central_capacity_kwh`, charge/discharge efficiencies |
| **Grid Price** | `min_buy_price`, `max_buy_price`, `min_sell_price`, `max_sell_price` |
| **Grid Capacity** | `use_capacity_limits`, `default_import_capacity_kw`, `default_export_capacity_kw` |
| **Forecaster** | `history_size`, `prediction_size` |
| **Optimizer** | `p2p_transaction_cost`, `min_trade_threshold`, `wallet_penalty_weight` |

### Save/Load Parameters

```python
from simulation.params import SimulationParams

# Create and customize
params = SimulationParams()
params.household.max_households = 10
params.grid_capacity.use_capacity_limits = True

# Save to file
params.save("my_config.json")

# Load from file
params = SimulationParams.load("my_config.json")
```

### Grid Capacity Data

Grid capacity can be provided as:
1. **CSV/Parquet file** with `timestep`, `import_capacity_kw`, `export_capacity_kw` columns
2. **Synthetic data** generated via GUI or script
3. **Constant values** using default parameters

```python
from simulation.grid_capacity_data import GridCapacityData, generate_synthetic_grid_capacity

# Load from file
grid_data = GridCapacityData.from_file("grid_capacity.csv")

# Generate synthetic data with peak/off-peak patterns
grid_data = generate_synthetic_grid_capacity(
    steps=90,
    base_import_kw=5000,
    base_export_kw=4000,
    peak_reduction=0.5,
    noise_std=0.1
)

# Create constant capacity
grid_data = GridCapacityData.create_constant(steps=90, import_capacity_kw=10000, export_capacity_kw=8000)
```

## 📊 GUI Features

The GUI (`gui.py`) provides a streamlined interface for simulation control:

### Parameters Tab
- Edit all simulation parameters with organized sections
- **Tooltips** on every parameter explaining its purpose (hover over ⓘ icons)
- Save/load parameter configurations as JSON
- Reset to default values

### Data Tab
- Load household consumption/production data (CSV or Parquet)
- Load or generate grid capacity constraint data
- Generate synthetic grid data with configurable peak/off-peak patterns
- Configure output directory for auto-saved results

### Simulation Tab

The simulation tab shows minimal real-time visualization during simulation:

| Chart | Description |
|-------|-------------|
| 💰 **Wallet** | Total community wallet balance (baseline vs optimized) |
| 🔌 **Grid Buy** | Energy purchased from city grid (baseline vs optimized) |

#### Auto-Save Results
Results are automatically saved with timestamps to `simulation_results/sim_YYYYMMDD_HHMMSS/`:
- `baseline_results.csv` - Baseline simulation data
- `optimized_results.csv` - Optimized simulation data
- `params.json` - Parameters used for the simulation
- `grid_capacity.csv` - Grid capacity data (if available)

#### Launch Visualizer
Click "Launch Visualizer" to open the playback app for detailed analysis of saved results.

### Summary Panel
After simulation completes, a summary shows:
- Final wallet balances (baseline vs optimized)
- Total grid buy/sell amounts
- P2P trading volume
- Wallet improvement and grid buy reduction percentages

## 🎬 Visualizer Playback App

The standalone visualizer (`visualizer.py`) provides rich playback of saved simulation results:

### Features
- **Load simulation folders** - Automatically finds baseline/optimized CSV files
- **Playback controls** - Play/Pause, speed control (0.25x to 10x), step slider
- **Circular network view** - Households in a circle with city grid at center
- **P2P trade arrows** - Purple arrows showing who trades with whom and amounts
- **City grid arrows** - Red (buying) and green (selling) arrows to/from city
- **Emoji icons** - 🏙️ City, 🏠🏡🏚️ Households (status-dependent), 🔋 Battery, 💰💸 Wallet
- **Detailed tooltips** - Hover over any node to see full statistics
- **Toggleable plots** - Show/hide individual charts and confidence interval versions
- **Confidence interval plots** - Mean ± standard deviation bands across households

### Usage
```bash
# Option 1: Launch from GUI after simulation
# Click "Launch Visualizer" button

# Option 2: Launch directly
python visualizer.py
# Then click "Load Simulation Folder" and select a sim_* folder
```

### Plot Types
| Plot | CI Version | Description |
|------|------------|-------------|
| 💰 **Wallet** | ✅ | Cumulative wallet balance |
| 🔌 **Grid Buy** | ✅ | Energy purchased from grid |
| ⚡ **Grid Sell** | ✅ | Energy sold to grid |
| 🔋 **Battery** | ✅ | Battery storage levels |
| ⚡ **Energy** | ❌ | Production vs consumption |
| 🤝 **P2P Trading** | ❌ | Peer-to-peer trades |

### Network Visualization
The circular network view shows:
- **🏙️ City Grid** at the center with total buy/sell amounts
- **🏠 Households** arranged in a circle with:
  - House emoji indicating status (🏡 doing well, 🏚️ struggling, 🏠 normal)
  - Battery percentage with color coding
  - Wallet balance with 💰 or 💸
  - P2P trading amounts (📤 sold, 📥 bought)
- **Arrows showing energy flow**:
  - 🔴 Red: Buying from city grid
  - 🟢 Green: Selling to city grid
  - 🟣 Purple: P2P trades between households (with kWh labels)

## 🔬 How It Works

### Baseline Simulation
1. Each household consumes/produces energy independently
2. Excess energy stored in battery, deficit purchased from grid
3. No inter-household trading or optimization

### Optimized Simulation
1. Forecaster predicts future production/consumption
2. Convex optimizer plans optimal trades considering:
   - Grid price forecasts (buy/sell)
   - Grid capacity constraints (import/export limits)
   - Battery state and efficiency
   - P2P trading opportunities
   - **Transaction costs** (discourages many small trades)
3. Blockchain records all trades securely
4. Households execute optimized trading plan

### P2P Transaction Costs

The optimizer includes transaction costs to model real-world trading friction:
```
objective += transaction_cost × total_P2P_volume
```
This prevents the optimizer from creating many tiny trades that wouldn't be economical in practice. Configure via:
- `p2p_transaction_cost`: Fixed cost per trade (default: $0.50)
- `min_trade_threshold`: Minimum trade size to record (default: 0.1 kWh)

### Grid Capacity Constraints

The optimizer respects community-wide grid limits:
```
Σ(grid_buy) ≤ import_capacity    # Total buying limited
Σ(grid_sell) ≤ export_capacity   # Total selling limited
```

This models real-world scenarios where grid infrastructure has finite capacity.

## 📁 Output Files

Simulation results are auto-saved to `simulation_results/sim_YYYYMMDD_HHMMSS/`:

| File | Description |
|------|-------------|
| `optimized_results.csv` | Optimized simulation results per household per step |
| `baseline_results.csv` | Baseline simulation results per household per step |
| `params.json` | All parameters used for the simulation |
| `grid_capacity.csv` | Grid capacity data (if available) |

### CSV Columns
| Column | Description |
|--------|-------------|
| `step` | Simulation time step |
| `household_id` | Unique household identifier |
| `production` | Energy produced (kWh) |
| `consumption` | Energy consumed (kWh) |
| `stored_kwh` | Battery energy stored (kWh) |
| `battery_pct` | Battery charge percentage (0-100) |
| `wallet` | Current wallet balance ($) |
| `grid_buy` | Energy bought from city grid (kWh) |
| `grid_sell` | Energy sold to city grid (kWh) |
| `p2p_trades` | P2P trade details (format: `seller:amount-seller:amount`) |
| `p2p_buy_amount` | Total P2P energy bought (kWh) |
| `p2p_sell_amount` | Total P2P energy sold (kWh) |

---

## 👨‍💻 Development

### Code Style

We use these VS Code extensions:
- **Black** - Code formatter
- **isort** - Import sorter
- **MyPy** - Type checker

### VS Code Settings

```json
{
  "mypy-type-checker.importStrategy": "fromEnvironment",
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    }
  },
  "isort.args": ["--profile", "black"]
}
```

### Dependency Policy

1. Must be stable and support Python 3.12+
2. Available via pip
3. 50k+ downloads
4. MIT licensed (or compatible)
5. Versions pinned in `pyproject.toml`

### Git Workflow

1. Create feature branch from `dev`
2. Develop and test feature
3. Pull request to `dev` branch
4. After major features, PR from `dev` to `main`
5. Delete merged feature branches

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please follow the development policy above and submit pull requests to the `dev` branch.
