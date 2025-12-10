"""
GUI Application for Energy Trading Simulation.
Provides parameter configuration, data loading, and minimal real-time visualization.
Results are automatically saved with timestamps for later analysis in the visualizer.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import pandas as pd
from pathlib import Path
from datetime import datetime

from simulation.params import SimulationParams
from simulation.grid_capacity_data import GridCapacityData, generate_synthetic_grid_capacity
from simulation.data_collector import SimulationDataCollector, StepData, SimulationObserver
from simulation.unified_simulator import UnifiedSimulator

# Attempt to import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class SimulationGUI(tk.Tk, SimulationObserver):
    """Main GUI application window."""
    
    def __init__(self):
        super().__init__()
        
        self.title("Energy Trading Simulation")
        self.geometry("1000x700")
        self.minsize(800, 600)
        
        # State
        self.params = SimulationParams()
        self.household_data: pd.DataFrame | None = None
        self.grid_capacity: GridCapacityData | None = None
        self.data_collector = SimulationDataCollector()
        self.data_collector.add_observer(self)
        self.simulation_thread: threading.Thread | None = None
        self.is_running = False
        self.last_save_path: Path | None = None
        
        self._create_widgets()
        self._layout_widgets()
        
    def _create_widgets(self) -> None:
        """Create all GUI widgets."""
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self)
        
        # Tab 1: Parameters
        self.params_frame = ttk.Frame(self.notebook, padding=10)
        self._create_params_tab()
        
        # Tab 2: Data Loading
        self.data_frame = ttk.Frame(self.notebook, padding=10)
        self._create_data_tab()
        
        # Tab 3: Simulation Control
        self.sim_frame = ttk.Frame(self.notebook, padding=10)
        self._create_simulation_tab()
        
        self.notebook.add(self.params_frame, text="Parameters")
        self.notebook.add(self.data_frame, text="Data")
        self.notebook.add(self.sim_frame, text="Simulation")
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        
    def _create_params_tab(self) -> None:
        """Create parameter configuration widgets."""
        # Scrollable frame for parameters
        canvas = tk.Canvas(self.params_frame)
        scrollbar = ttk.Scrollbar(self.params_frame, orient="vertical", command=canvas.yview)
        self.params_inner = ttk.Frame(canvas)
        
        self.params_inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.params_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Parameter entry variables
        self.param_vars: dict[str, tk.Variable] = {}
        
        row = 0
        
        # Simulation section
        row = self._add_section_header(self.params_inner, "Simulation", row)
        row = self._add_param_entry(self.params_inner, "simulation_steps", "Simulation Steps", self.params.simulation_steps, row,
            "Total number of time steps to simulate. Each step represents one time_step_hours period.")
        row = self._add_param_entry(self.params_inner, "time_step_hours", "Time Step (hours)", self.params.time_step_hours, row,
            "Duration of each simulation step in hours. E.g., 24 = daily resolution, 1 = hourly resolution.")
        
        # Household section
        row = self._add_section_header(self.params_inner, "Household", row)
        row = self._add_param_entry(self.params_inner, "max_households", "Max Households", self.params.household.max_households, row,
            "Maximum number of households to include in the simulation. More households = longer computation time.")
        row = self._add_param_entry(self.params_inner, "shared_battery_probability", "Shared Battery Prob", self.params.household.shared_battery_probability, row,
            "Probability (0-1) that a household uses the shared central battery instead of having its own battery.")
        row = self._add_param_entry(self.params_inner, "initial_wallet", "Initial Wallet ($)", self.params.household.initial_wallet, row,
            "Starting wallet balance for each household. Negative values are allowed.")
        
        # Battery section
        row = self._add_section_header(self.params_inner, "Battery", row)
        row = self._add_param_entry(self.params_inner, "simple_capacity_kwh", "Simple Capacity (kWh)", self.params.battery.simple_capacity_kwh, row,
            "Energy storage capacity for individual household batteries in kilowatt-hours.")
        row = self._add_param_entry(self.params_inner, "central_capacity_kwh", "Central Capacity (kWh)", self.params.battery.central_capacity_kwh, row,
            "Energy storage capacity for the shared community central battery in kilowatt-hours.")
        row = self._add_param_entry(self.params_inner, "simple_charge_efficiency", "Charge Efficiency", self.params.battery.simple_charge_efficiency, row,
            "Battery charging efficiency (0-1). A value of 0.9 means 10% energy loss when charging.")
        row = self._add_param_entry(self.params_inner, "simple_discharge_efficiency", "Discharge Efficiency", self.params.battery.simple_discharge_efficiency, row,
            "Battery discharging efficiency (0-1). A value of 0.9 means 10% energy loss when discharging.")
        
        # Grid Price section
        row = self._add_section_header(self.params_inner, "Grid Prices", row)
        row = self._add_param_entry(self.params_inner, "min_buy_price", "Min Buy Price ($/kWh)", self.params.grid_price.min_buy_price, row,
            "Minimum price to buy energy from the city grid. Prices vary based on supply/demand.")
        row = self._add_param_entry(self.params_inner, "max_buy_price", "Max Buy Price ($/kWh)", self.params.grid_price.max_buy_price, row,
            "Maximum price to buy energy from the city grid during peak demand periods.")
        row = self._add_param_entry(self.params_inner, "min_sell_price", "Min Sell Price ($/kWh)", self.params.grid_price.min_sell_price, row,
            "Minimum price when selling excess energy back to the city grid.")
        row = self._add_param_entry(self.params_inner, "max_sell_price", "Max Sell Price ($/kWh)", self.params.grid_price.max_sell_price, row,
            "Maximum price when selling excess energy back to the city grid during high demand.")
        
        # Grid Capacity section
        row = self._add_section_header(self.params_inner, "Grid Capacity", row)
        row = self._add_checkbox_entry(self.params_inner, "use_capacity_limits", "Enable Capacity Limits", self.params.grid_capacity.use_capacity_limits, row,
            "When enabled, limits how much energy the community can import/export from the city grid at once.")
        row = self._add_param_entry(self.params_inner, "default_import_capacity_kw", "Default Import (kW)", self.params.grid_capacity.default_import_capacity_kw, row,
            "Maximum power (kW) that can be imported from the city grid at any time step.")
        row = self._add_param_entry(self.params_inner, "default_export_capacity_kw", "Default Export (kW)", self.params.grid_capacity.default_export_capacity_kw, row,
            "Maximum power (kW) that can be exported to the city grid at any time step.")
        
        # Forecaster section
        row = self._add_section_header(self.params_inner, "Forecaster", row)
        row = self._add_param_entry(self.params_inner, "history_size", "History Size", self.params.forecaster.history_size, row,
            "Number of past time steps the forecaster uses to predict future values.")
        row = self._add_param_entry(self.params_inner, "prediction_size", "Prediction Size", self.params.forecaster.prediction_size, row,
            "Number of future time steps the optimizer looks ahead when planning trades. Lower = faster but less optimal.")
        
        # Optimizer section
        row = self._add_section_header(self.params_inner, "Optimizer", row)
        row = self._add_param_entry(self.params_inner, "p2p_transaction_cost", "P2P Transaction Cost ($)", self.params.optimizer.p2p_transaction_cost, row,
            "Fixed cost per P2P trade. Higher values discourage small trades, leading to fewer but larger transactions.")
        row = self._add_param_entry(self.params_inner, "min_trade_threshold", "Min Trade Threshold (kWh)", self.params.optimizer.min_trade_threshold, row,
            "Minimum energy amount for a P2P trade. Trades below this threshold are ignored to reduce noise.")
        row = self._add_param_entry(self.params_inner, "wallet_penalty_weight", "Wallet Penalty Weight", self.params.optimizer.wallet_penalty_weight, row,
            "Penalty multiplier for negative wallet balances. Higher values make the optimizer avoid debt more strongly.")
        row = self._add_combobox_entry(self.params_inner, "solver", "Solver", self.params.optimizer.solver, 
            ["CLARABEL", "ECOS", "OSQP", "SCS"], row,
            "Optimization solver. CLARABEL/ECOS are fast, SCS is slower but more robust. Try different solvers if one fails.")
        row = self._add_checkbox_entry(self.params_inner, "warm_start", "Enable Warm Start", self.params.optimizer.warm_start, row,
            "Reuse previous solution as starting point. Speeds up consecutive optimizations significantly.")
        
        # Button row
        btn_frame = ttk.Frame(self.params_inner)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=20)
        
        ttk.Button(btn_frame, text="Save Params...", command=self._save_params).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Load Params...", command=self._load_params).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reset to Default", command=self._reset_params).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Apply Changes", command=self._apply_params).pack(side=tk.LEFT, padx=5)
        
    def _add_section_header(self, parent: ttk.Frame, text: str, row: int) -> int:
        """Add a section header."""
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 5))
        ttk.Label(parent, text=text, font=("TkDefaultFont", 10, "bold")).grid(row=row+1, column=0, columnspan=3, sticky="w", pady=(0, 5))
        return row + 2
    
    def _add_param_entry(self, parent: ttk.Frame, key: str, label: str, default: float | int, row: int, tooltip: str = "") -> int:
        """Add a parameter entry field with optional tooltip."""
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", padx=5)
        
        if isinstance(default, int):
            var = tk.IntVar(value=default)
        else:
            var = tk.DoubleVar(value=default)
        
        self.param_vars[key] = var
        entry = ttk.Entry(parent, textvariable=var, width=15)
        entry.grid(row=row, column=1, padx=5, pady=2)
        
        # Add tooltip if provided
        if tooltip:
            tooltip_lbl = ttk.Label(parent, text="ⓘ", foreground="blue", cursor="hand2")
            tooltip_lbl.grid(row=row, column=2, padx=2)
            self._create_tooltip(tooltip_lbl, tooltip)
            self._create_tooltip(lbl, tooltip)
            self._create_tooltip(entry, tooltip)
        
        return row + 1
    
    def _add_checkbox_entry(self, parent: ttk.Frame, key: str, label: str, default: bool, row: int, tooltip: str = "") -> int:
        """Add a checkbox parameter with optional tooltip."""
        var = tk.BooleanVar(value=default)
        self.param_vars[key] = var
        cb = ttk.Checkbutton(parent, text=label, variable=var)
        cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        
        # Add tooltip if provided
        if tooltip:
            tooltip_lbl = ttk.Label(parent, text="ⓘ", foreground="blue", cursor="hand2")
            tooltip_lbl.grid(row=row, column=2, padx=2)
            self._create_tooltip(tooltip_lbl, tooltip)
            self._create_tooltip(cb, tooltip)
        
        return row + 1
    
    def _add_combobox_entry(self, parent: ttk.Frame, key: str, label: str, default: str, 
                           options: list[str], row: int, tooltip: str = "") -> int:
        """Add a combobox (dropdown) parameter with optional tooltip."""
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", padx=5)
        
        var = tk.StringVar(value=default)
        self.param_vars[key] = var
        
        combo = ttk.Combobox(parent, textvariable=var, values=options, width=12, state="readonly")
        combo.grid(row=row, column=1, padx=5, pady=2)
        
        # Add tooltip if provided
        if tooltip:
            tooltip_lbl = ttk.Label(parent, text="ⓘ", foreground="blue", cursor="hand2")
            tooltip_lbl.grid(row=row, column=2, padx=2)
            self._create_tooltip(tooltip_lbl, tooltip)
            self._create_tooltip(lbl, tooltip)
            self._create_tooltip(combo, tooltip)
        
        return row + 1
    
    def _create_tooltip(self, widget, text: str) -> None:
        """Create a tooltip for a widget."""
        def show_tooltip(event):
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
            
            frame = ttk.Frame(tooltip, relief="solid", borderwidth=1)
            frame.pack()
            
            label = ttk.Label(frame, text=text, wraplength=300, padding=5,
                             background="#ffffe0", foreground="#333333")
            label.pack()
            
            widget._tooltip = tooltip
            
            def hide_tooltip(e=None):
                if hasattr(widget, '_tooltip') and widget._tooltip:
                    widget._tooltip.destroy()
                    widget._tooltip = None
            
            tooltip.bind("<Leave>", hide_tooltip)
            widget.bind("<Leave>", hide_tooltip)
            widget._hide_tooltip = hide_tooltip
        
        widget.bind("<Enter>", show_tooltip)
    
    def _create_data_tab(self) -> None:
        """Create data loading widgets."""
        # Household data section
        hh_frame = ttk.LabelFrame(self.data_frame, text="Household Data", padding=10)
        hh_frame.pack(fill=tk.X, pady=5)
        
        self.hh_path_var = tk.StringVar(value=self.params.paths.household_db_path)
        ttk.Entry(hh_frame, textvariable=self.hh_path_var, width=60).pack(side=tk.LEFT, padx=5)
        ttk.Button(hh_frame, text="Browse...", command=self._browse_household_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(hh_frame, text="Load", command=self._load_household_data).pack(side=tk.LEFT, padx=5)
        
        self.hh_status = ttk.Label(self.data_frame, text="Not loaded")
        self.hh_status.pack(anchor=tk.W, padx=15)
        
        # Grid capacity section
        gc_frame = ttk.LabelFrame(self.data_frame, text="Grid Capacity Data (Optional)", padding=10)
        gc_frame.pack(fill=tk.X, pady=10)
        
        self.gc_path_var = tk.StringVar()
        ttk.Entry(gc_frame, textvariable=self.gc_path_var, width=60).pack(side=tk.LEFT, padx=5)
        ttk.Button(gc_frame, text="Browse...", command=self._browse_grid_capacity).pack(side=tk.LEFT, padx=5)
        ttk.Button(gc_frame, text="Load", command=self._load_grid_capacity).pack(side=tk.LEFT, padx=5)
        
        self.gc_status = ttk.Label(self.data_frame, text="Not loaded (will use constant values)")
        self.gc_status.pack(anchor=tk.W, padx=15)
        
        # Generate synthetic grid capacity
        gen_frame = ttk.LabelFrame(self.data_frame, text="Generate Synthetic Grid Capacity", padding=10)
        gen_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(gen_frame, text="Base Import (kW):").grid(row=0, column=0, padx=5)
        self.gen_import_var = tk.DoubleVar(value=5000)
        ttk.Entry(gen_frame, textvariable=self.gen_import_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(gen_frame, text="Base Export (kW):").grid(row=0, column=2, padx=5)
        self.gen_export_var = tk.DoubleVar(value=4000)
        ttk.Entry(gen_frame, textvariable=self.gen_export_var, width=10).grid(row=0, column=3, padx=5)
        
        ttk.Label(gen_frame, text="Peak Reduction:").grid(row=1, column=0, padx=5, pady=5)
        self.gen_peak_var = tk.DoubleVar(value=0.5)
        ttk.Entry(gen_frame, textvariable=self.gen_peak_var, width=10).grid(row=1, column=1, padx=5)
        
        ttk.Button(gen_frame, text="Generate", command=self._generate_grid_capacity).grid(row=1, column=2, columnspan=2, padx=5)
        
        # Output directory section
        out_frame = ttk.LabelFrame(self.data_frame, text="Output Directory", padding=10)
        out_frame.pack(fill=tk.X, pady=10)
        
        self.output_dir_var = tk.StringVar(value="simulation_results")
        ttk.Entry(out_frame, textvariable=self.output_dir_var, width=60).pack(side=tk.LEFT, padx=5)
        ttk.Button(out_frame, text="Browse...", command=self._browse_output_dir).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(self.data_frame, text="Results will be saved with timestamp in this directory", 
                 font=("TkDefaultFont", 9, "italic")).pack(anchor=tk.W, padx=15)
        
    def _create_simulation_tab(self) -> None:
        """Create simulation control and minimal visualization widgets."""
        # Control panel
        ctrl_frame = ttk.Frame(self.sim_frame)
        ctrl_frame.pack(fill=tk.X, pady=5)
        
        self.start_btn = ttk.Button(ctrl_frame, text="▶ Start Simulation", command=self._start_simulation)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(ctrl_frame, text="⬛ Stop", command=self._stop_simulation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(ctrl_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Button(ctrl_frame, text="📂 Open Results Folder", command=self._open_results_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="🔍 Launch Visualizer", command=self._launch_visualizer).pack(side=tk.LEFT, padx=5)
        
        # Progress
        progress_frame = ttk.Frame(self.sim_frame)
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, length=400)
        self.progress_bar.pack(side=tk.LEFT, padx=5)
        
        self.step_label = ttk.Label(progress_frame, text="Step: 0 / 0")
        self.step_label.pack(side=tk.LEFT, padx=10)
        
        # Simple visualization - just 2 key plots
        if HAS_MATPLOTLIB:
            self._create_simple_viz()
        else:
            self._create_text_viz()
        
        # Results summary
        summary_frame = ttk.LabelFrame(self.sim_frame, text="Summary", padding=10)
        summary_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.summary_text = tk.Text(summary_frame, height=10, state=tk.DISABLED, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
    def _create_simple_viz(self) -> None:
        """Create simple 2-plot visualization."""
        viz_frame = ttk.LabelFrame(self.sim_frame, text="Live Progress", padding=5)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.fig = Figure(figsize=(10, 3), dpi=100)
        
        # Just 2 plots: Wallet and Grid Buy
        self.ax_wallet = self.fig.add_subplot(1, 2, 1)
        self.ax_grid = self.fig.add_subplot(1, 2, 2)
        
        self.fig.tight_layout(pad=2.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _create_text_viz(self) -> None:
        """Create text-based visualization fallback."""
        viz_frame = ttk.LabelFrame(self.sim_frame, text="Live Progress", padding=5)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.text_viz = tk.Text(viz_frame, height=8, state=tk.DISABLED)
        self.text_viz.pack(fill=tk.BOTH, expand=True)
        
    def _layout_widgets(self) -> None:
        """Layout main widgets."""
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
    # --- Parameter management ---
    
    def _apply_params(self) -> None:
        """Apply parameter values from GUI to params object."""
        try:
            self.params.simulation_steps = self.param_vars["simulation_steps"].get()
            self.params.time_step_hours = self.param_vars["time_step_hours"].get()
            
            self.params.household.max_households = self.param_vars["max_households"].get()
            self.params.household.shared_battery_probability = self.param_vars["shared_battery_probability"].get()
            self.params.household.initial_wallet = self.param_vars["initial_wallet"].get()
            
            self.params.battery.simple_capacity_kwh = self.param_vars["simple_capacity_kwh"].get()
            self.params.battery.central_capacity_kwh = self.param_vars["central_capacity_kwh"].get()
            self.params.battery.simple_charge_efficiency = self.param_vars["simple_charge_efficiency"].get()
            self.params.battery.simple_discharge_efficiency = self.param_vars["simple_discharge_efficiency"].get()
            
            self.params.grid_price.min_buy_price = self.param_vars["min_buy_price"].get()
            self.params.grid_price.max_buy_price = self.param_vars["max_buy_price"].get()
            self.params.grid_price.min_sell_price = self.param_vars["min_sell_price"].get()
            self.params.grid_price.max_sell_price = self.param_vars["max_sell_price"].get()
            
            self.params.grid_capacity.use_capacity_limits = self.param_vars["use_capacity_limits"].get()
            self.params.grid_capacity.default_import_capacity_kw = self.param_vars["default_import_capacity_kw"].get()
            self.params.grid_capacity.default_export_capacity_kw = self.param_vars["default_export_capacity_kw"].get()
            
            self.params.forecaster.history_size = self.param_vars["history_size"].get()
            self.params.forecaster.prediction_size = self.param_vars["prediction_size"].get()
            
            self.params.optimizer.p2p_transaction_cost = self.param_vars["p2p_transaction_cost"].get()
            self.params.optimizer.min_trade_threshold = self.param_vars["min_trade_threshold"].get()
            self.params.optimizer.wallet_penalty_weight = self.param_vars["wallet_penalty_weight"].get()
            self.params.optimizer.solver = self.param_vars["solver"].get()
            self.params.optimizer.warm_start = self.param_vars["warm_start"].get()
            
            self.status_var.set("Parameters applied")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply parameters: {e}")
    
    def _save_params(self) -> None:
        """Save parameters to JSON file."""
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save Parameters"
        )
        if path:
            self._apply_params()
            self.params.save(path)
            self.status_var.set(f"Parameters saved to {path}")
    
    def _load_params(self) -> None:
        """Load parameters from JSON file."""
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Load Parameters"
        )
        if path:
            self.params = SimulationParams.load(path)
            self._update_gui_from_params()
            self.status_var.set(f"Parameters loaded from {path}")
    
    def _reset_params(self) -> None:
        """Reset parameters to defaults."""
        self.params = SimulationParams()
        self._update_gui_from_params()
        self.status_var.set("Parameters reset to defaults")
    
    def _update_gui_from_params(self) -> None:
        """Update GUI entries from params object."""
        self.param_vars["simulation_steps"].set(self.params.simulation_steps)
        self.param_vars["time_step_hours"].set(self.params.time_step_hours)
        self.param_vars["max_households"].set(self.params.household.max_households)
        self.param_vars["shared_battery_probability"].set(self.params.household.shared_battery_probability)
        self.param_vars["initial_wallet"].set(self.params.household.initial_wallet)
        self.param_vars["simple_capacity_kwh"].set(self.params.battery.simple_capacity_kwh)
        self.param_vars["central_capacity_kwh"].set(self.params.battery.central_capacity_kwh)
        self.param_vars["simple_charge_efficiency"].set(self.params.battery.simple_charge_efficiency)
        self.param_vars["simple_discharge_efficiency"].set(self.params.battery.simple_discharge_efficiency)
        self.param_vars["min_buy_price"].set(self.params.grid_price.min_buy_price)
        self.param_vars["max_buy_price"].set(self.params.grid_price.max_buy_price)
        self.param_vars["min_sell_price"].set(self.params.grid_price.min_sell_price)
        self.param_vars["max_sell_price"].set(self.params.grid_price.max_sell_price)
        self.param_vars["use_capacity_limits"].set(self.params.grid_capacity.use_capacity_limits)
        self.param_vars["default_import_capacity_kw"].set(self.params.grid_capacity.default_import_capacity_kw)
        self.param_vars["default_export_capacity_kw"].set(self.params.grid_capacity.default_export_capacity_kw)
        self.param_vars["history_size"].set(self.params.forecaster.history_size)
        self.param_vars["prediction_size"].set(self.params.forecaster.prediction_size)
        self.param_vars["p2p_transaction_cost"].set(self.params.optimizer.p2p_transaction_cost)
        self.param_vars["min_trade_threshold"].set(self.params.optimizer.min_trade_threshold)
        self.param_vars["wallet_penalty_weight"].set(self.params.optimizer.wallet_penalty_weight)
        self.param_vars["solver"].set(self.params.optimizer.solver)
        self.param_vars["warm_start"].set(self.params.optimizer.warm_start)
        
    # --- Data loading ---
    
    def _browse_household_data(self) -> None:
        """Browse for household data file."""
        path = filedialog.askopenfilename(
            filetypes=[("Parquet files", "*.parquet"), ("CSV files", "*.csv"), ("All files", "*.*")],
            title="Select Household Data"
        )
        if path:
            self.hh_path_var.set(path)
    
    def _load_household_data(self) -> None:
        """Load household data from file."""
        path = self.hh_path_var.get()
        try:
            if path.endswith('.parquet'):
                self.household_data = pd.read_parquet(path)
            else:
                self.household_data = pd.read_csv(path)
            
            n_households = self.household_data['id'].nunique() if 'id' in self.household_data.columns else 'unknown'
            n_rows = len(self.household_data)
            self.hh_status.config(text=f"✓ Loaded: {n_households} households, {n_rows} rows")
            self.status_var.set(f"Household data loaded from {path}")
        except Exception as e:
            self.hh_status.config(text=f"✗ Error: {e}")
            messagebox.showerror("Error", f"Failed to load household data: {e}")
    
    def _browse_grid_capacity(self) -> None:
        """Browse for grid capacity file."""
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Parquet files", "*.parquet"), ("All files", "*.*")],
            title="Select Grid Capacity Data"
        )
        if path:
            self.gc_path_var.set(path)
    
    def _load_grid_capacity(self) -> None:
        """Load grid capacity data from file."""
        path = self.gc_path_var.get()
        try:
            self.grid_capacity = GridCapacityData.from_file(path)
            n_steps = len(self.grid_capacity.data)
            self.gc_status.config(text=f"✓ Loaded: {n_steps} timesteps")
            self.status_var.set(f"Grid capacity data loaded from {path}")
        except Exception as e:
            self.gc_status.config(text=f"✗ Error: {e}")
            messagebox.showerror("Error", f"Failed to load grid capacity data: {e}")
    
    def _generate_grid_capacity(self) -> None:
        """Generate synthetic grid capacity data."""
        try:
            steps = self.param_vars["simulation_steps"].get()
            self.grid_capacity = generate_synthetic_grid_capacity(
                steps=steps,
                base_import_kw=self.gen_import_var.get(),
                base_export_kw=self.gen_export_var.get(),
                peak_hour_reduction=self.gen_peak_var.get()
            )
            self.gc_status.config(text=f"✓ Generated: {steps} timesteps")
            self.status_var.set("Synthetic grid capacity data generated")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate grid capacity: {e}")
    
    def _browse_output_dir(self) -> None:
        """Browse for output directory."""
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_dir_var.set(path)
    
    # --- Simulation control ---
    
    def _start_simulation(self) -> None:
        """Start the simulation."""
        if self.household_data is None:
            # Try to load default
            try:
                self._load_household_data()
            except:
                pass
            
            if self.household_data is None:
                messagebox.showwarning("Warning", "Please load household data first")
                return
        
        self._apply_params()
        self.data_collector.clear()
        
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.is_running = True
        
        # Clear plots
        if HAS_MATPLOTLIB:
            self.ax_wallet.clear()
            self.ax_grid.clear()
            self.canvas.draw()
        
        # Clear summary
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.config(state=tk.DISABLED)
        
        self.simulation_thread = threading.Thread(target=self._run_simulation, daemon=True)
        self.simulation_thread.start()
        
        self._poll_simulation()
    
    def _stop_simulation(self) -> None:
        """Stop the simulation."""
        self.is_running = False
        self.status_var.set("Stopping simulation...")
    
    def _run_simulation(self) -> None:
        """Run simulation in background thread."""
        try:
            sim = UnifiedSimulator(
                self.household_data,
                self.params,
                self.grid_capacity,
                self.data_collector
            )
            
            baseline_df, optimized_df = sim.run_all()
            
            if self.is_running:
                # Auto-save results with timestamp
                self._auto_save_results()
                
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Simulation Error", str(e)))
        finally:
            self.after(0, self._simulation_finished)
    
    def _poll_simulation(self) -> None:
        """Poll simulation progress and update GUI."""
        if not self.is_running and self.simulation_thread and not self.simulation_thread.is_alive():
            return
        
        # Update progress
        total_steps = self.params.simulation_steps
        current_step = self.data_collector.get_current_step("baseline")
        
        if total_steps > 0:
            self._update_progress(current_step, total_steps)
        
        # Update visualization (throttled)
        if current_step % 5 == 0 or current_step == total_steps:
            self._update_visualization()
        
        if self.is_running:
            self.after(200, self._poll_simulation)
    
    def _simulation_finished(self) -> None:
        """Handle simulation completion."""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        self._update_visualization()
        self._update_summary()
        
        if self.last_save_path:
            self.status_var.set(f"Simulation complete! Results saved to: {self.last_save_path}")
        else:
            self.status_var.set("Simulation complete!")
    
    def _update_progress(self, current: int, total: int) -> None:
        """Update progress bar and label."""
        progress = (current / total) * 100 if total > 0 else 0
        self.progress_var.set(progress)
        self.step_label.config(text=f"Step: {current} / {total}")
        self.status_var.set(f"Running simulation... Step {current}/{total}")
    
    def _update_visualization(self) -> None:
        """Update simple visualization."""
        if HAS_MATPLOTLIB:
            self._update_matplotlib_viz()
        else:
            self._update_text_viz()
    
    def _update_matplotlib_viz(self) -> None:
        """Update the 2-plot visualization."""
        baseline_df = self.data_collector.get_aggregated_by_step("baseline")
        optimized_df = self.data_collector.get_aggregated_by_step("optimized")
        
        self.ax_wallet.clear()
        self.ax_grid.clear()
        
        if baseline_df.empty and optimized_df.empty:
            self.canvas.draw()
            return
        
        # Colors
        baseline_color = "#e74c3c"  # Red
        optimized_color = "#27ae60"  # Green
        
        # Plot 1: Wallet
        self.ax_wallet.set_title("💰 Total Wallet", fontsize=10, fontweight='bold')
        if not baseline_df.empty:
            self.ax_wallet.plot(baseline_df["step"], baseline_df["wallet"], 
                               color=baseline_color, label="Baseline", linewidth=1.5)
        if not optimized_df.empty:
            self.ax_wallet.plot(optimized_df["step"], optimized_df["wallet"],
                               color=optimized_color, label="Optimized", linewidth=1.5)
        self.ax_wallet.set_xlabel("Step")
        self.ax_wallet.set_ylabel("Wallet ($)")
        self.ax_wallet.legend(fontsize=8)
        self.ax_wallet.grid(True, alpha=0.3)
        
        # Plot 2: Grid Buy
        self.ax_grid.set_title("🔌 Grid Energy Purchase", fontsize=10, fontweight='bold')
        if not baseline_df.empty:
            self.ax_grid.plot(baseline_df["step"], baseline_df["grid_buy"],
                             color=baseline_color, label="Baseline", linewidth=1.5)
        if not optimized_df.empty:
            self.ax_grid.plot(optimized_df["step"], optimized_df["grid_buy"],
                             color=optimized_color, label="Optimized", linewidth=1.5)
        self.ax_grid.set_xlabel("Step")
        self.ax_grid.set_ylabel("Grid Buy (kWh)")
        self.ax_grid.legend(fontsize=8)
        self.ax_grid.grid(True, alpha=0.3)
        
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()
    
    def _update_text_viz(self) -> None:
        """Update text-based visualization."""
        self.text_viz.config(state=tk.NORMAL)
        self.text_viz.delete(1.0, tk.END)
        
        baseline_df = self.data_collector.get_aggregated_by_step("baseline")
        optimized_df = self.data_collector.get_aggregated_by_step("optimized")
        
        if not baseline_df.empty:
            latest_b = baseline_df.iloc[-1]
            self.text_viz.insert(tk.END, f"BASELINE - Step {int(latest_b['step'])}:\n")
            self.text_viz.insert(tk.END, f"  Wallet: ${latest_b['wallet']:.2f}\n")
            self.text_viz.insert(tk.END, f"  Grid Buy: {latest_b['grid_buy']:.2f} kWh\n\n")
        
        if not optimized_df.empty:
            latest_o = optimized_df.iloc[-1]
            self.text_viz.insert(tk.END, f"OPTIMIZED - Step {int(latest_o['step'])}:\n")
            self.text_viz.insert(tk.END, f"  Wallet: ${latest_o['wallet']:.2f}\n")
            self.text_viz.insert(tk.END, f"  Grid Buy: {latest_o['grid_buy']:.2f} kWh\n")
        
        self.text_viz.config(state=tk.DISABLED)
    
    def _update_summary(self) -> None:
        """Update the summary text."""
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        
        baseline_df = self.data_collector.get_aggregated_by_step("baseline")
        optimized_df = self.data_collector.get_aggregated_by_step("optimized")
        
        if baseline_df.empty and optimized_df.empty:
            self.summary_text.insert(tk.END, "No simulation data available.")
            self.summary_text.config(state=tk.DISABLED)
            return
        
        summary = ""
        
        if not baseline_df.empty:
            final_b = baseline_df.iloc[-1]
            summary += "📊 BASELINE RESULTS:\n"
            summary += "─" * 35 + "\n"
            summary += f"  Final Wallet:     ${final_b['wallet']:.2f}\n"
            summary += f"  Total Grid Buy:   {baseline_df['grid_buy'].sum():.2f} kWh\n"
            summary += f"  Total Grid Sell:  {baseline_df['grid_sell'].sum():.2f} kWh\n\n"
        
        if not optimized_df.empty:
            final_o = optimized_df.iloc[-1]
            summary += "🚀 OPTIMIZED RESULTS:\n"
            summary += "─" * 35 + "\n"
            summary += f"  Final Wallet:     ${final_o['wallet']:.2f}\n"
            summary += f"  Total Grid Buy:   {optimized_df['grid_buy'].sum():.2f} kWh\n"
            summary += f"  Total Grid Sell:  {optimized_df['grid_sell'].sum():.2f} kWh\n"
            if 'p2p_buy_amount' in optimized_df.columns:
                summary += f"  Total P2P Trade:  {optimized_df['p2p_buy_amount'].sum():.2f} kWh\n"
            summary += "\n"
        
        if not baseline_df.empty and not optimized_df.empty:
            final_b = baseline_df.iloc[-1]
            final_o = optimized_df.iloc[-1]
            wallet_diff = final_o['wallet'] - final_b['wallet']
            
            baseline_grid_buy = baseline_df['grid_buy'].sum()
            optimized_grid_buy = optimized_df['grid_buy'].sum()
            grid_buy_reduction = baseline_grid_buy - optimized_grid_buy
            grid_buy_pct = (grid_buy_reduction / baseline_grid_buy * 100) if baseline_grid_buy > 0 else 0
            
            summary += "📈 COMPARISON:\n"
            summary += "─" * 35 + "\n"
            summary += f"  Wallet Improvement: ${wallet_diff:+.2f}\n"
            summary += f"  Grid Buy Reduction: {grid_buy_reduction:.2f} kWh ({grid_buy_pct:.1f}%)\n"
        
        if self.last_save_path:
            summary += "\n" + "─" * 35 + "\n"
            summary += f"📁 Results saved to:\n   {self.last_save_path}\n"
        
        self.summary_text.insert(tk.END, summary)
        self.summary_text.config(state=tk.DISABLED)
    
    def _auto_save_results(self) -> None:
        """Auto-save simulation results with timestamp."""
        try:
            output_dir = Path(self.output_dir_var.get())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = output_dir / f"sim_{timestamp}"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            baseline_path = save_dir / "baseline_results.csv"
            optimized_path = save_dir / "optimized_results.csv"
            params_path = save_dir / "params.json"
            grid_capacity_path = save_dir / "grid_capacity.csv"
            
            self.data_collector.export_to_csv("baseline", str(baseline_path))
            self.data_collector.export_to_csv("optimized", str(optimized_path))
            self.params.save(str(params_path))
            
            # Save grid capacity data if available
            if self.grid_capacity is not None:
                self.grid_capacity.save(str(grid_capacity_path))
            
            self.last_save_path = save_dir
            
        except Exception as e:
            print(f"Failed to auto-save results: {e}")
    
    def _open_results_folder(self) -> None:
        """Open the results folder in file explorer."""
        import subprocess
        import sys
        
        if self.last_save_path and self.last_save_path.exists():
            folder = self.last_save_path
        else:
            folder = Path(self.output_dir_var.get())
            folder.mkdir(parents=True, exist_ok=True)
        
        if sys.platform == 'win32':
            subprocess.run(['explorer', str(folder)])
        elif sys.platform == 'darwin':
            subprocess.run(['open', str(folder)])
        else:
            subprocess.run(['xdg-open', str(folder)])
    
    def _launch_visualizer(self) -> None:
        """Launch the visualizer application."""
        import subprocess
        import sys
        
        try:
            subprocess.Popen([sys.executable, 'visualizer.py'])
            self.status_var.set("Visualizer launched")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch visualizer: {e}")
    
    # --- Observer interface ---
    
    def on_step_complete(self, sim_type: str, step: int, data: list[StepData]) -> None:
        """Called when a simulation step completes (from SimulationObserver protocol)."""
        pass  # Updates handled in main thread via after()
    
    def on_simulation_complete(self, sim_type: str) -> None:
        """Called when simulation completes (from SimulationObserver protocol)."""
        pass


def main():
    """Entry point for GUI application."""
    app = SimulationGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
