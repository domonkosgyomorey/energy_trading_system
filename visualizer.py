"""
Simulation Visualizer Application.
Plays back simulation results from CSV files with speed control, confidence intervals,
circular network layout, and detailed tooltips including P2P trading information.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import math
from pathlib import Path
from dataclasses import dataclass

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class PlotConfig:
    """Configuration for a single plot."""
    name: str
    title: str
    ylabel: str
    columns: list[str]
    colors: list[str]
    labels: list[str]
    show_ci: bool = True  # Show confidence intervals
    enabled: bool = True


class SimulationPlayer:
    """Handles playback of simulation data with statistics."""
    
    def __init__(self, baseline_df: pd.DataFrame, optimized_df: pd.DataFrame):
        self.baseline_df = baseline_df
        self.optimized_df = optimized_df
        
        # Precompute aggregated data
        self.baseline_agg = self._aggregate(baseline_df) if not baseline_df.empty else pd.DataFrame()
        self.optimized_agg = self._aggregate(optimized_df) if not optimized_df.empty else pd.DataFrame()
        self.baseline_stats = self._compute_stats(baseline_df) if not baseline_df.empty else pd.DataFrame()
        self.optimized_stats = self._compute_stats(optimized_df) if not optimized_df.empty else pd.DataFrame()
        
        self.max_step = 0
        if not self.baseline_agg.empty:
            self.max_step = max(self.max_step, int(self.baseline_agg["step"].max()))
        if not self.optimized_agg.empty:
            self.max_step = max(self.max_step, int(self.optimized_agg["step"].max()))
    
    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by step (sum across households)."""
        if df.empty:
            return df
        
        agg_cols = {}
        for col in ["production", "consumption", "stored_kwh", "wallet", 
                   "grid_buy", "grid_sell", "p2p_buy_amount", "p2p_sell_amount"]:
            if col in df.columns:
                agg_cols[col] = "sum"
        
        agg_cols["household_id"] = "count"
        
        result = df.groupby("step").agg(agg_cols).reset_index()
        result = result.rename(columns={"household_id": "household_count"})
        return result
    
    def _compute_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute mean and std by step for confidence intervals."""
        if df.empty:
            return df
        
        numeric_cols = ["production", "consumption", "stored_kwh", "wallet", 
                       "grid_buy", "grid_sell", "battery_pct", "p2p_buy_amount", "p2p_sell_amount"]
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        
        agg_dict = {col: ["mean", "std", "sum"] for col in numeric_cols}
        result = df.groupby("step").agg(agg_dict)
        result.columns = ['_'.join(col) for col in result.columns]
        return result.reset_index()
    
    def get_data_up_to_step(self, step: int, sim_type: str = "optimized"):
        """Get aggregated data and stats up to given step."""
        if sim_type == "baseline":
            agg = self.baseline_agg[self.baseline_agg["step"] <= step] if not self.baseline_agg.empty else pd.DataFrame()
            stats = self.baseline_stats[self.baseline_stats["step"] <= step] if not self.baseline_stats.empty else pd.DataFrame()
        else:
            agg = self.optimized_agg[self.optimized_agg["step"] <= step] if not self.optimized_agg.empty else pd.DataFrame()
            stats = self.optimized_stats[self.optimized_stats["step"] <= step] if not self.optimized_stats.empty else pd.DataFrame()
        return agg, stats
    
    def get_step_data(self, step: int, sim_type: str = "optimized") -> pd.DataFrame:
        """Get raw data for a specific step."""
        df = self.baseline_df if sim_type == "baseline" else self.optimized_df
        if df.empty:
            return pd.DataFrame()
        return df[df["step"] == step].copy()


class CIPlotPanel(ttk.Frame):
    """Plot panel with confidence interval support."""
    
    def __init__(self, parent, config: PlotConfig):
        super().__init__(parent)
        self.plot_config = config
        
        self.fig = Figure(figsize=(5, 3), dpi=85)
        self.fig.tight_layout()
        self.ax = self.fig.add_subplot(111)
        
        self._setup_plot()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _setup_plot(self):
        """Initialize plot appearance."""
        self.ax.set_title(self.plot_config.title, fontsize=10, fontweight='bold')
        self.ax.set_xlabel("Step", fontsize=9)
        self.ax.set_ylabel(self.plot_config.ylabel, fontsize=9)
        self.ax.tick_params(labelsize=8)
        self.ax.grid(True, alpha=0.3)
    
    def update_data(self, baseline_agg: pd.DataFrame, optimized_agg: pd.DataFrame,
                   baseline_stats: pd.DataFrame, optimized_stats: pd.DataFrame):
        """Update plot with new data including confidence intervals."""
        if not self.plot_config.enabled:
            return
        
        self.ax.clear()
        self._setup_plot()
        
        baseline_color = "#e74c3c"
        optimized_color = "#27ae60"
        
        for col, color, label in zip(self.plot_config.columns, self.plot_config.colors, self.plot_config.labels):
            # Plot baseline with CI
            if not baseline_stats.empty:
                mean_col = f"{col}_mean"
                std_col = f"{col}_std"
                sum_col = f"{col}_sum"
                
                # Use sum for totals, mean for per-household
                if sum_col in baseline_stats.columns:
                    y_data = baseline_stats[sum_col]
                elif mean_col in baseline_stats.columns:
                    y_data = baseline_stats[mean_col]
                else:
                    continue
                
                x_data = baseline_stats["step"]
                self.ax.plot(x_data, y_data, color=baseline_color, linewidth=1.5,
                            linestyle='--', alpha=0.8, label=f"{label} (Baseline)")
                
                # Confidence interval
                if self.plot_config.show_ci and std_col in baseline_stats.columns:
                    std_data = baseline_stats[std_col].fillna(0)
                    # Scale std by household count for sum metrics
                    if sum_col in baseline_stats.columns and mean_col in baseline_stats.columns:
                        mean_data = baseline_stats[mean_col]
                        n_households = (y_data / mean_data).fillna(1).replace([float('inf'), -float('inf')], 1)
                        ci_range = std_data * n_households.apply(lambda x: math.sqrt(max(1, x)))
                    else:
                        ci_range = std_data
                    
                    self.ax.fill_between(x_data, y_data - ci_range, y_data + ci_range,
                                        color=baseline_color, alpha=0.15)
            
            # Plot optimized with CI
            if not optimized_stats.empty:
                mean_col = f"{col}_mean"
                std_col = f"{col}_std"
                sum_col = f"{col}_sum"
                
                if sum_col in optimized_stats.columns:
                    y_data = optimized_stats[sum_col]
                elif mean_col in optimized_stats.columns:
                    y_data = optimized_stats[mean_col]
                else:
                    continue
                
                x_data = optimized_stats["step"]
                self.ax.plot(x_data, y_data, color=optimized_color, linewidth=2,
                            label=f"{label} (Optimized)")
                
                if self.plot_config.show_ci and std_col in optimized_stats.columns:
                    std_data = optimized_stats[std_col].fillna(0)
                    if sum_col in optimized_stats.columns and mean_col in optimized_stats.columns:
                        mean_data = optimized_stats[mean_col]
                        n_households = (y_data / mean_data).fillna(1).replace([float('inf'), -float('inf')], 1)
                        ci_range = std_data * n_households.apply(lambda x: math.sqrt(max(1, x)))
                    else:
                        ci_range = std_data
                    
                    self.ax.fill_between(x_data, y_data - ci_range, y_data + ci_range,
                                        color=optimized_color, alpha=0.15)
        
        self.ax.legend(fontsize=7, loc='best')
        self.canvas.draw_idle()
    
    def set_enabled(self, enabled: bool):
        self.plot_config.enabled = enabled


class CircularNetworkPanel(ttk.Frame):
    """Network visualization with circular household layout and P2P info."""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # View selector
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill=tk.X, pady=2)
        
        self.view_var = tk.StringVar(value="optimized")
        ttk.Radiobutton(ctrl_frame, text="Baseline", variable=self.view_var,
                       value="baseline", command=self._on_view_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(ctrl_frame, text="Optimized", variable=self.view_var,
                       value="optimized", command=self._on_view_change).pack(side=tk.LEFT, padx=5)
        
        self.canvas = tk.Canvas(self, bg="#f8f9fa", highlightthickness=1, 
                               highlightbackground="#dee2e6")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Leave>", self._on_mouse_leave)
        
        self.nodes = {}
        self.current_data = None
        self.all_step_data = None
        self.tooltip_window = None
        self.last_draw_size = (0, 0)
        self._view_change_callback = None
    
    def set_view_change_callback(self, callback):
        """Set callback for view changes."""
        self._view_change_callback = callback
    
    def _on_view_change(self):
        if self._view_change_callback:
            self._view_change_callback()
    
    def _on_resize(self, event):
        new_size = (event.width, event.height)
        if abs(new_size[0] - self.last_draw_size[0]) > 30 or \
           abs(new_size[1] - self.last_draw_size[1]) > 30:
            self.last_draw_size = new_size
            if self.current_data is not None:
                self.update_network(self.current_data, self.all_step_data)
    
    def _on_mouse_move(self, event):
        x, y = event.x, event.y
        
        for node_id, info in self.nodes.items():
            px, py = info["pos"]
            radius = info.get("radius", 30)
            if ((x - px) ** 2 + (y - py) ** 2) ** 0.5 < radius + 10:
                self._show_tooltip(event, node_id, info)
                return
        
        self._hide_tooltip()
    
    def _on_mouse_leave(self, event):
        self._hide_tooltip()
    
    def _show_tooltip(self, event, node_id, info):
        """Show detailed tooltip with P2P info."""
        self._hide_tooltip()
        
        data = info.get("data", {})
        node_type = info.get("type", "")
        
        # Create tooltip window
        self.tooltip_window = tk.Toplevel(self)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_attributes("-topmost", True)
        
        frame = ttk.Frame(self.tooltip_window, padding=8)
        frame.pack()
        
        if node_type == "household":
            title = f"🏠 Household"
            ttk.Label(frame, text=title, font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W)
            ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
            
            # Financial
            wallet = data.get('wallet', 0)
            wallet_color = "green" if wallet >= 0 else "red"
            ttk.Label(frame, text=f"💰 Wallet: ${wallet:.2f}", 
                     foreground=wallet_color).pack(anchor=tk.W)
            
            # Battery
            battery_pct = data.get('battery_pct', 0)
            stored = data.get('stored_kwh', 0)
            ttk.Label(frame, text=f"🔋 Battery: {battery_pct:.1f}% ({stored:.2f} kWh)").pack(anchor=tk.W)
            
            # Energy
            prod = data.get('production', 0)
            cons = data.get('consumption', 0)
            ttk.Label(frame, text=f"⚡ Production: {prod:.2f} kWh").pack(anchor=tk.W)
            ttk.Label(frame, text=f"🏠 Consumption: {cons:.2f} kWh").pack(anchor=tk.W)
            
            # Grid trading
            grid_buy = data.get('grid_buy', 0)
            grid_sell = data.get('grid_sell', 0)
            if grid_buy > 0:
                ttk.Label(frame, text=f"🔌 Grid Buy: {grid_buy:.2f} kWh", 
                         foreground="red").pack(anchor=tk.W)
            if grid_sell > 0:
                ttk.Label(frame, text=f"⚡ Grid Sell: {grid_sell:.2f} kWh",
                         foreground="green").pack(anchor=tk.W)
            
            # P2P Trading - important info!
            p2p_buy = data.get('p2p_buy_amount', 0)
            p2p_sell = data.get('p2p_sell_amount', 0)
            if p2p_buy > 0 or p2p_sell > 0:
                ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
                ttk.Label(frame, text="🤝 P2P Trading:", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W)
                if p2p_buy > 0:
                    ttk.Label(frame, text=f"   Bought: {p2p_buy:.2f} kWh",
                             foreground="#3498db").pack(anchor=tk.W)
                if p2p_sell > 0:
                    ttk.Label(frame, text=f"   Sold: {p2p_sell:.2f} kWh",
                             foreground="#9b59b6").pack(anchor=tk.W)
            
            # P2P trade details if available
            p2p_trades = data.get('p2p_trades', '')
            # Handle case where p2p_trades might be NaN (float) or not a string
            if isinstance(p2p_trades, str) and p2p_trades and p2p_trades != '[]':
                display_trades = p2p_trades[:50] + "..." if len(p2p_trades) > 50 else p2p_trades
                ttk.Label(frame, text=f"   Details: {display_trades}",
                         font=("TkDefaultFont", 7)).pack(anchor=tk.W)
        
        elif node_type == "city":
            ttk.Label(frame, text="🏙️ City Grid", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W)
            ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
            ttk.Label(frame, text=f"Total Buy: {data.get('total_buy', 0):.2f} kWh",
                     foreground="red").pack(anchor=tk.W)
            ttk.Label(frame, text=f"Total Sell: {data.get('total_sell', 0):.2f} kWh",
                     foreground="green").pack(anchor=tk.W)
            
            # P2P summary
            ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
            ttk.Label(frame, text=f"🤝 Total P2P Volume: {data.get('total_p2p', 0):.2f} kWh",
                     foreground="#9b59b6").pack(anchor=tk.W)
        
        # Position tooltip
        x = self.winfo_rootx() + event.x + 15
        y = self.winfo_rooty() + event.y + 10
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
    
    def _hide_tooltip(self):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
    
    def update_network(self, step_data: pd.DataFrame, all_data: pd.DataFrame = None):
        """Update network visualization with circular layout."""
        self.current_data = step_data
        self.all_step_data = all_data
        self.canvas.delete("all")
        self.nodes.clear()
        
        if step_data.empty:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="No data for this step",
                font=("TkDefaultFont", 12),
                fill="#6c757d"
            )
            return
        
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        if w < 100 or h < 100:
            return
        
        center_x = w / 2
        center_y = h / 2 + 20
        
        households = step_data["household_id"].unique()
        n = len(households)
        
        # City at center-top
        city_pos = (center_x, 50)
        total_buy = step_data["grid_buy"].sum()
        total_sell = step_data["grid_sell"].sum() if "grid_sell" in step_data.columns else 0
        total_p2p = 0
        if "p2p_buy_amount" in step_data.columns:
            total_p2p = step_data["p2p_buy_amount"].sum()
        
        self._draw_city(city_pos, total_buy, total_sell, total_p2p)
        
        # Calculate circular positions for households
        radius = min(w, h) * 0.35
        household_positions = {}
        
        for i, hh_id in enumerate(households):
            # Full circle, starting from top
            angle = 2 * math.pi * i / n - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            household_positions[hh_id] = (x, y)
        
        # Draw P2P connections first (behind everything)
        if "p2p_trades" in step_data.columns or "p2p_buy_amount" in step_data.columns:
            self._draw_p2p_connections(step_data, household_positions)
        
        # Draw grid connections
        for hh_id in households:
            hh_data = step_data[step_data["household_id"] == hh_id].iloc[0]
            pos = household_positions[hh_id]
            
            grid_buy = hh_data.get("grid_buy", 0)
            grid_sell = hh_data.get("grid_sell", 0)
            
            if grid_buy > 0.1:
                self._draw_flow_arrow(city_pos, pos, "#e74c3c", grid_buy, "buy")
            if grid_sell > 0.1:
                self._draw_flow_arrow(pos, city_pos, "#27ae60", grid_sell, "sell")
        
        # Draw households on top
        for hh_id in households:
            hh_data = step_data[step_data["household_id"] == hh_id].iloc[0]
            pos = household_positions[hh_id]
            self._draw_household(pos, hh_id, hh_data)
    
    def _draw_city(self, pos, total_buy, total_sell, total_p2p):
        """Draw city grid node using emoji icon."""
        x, y = pos
        r = 35
        
        # Large emoji as main icon
        self.canvas.create_text(x, y, text="🏙️", font=("Segoe UI Emoji", 36))
        
        # Label below
        self.canvas.create_text(x, y + 35, text="City Grid", 
                               font=("TkDefaultFont", 9, "bold"), fill="#2c3e50")
        
        # Stats below
        self.canvas.create_text(x, y + 50, text=f"⬇️ {total_buy:.1f} kWh",
                               font=("TkDefaultFont", 8), fill="#e74c3c")
        self.canvas.create_text(x, y + 64, text=f"⬆️ {total_sell:.1f} kWh",
                               font=("TkDefaultFont", 8), fill="#27ae60")
        
        self.nodes["city"] = {
            "pos": pos, "type": "city", "radius": r,
            "data": {"total_buy": total_buy, "total_sell": total_sell, "total_p2p": total_p2p}
        }
    
    def _draw_household(self, pos, hh_id, data):
        """Draw household node using emoji icons."""
        x, y = pos
        r = 30
        
        wallet = data.get("wallet", 0)
        battery_pct = data.get("battery_pct", 0)
        p2p_buy = data.get("p2p_buy_amount", 0)
        p2p_sell = data.get("p2p_sell_amount", 0)
        
        # Choose house emoji based on status
        # 🏠 normal house, 🏡 house with garden (producing), 🏚️ needs energy
        if wallet >= 0 and battery_pct > 50:
            house_emoji = "🏡"  # Doing well - green house
        elif wallet < -50 or battery_pct < 10:
            house_emoji = "🏚️"  # Struggling
        else:
            house_emoji = "🏠"  # Normal
        
        # Large emoji as main icon
        self.canvas.create_text(x, y - 5, text=house_emoji, font=("Segoe UI Emoji", 28))
        
        # Battery indicator with emoji
        if battery_pct >= 80:
            battery_emoji = "🔋"  # Full
        elif battery_pct >= 40:
            battery_emoji = "🔋"  # Medium (same emoji, different color text)
        elif battery_pct >= 10:
            battery_emoji = "🪫"  # Low battery
        else:
            battery_emoji = "🪫"  # Very low
        
        # Battery + percentage
        battery_color = "#27ae60" if battery_pct > 30 else "#f39c12" if battery_pct > 10 else "#e74c3c"
        self.canvas.create_text(x, y + 25, text=f"{battery_emoji} {battery_pct:.0f}%",
                               font=("TkDefaultFont", 8), fill=battery_color)
        
        # Wallet with emoji
        wallet_emoji = "💰" if wallet >= 0 else "💸"
        wallet_color = "#27ae60" if wallet >= 0 else "#e74c3c"
        self.canvas.create_text(x, y + 40, text=f"{wallet_emoji} ${wallet:.0f}",
                               font=("TkDefaultFont", 8, "bold"), fill=wallet_color)
        
        # P2P trading indicator
        if p2p_buy > 0 or p2p_sell > 0:
            trade_text = ""
            if p2p_sell > 0:
                trade_text += f"📤{p2p_sell:.1f}"
            if p2p_buy > 0:
                if trade_text:
                    trade_text += " "
                trade_text += f"📥{p2p_buy:.1f}"
            self.canvas.create_text(x, y + 55, text=trade_text,
                                   font=("TkDefaultFont", 7), fill="#9b59b6")
        
        self.nodes[hh_id] = {"pos": pos, "type": "household", "radius": r, "data": data.to_dict()}
    
    def _draw_p2p_connections(self, step_data, positions):
        """Draw P2P trading connections between households with arrows showing direction."""
        # Parse p2p_trades to get actual seller->buyer relationships
        # Format: "seller_id:amount-seller_id:amount-..."
        
        # Collect all trades: (seller_id, buyer_id, amount)
        trades = []
        
        for _, row in step_data.iterrows():
            buyer_id = row["household_id"]
            p2p_trades_str = row.get("p2p_trades", "")
            
            if p2p_trades_str and isinstance(p2p_trades_str, str) and p2p_trades_str.strip():
                # Parse the trades string
                trade_parts = p2p_trades_str.split("-")
                for part in trade_parts:
                    if ":" in part:
                        try:
                            seller_id, amount_str = part.split(":", 1)
                            amount = float(amount_str)
                            if amount > 0.01:  # Only show meaningful trades
                                trades.append((seller_id.strip(), buyer_id, amount))
                        except (ValueError, IndexError):
                            continue
        
        # Draw arrows for each trade (seller -> buyer)
        for seller_id, buyer_id, amount in trades:
            if seller_id in positions and buyer_id in positions:
                self._draw_p2p_trade_arrow(
                    positions[seller_id], 
                    positions[buyer_id], 
                    amount,
                    seller_id,
                    buyer_id
                )
    
    def _draw_p2p_trade_arrow(self, from_pos, to_pos, amount, seller_id, buyer_id):
        """Draw curved P2P trade arrow from seller to buyer."""
        x1, y1 = from_pos
        x2, y2 = to_pos
        
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length < 60:
            return
        
        # Calculate control point for curve (perpendicular offset)
        # Offset direction based on relative position for consistent curve direction
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Perpendicular vector
        perp_x = -dy / length
        perp_y = dx / length
        
        # Curve offset (larger for longer distances)
        curve_offset = min(40, length * 0.15)
        
        # Control point
        cx = mid_x + perp_x * curve_offset
        cy = mid_y + perp_y * curve_offset
        
        # Shorten arrow to not overlap with node circles
        offset = 35
        ratio_start = offset / length
        ratio_end = (length - offset) / length
        
        # Calculate bezier curve points
        points = []
        for t in [i/12 for i in range(13)]:
            # Quadratic bezier with adjusted endpoints
            t_adj = ratio_start + t * (ratio_end - ratio_start)
            px = (1-t_adj)**2 * x1 + 2*(1-t_adj)*t_adj * cx + t_adj**2 * x2
            py = (1-t_adj)**2 * y1 + 2*(1-t_adj)*t_adj * cy + t_adj**2 * y2
            points.extend([px, py])
        
        # Arrow width based on amount
        width = max(1.5, min(4, 1.5 + amount / 5))
        
        # Draw the curved line
        self.canvas.create_line(points, fill="#9b59b6", width=width, smooth=True)
        
        # Draw arrowhead at the end
        # Get the last segment direction
        end_x, end_y = points[-2], points[-1]
        prev_x, prev_y = points[-4], points[-3]
        
        arrow_dx = end_x - prev_x
        arrow_dy = end_y - prev_y
        arrow_len = math.sqrt(arrow_dx*arrow_dx + arrow_dy*arrow_dy)
        
        if arrow_len > 1:
            # Normalize
            arrow_dx /= arrow_len
            arrow_dy /= arrow_len
            
            # Arrowhead size
            head_len = 10
            head_width = 6
            
            # Arrowhead points
            tip_x, tip_y = end_x, end_y
            base_x = tip_x - arrow_dx * head_len
            base_y = tip_y - arrow_dy * head_len
            
            # Perpendicular for arrowhead wings
            left_x = base_x - arrow_dy * head_width
            left_y = base_y + arrow_dx * head_width
            right_x = base_x + arrow_dy * head_width
            right_y = base_y - arrow_dx * head_width
            
            self.canvas.create_polygon(
                tip_x, tip_y, left_x, left_y, right_x, right_y,
                fill="#9b59b6", outline="#7d3c98"
            )
        
        # Draw amount label near the middle of the curve
        label_t = 0.5
        label_x = (1-label_t)**2 * x1 + 2*(1-label_t)*label_t * cx + label_t**2 * x2
        label_y = (1-label_t)**2 * y1 + 2*(1-label_t)*label_t * cy + label_t**2 * y2
        
        # Offset label slightly from the curve
        label_x += perp_x * 12
        label_y += perp_y * 12
        
        self.canvas.create_text(
            label_x, label_y, 
            text=f"{amount:.2f}",
            font=("TkDefaultFont", 7, "bold"),
            fill="#7d3c98"
        )
    
    def _draw_flow_arrow(self, from_pos, to_pos, color, amount, flow_type):
        """Draw energy flow arrow."""
        x1, y1 = from_pos
        x2, y2 = to_pos
        
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length < 80:
            return
        
        # Shorten arrow
        offset = 45
        ratio_start = offset / length
        ratio_end = (length - offset) / length
        
        x1_new = x1 + dx * ratio_start
        y1_new = y1 + dy * ratio_start
        x2_new = x1 + dx * ratio_end
        y2_new = y1 + dy * ratio_end
        
        # Width based on amount
        width = max(1, min(4, amount / 10))
        
        self.canvas.create_line(x1_new, y1_new, x2_new, y2_new,
                               fill=color, width=width, arrow=tk.LAST,
                               arrowshape=(8, 10, 4))


class VisualizerApp(tk.Tk):
    """Main visualizer application."""
    
    def __init__(self):
        super().__init__()
        
        self.title("Energy Trading Simulation Visualizer")
        self.geometry("1500x900")
        self.minsize(1200, 700)
        
        self.player: SimulationPlayer | None = None
        self.plot_panels: dict[str, CIPlotPanel] = {}
        self.is_playing = False
        self.play_speed = 1.0
        self.update_interval = 150
        
        self._create_widgets()
        self._create_plot_configs()
    
    def _create_widgets(self):
        """Create all widgets."""
        # Top control bar
        ctrl_frame = ttk.Frame(self, padding=5)
        ctrl_frame.pack(fill=tk.X)
        
        # File loading
        ttk.Button(ctrl_frame, text="📂 Load Simulation Folder", 
                  command=self._load_simulation_folder).pack(side=tk.LEFT, padx=3)
        
        ttk.Separator(ctrl_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        
        ttk.Button(ctrl_frame, text="Load Baseline", 
                  command=lambda: self._load_file("baseline")).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="Load Optimized", 
                  command=lambda: self._load_file("optimized")).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(ctrl_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        
        # Playback controls
        self.play_btn = ttk.Button(ctrl_frame, text="▶ Play", command=self._toggle_play, width=8)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(ctrl_frame, text="⏮", command=self._reset, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(ctrl_frame, text="◀", command=self._step_back, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(ctrl_frame, text="▶", command=self._step_forward, width=3).pack(side=tk.LEFT, padx=1)
        
        # Speed control
        ttk.Label(ctrl_frame, text="Speed:").pack(side=tk.LEFT, padx=(15, 3))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_combo = ttk.Combobox(ctrl_frame, textvariable=self.speed_var, 
                                   values=[0.25, 0.5, 1.0, 2.0, 5.0, 10.0], width=5)
        speed_combo.pack(side=tk.LEFT)
        speed_combo.bind("<<ComboboxSelected>>", self._on_speed_change)
        
        # Step slider
        ttk.Label(ctrl_frame, text="Step:").pack(side=tk.LEFT, padx=(20, 3))
        self.step_var = tk.IntVar(value=0)
        self.step_slider = ttk.Scale(ctrl_frame, from_=0, to=100, variable=self.step_var,
                                     orient=tk.HORIZONTAL, length=250, command=self._on_step_change)
        self.step_slider.pack(side=tk.LEFT, padx=3)
        self.step_label = ttk.Label(ctrl_frame, text="0 / 0", width=12)
        self.step_label.pack(side=tk.LEFT, padx=5)
        
        # Main notebook for tabs
        self.main_notebook = ttk.Notebook(self)
        self.main_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # === Tab 1: Playback View ===
        playback_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(playback_frame, text="📺 Playback")
        
        # Main content - horizontal paned window
        main_paned = ttk.PanedWindow(playback_frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Left: Plot toggles
        left_frame = ttk.LabelFrame(main_paned, text="📊 Plots", padding=5)
        main_paned.add(left_frame, weight=0)
        
        self.plot_vars: dict[str, tk.BooleanVar] = {}
        self.toggle_frame = ttk.Frame(left_frame)
        self.toggle_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add CI toggle
        self.show_ci_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.toggle_frame, text="📊 Show Confidence Intervals",
                       variable=self.show_ci_var, command=self._toggle_ci).pack(anchor=tk.W, pady=5)
        ttk.Separator(self.toggle_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Center: Plots
        center_frame = ttk.Frame(main_paned)
        main_paned.add(center_frame, weight=3)
        
        self.plots_container = ttk.Frame(center_frame)
        self.plots_container.pack(fill=tk.BOTH, expand=True)
        
        # Right: Network view
        right_frame = ttk.LabelFrame(main_paned, text="🔗 Network View", padding=5)
        main_paned.add(right_frame, weight=2)
        self.network_panel = CircularNetworkPanel(right_frame)
        self.network_panel.pack(fill=tk.BOTH, expand=True)
        self.network_panel.set_view_change_callback(self._update_display)
        
        # === Tab 2: Analysis & Comparison ===
        analysis_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(analysis_frame, text="📊 Analysis")
        self._create_analysis_tab(analysis_frame)
        
        # === Tab 3: Per-Household Metrics ===
        household_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(household_frame, text="🏠 Households")
        self._create_household_tab(household_frame)
        
        # Status bar
        self.status_var = tk.StringVar(value="Load simulation files to begin")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, padding=3)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def _create_analysis_tab(self, parent: ttk.Frame) -> None:
        """Create the analysis/comparison tab with bar charts and metrics."""
        # Two-column layout
        left_frame = ttk.Frame(parent)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        right_frame = ttk.Frame(parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left: Bar chart comparison
        if HAS_MATPLOTLIB:
            chart_frame = ttk.LabelFrame(left_frame, text="📊 Baseline vs Optimized Comparison", padding=5)
            chart_frame.pack(fill=tk.BOTH, expand=True)
            
            self.comparison_fig = Figure(figsize=(8, 6), dpi=90)
            self.comparison_canvas = FigureCanvasTkAgg(self.comparison_fig, master=chart_frame)
            self.comparison_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Right: Key metrics and summary
        metrics_frame = ttk.LabelFrame(right_frame, text="📈 Key Performance Indicators", padding=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        self.metrics_text = tk.Text(metrics_frame, height=30, state=tk.DISABLED, 
                                    font=("Consolas", 10), wrap=tk.WORD)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        
        # Refresh button
        ttk.Button(right_frame, text="🔄 Refresh Analysis", 
                  command=self._update_analysis_tab).pack(pady=10)
    
    def _create_household_tab(self, parent: ttk.Frame) -> None:
        """Create the per-household comparison tab."""
        # Top: Charts showing per-household data
        charts_frame = ttk.Frame(parent)
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        if HAS_MATPLOTLIB:
            # Create figure with subplots
            self.household_fig = Figure(figsize=(14, 8), dpi=90)
            self.household_canvas = FigureCanvasTkAgg(self.household_fig, master=charts_frame)
            self.household_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Bottom: Refresh button and info
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(bottom_frame, text="🔄 Refresh Household Data", 
                  command=self._update_household_tab).pack(side=tk.LEFT, padx=5)
        
        self.household_info_label = ttk.Label(bottom_frame, text="Load simulation data to see per-household analysis")
        self.household_info_label.pack(side=tk.LEFT, padx=10)
    
    def _update_analysis_tab(self) -> None:
        """Update the analysis tab with comparison charts and metrics."""
        if self.player is None:
            return
        
        # Update bar chart
        if HAS_MATPLOTLIB:
            self.comparison_fig.clear()
            
            # Create subplots
            ax1 = self.comparison_fig.add_subplot(2, 2, 1)
            ax2 = self.comparison_fig.add_subplot(2, 2, 2)
            ax3 = self.comparison_fig.add_subplot(2, 2, 3)
            ax4 = self.comparison_fig.add_subplot(2, 2, 4)
            
            baseline_df = self.player.baseline_df
            optimized_df = self.player.optimized_df
            
            if not baseline_df.empty and not optimized_df.empty:
                # Get final values
                b_final = baseline_df.groupby('household_id').last()
                o_final = optimized_df.groupby('household_id').last()
                
                # 1. Final wallet comparison
                x = range(len(b_final))
                width = 0.35
                ax1.bar([i - width/2 for i in x], b_final['wallet'].values, width, label='Baseline', color='#e74c3c', alpha=0.8)
                ax1.bar([i + width/2 for i in x], o_final['wallet'].values, width, label='Optimized', color='#27ae60', alpha=0.8)
                ax1.set_xlabel('Household')
                ax1.set_ylabel('Final Wallet ($)')
                ax1.set_title('💰 Final Wallet by Household')
                ax1.legend()
                ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                
                # 2. Total Grid Buy
                b_grid = baseline_df.groupby('household_id')['grid_buy'].sum()
                o_grid = optimized_df.groupby('household_id')['grid_buy'].sum()
                ax2.bar([i - width/2 for i in x], b_grid.values, width, label='Baseline', color='#e74c3c', alpha=0.8)
                ax2.bar([i + width/2 for i in x], o_grid.values, width, label='Optimized', color='#27ae60', alpha=0.8)
                ax2.set_xlabel('Household')
                ax2.set_ylabel('Total Grid Buy (kWh)')
                ax2.set_title('🔌 Grid Purchases by Household')
                ax2.legend()
                
                # 3. Savings per household (bar chart)
                savings = o_final['wallet'].values - b_final['wallet'].values
                colors = ['#27ae60' if s >= 0 else '#e74c3c' for s in savings]
                ax3.bar(x, savings, color=colors, alpha=0.8)
                ax3.set_xlabel('Household')
                ax3.set_ylabel('Savings ($)')
                ax3.set_title('📈 Savings per Household (Optimized - Baseline)')
                ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax3.axhline(y=savings.mean(), color='blue', linestyle='--', linewidth=1, label=f'Mean: ${savings.mean():.2f}')
                ax3.legend()
                
                # 4. P2P Trading Volume (if available)
                if 'p2p_buy_amount' in optimized_df.columns:
                    p2p_buy = optimized_df.groupby('household_id')['p2p_buy_amount'].sum()
                    p2p_sell = optimized_df.groupby('household_id')['p2p_sell_amount'].sum() if 'p2p_sell_amount' in optimized_df.columns else p2p_buy * 0
                    ax4.bar([i - width/2 for i in x], p2p_buy.values, width, label='P2P Buy', color='#3498db', alpha=0.8)
                    ax4.bar([i + width/2 for i in x], p2p_sell.values, width, label='P2P Sell', color='#9b59b6', alpha=0.8)
                    ax4.set_xlabel('Household')
                    ax4.set_ylabel('Energy (kWh)')
                    ax4.set_title('🤝 P2P Trading Volume by Household')
                    ax4.legend()
                else:
                    ax4.text(0.5, 0.5, 'No P2P data available', ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('🤝 P2P Trading Volume')
            
            self.comparison_fig.tight_layout()
            self.comparison_canvas.draw()
        
        # Update metrics text
        self._update_metrics_text()
    
    def _update_metrics_text(self) -> None:
        """Update the key metrics text."""
        if self.player is None:
            return
        
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        
        baseline_df = self.player.baseline_df
        optimized_df = self.player.optimized_df
        
        if baseline_df.empty or optimized_df.empty:
            self.metrics_text.insert(tk.END, "Load both baseline and optimized data for comparison.")
            self.metrics_text.config(state=tk.DISABLED)
            return
        
        text = ""
        
        # Community totals
        b_wallet = baseline_df.groupby('household_id')['wallet'].last().sum()
        o_wallet = optimized_df.groupby('household_id')['wallet'].last().sum()
        b_grid_buy = baseline_df['grid_buy'].sum()
        o_grid_buy = optimized_df['grid_buy'].sum()
        
        text += "═" * 45 + "\n"
        text += "           COMMUNITY PERFORMANCE SUMMARY\n"
        text += "═" * 45 + "\n\n"
        
        text += "💰 FINANCIAL METRICS\n"
        text += "─" * 45 + "\n"
        text += f"  Baseline Total Wallet:    ${b_wallet:>12.2f}\n"
        text += f"  Optimized Total Wallet:   ${o_wallet:>12.2f}\n"
        text += f"  Community Savings:        ${o_wallet - b_wallet:>+12.2f}\n"
        improvement_pct = ((o_wallet - b_wallet) / abs(b_wallet) * 100) if b_wallet != 0 else 0
        text += f"  Improvement:              {improvement_pct:>+12.1f}%\n\n"
        
        text += "⚡ ENERGY METRICS\n"
        text += "─" * 45 + "\n"
        text += f"  Baseline Grid Buy:        {b_grid_buy:>12.2f} kWh\n"
        text += f"  Optimized Grid Buy:       {o_grid_buy:>12.2f} kWh\n"
        grid_reduction = b_grid_buy - o_grid_buy
        grid_pct = (grid_reduction / b_grid_buy * 100) if b_grid_buy > 0 else 0
        text += f"  Grid Buy Reduction:       {grid_reduction:>+12.2f} kWh\n"
        text += f"  Reduction:                {grid_pct:>12.1f}%\n\n"
        
        # P2P metrics
        if 'p2p_buy_amount' in optimized_df.columns:
            p2p_total = optimized_df['p2p_buy_amount'].sum()
            text += "🤝 P2P TRADING\n"
            text += "─" * 45 + "\n"
            text += f"  Total P2P Volume:         {p2p_total:>12.2f} kWh\n"
            p2p_pct = (p2p_total / (p2p_total + o_grid_buy) * 100) if (p2p_total + o_grid_buy) > 0 else 0
            text += f"  P2P vs Total Energy:      {p2p_pct:>12.1f}%\n\n"
        
        # Self-sufficiency
        total_consumption = optimized_df['consumption'].sum()
        total_production = optimized_df['production'].sum()
        self_sufficiency = ((total_consumption - o_grid_buy) / total_consumption * 100) if total_consumption > 0 else 0
        
        text += "🌱 SUSTAINABILITY\n"
        text += "─" * 45 + "\n"
        text += f"  Self-Sufficiency:         {self_sufficiency:>12.1f}%\n"
        text += f"  Total Production:         {total_production:>12.2f} kWh\n"
        text += f"  Total Consumption:        {total_consumption:>12.2f} kWh\n\n"
        
        # Equity analysis
        households = optimized_df['household_id'].unique()
        n_hh = len(households)
        savings_list = []
        for hh_id in households:
            b_hh = baseline_df[baseline_df['household_id'] == hh_id]['wallet'].iloc[-1] if len(baseline_df[baseline_df['household_id'] == hh_id]) > 0 else 0
            o_hh = optimized_df[optimized_df['household_id'] == hh_id]['wallet'].iloc[-1] if len(optimized_df[optimized_df['household_id'] == hh_id]) > 0 else 0
            savings_list.append(o_hh - b_hh)
        
        if savings_list:
            text += "📊 EQUITY ANALYSIS\n"
            text += "─" * 45 + "\n"
            text += f"  Number of Households:     {n_hh:>12}\n"
            text += f"  Mean Savings:             ${sum(savings_list)/len(savings_list):>+11.2f}\n"
            text += f"  Best Performer:           ${max(savings_list):>+11.2f}\n"
            text += f"  Worst Performer:          ${min(savings_list):>+11.2f}\n"
            text += f"  Savings Spread:           ${max(savings_list) - min(savings_list):>11.2f}\n"
            
            # Calculate Gini coefficient for equity
            wallets = sorted([optimized_df[optimized_df['household_id'] == hh_id]['wallet'].iloc[-1] for hh_id in households])
            n = len(wallets)
            if n > 1 and sum(wallets) > 0:
                gini_sum = sum((2*i - n - 1) * wallets[i] for i in range(n))
                mean_wallet = sum(wallets) / n
                gini = abs(gini_sum / (n * n * mean_wallet)) if mean_wallet > 0 else 0
                equity_score = (1 - gini) * 100
                text += f"  Equity Score:             {equity_score:>11.1f}%\n"
                text += f"  (100% = perfect equality)\n"
        
        self.metrics_text.insert(tk.END, text)
        self.metrics_text.config(state=tk.DISABLED)
    
    def _update_household_tab(self) -> None:
        """Update the per-household comparison tab."""
        if self.player is None or not HAS_MATPLOTLIB:
            return
        
        self.household_fig.clear()
        
        baseline_df = self.player.baseline_df
        optimized_df = self.player.optimized_df
        
        if baseline_df.empty or optimized_df.empty:
            ax = self.household_fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Load both baseline and optimized data', ha='center', va='center')
            self.household_canvas.draw()
            return
        
        households = list(optimized_df['household_id'].unique())
        n_hh = len(households)
        
        if n_hh == 0:
            return
        
        # Create subplots - 2 rows: wallet evolution, savings breakdown
        ax1 = self.household_fig.add_subplot(2, 1, 1)
        ax2 = self.household_fig.add_subplot(2, 1, 2)
        
        # Color palette - use predefined colors
        color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Plot 1: Wallet evolution over time (all households)
        for i, hh_id in enumerate(households[:10]):  # Limit to 10 for readability
            hh_data = optimized_df[optimized_df['household_id'] == hh_id].sort_values('step')
            color = color_list[i % len(color_list)]
            ax1.plot(hh_data['step'], hh_data['wallet'], label=f'{str(hh_id)[:8]}', 
                    color=color, linewidth=1.5, alpha=0.8)
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Wallet ($)')
        ax1.set_title('💰 Wallet Evolution per Household (Optimized)')
        ax1.legend(loc='upper left', fontsize=8, ncol=min(5, n_hh))
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot 2: Horizontal bar chart - savings ranking
        savings_data = []
        for hh_id in households:
            b_hh = baseline_df[baseline_df['household_id'] == hh_id]
            o_hh = optimized_df[optimized_df['household_id'] == hh_id]
            
            if not b_hh.empty and not o_hh.empty:
                b_wallet = b_hh['wallet'].iloc[-1]
                o_wallet = o_hh['wallet'].iloc[-1]
                savings_data.append((str(hh_id)[:8], o_wallet - b_wallet, o_wallet))
        
        if savings_data:
            # Sort by savings
            savings_data.sort(key=lambda x: x[1], reverse=True)
            labels = [d[0] for d in savings_data]
            savings = [d[1] for d in savings_data]
            
            colors_bar = ['#27ae60' if s >= 0 else '#e74c3c' for s in savings]
            y_pos = np.arange(len(labels))
            
            ax2.barh(y_pos, savings, color=colors_bar, alpha=0.8)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(labels)
            ax2.set_xlabel('Savings ($)')
            ax2.set_title('📈 Savings Ranking (Optimized - Baseline)')
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add value labels
            for i, (label, val, _) in enumerate(savings_data):
                ax2.text(val, i, f' ${val:+.2f}', va='center', fontsize=8)
        
        self.household_fig.tight_layout()
        self.household_canvas.draw()
        
        self.household_info_label.config(text=f"Showing {n_hh} households. Best saver: ${max(savings) if savings_data else 0:+.2f}")
    
    def _create_plot_configs(self):
        """Create plot configurations with CI support."""
        self.plot_configs = [
            PlotConfig("wallet", "💰 Total Wallet ($)", "Dollars ($)", 
                      ["wallet"], ["#2ecc71"], ["Wallet"], show_ci=True),
            PlotConfig("grid_buy", "🔌 Grid Buy (kWh)", "Energy (kWh)",
                      ["grid_buy"], ["#e74c3c"], ["Buy"], show_ci=True),
            PlotConfig("grid_sell", "⚡ Grid Sell (kWh)", "Energy (kWh)",
                      ["grid_sell"], ["#3498db"], ["Sell"], show_ci=True),
            PlotConfig("battery", "🔋 Battery Storage (kWh)", "Energy (kWh)",
                      ["stored_kwh"], ["#9b59b6"], ["Stored"], show_ci=True),
            PlotConfig("energy", "⚡ Production vs Consumption", "Energy (kWh)",
                      ["production", "consumption"], ["#f39c12", "#e74c3c"], 
                      ["Production", "Consumption"], show_ci=True),
            PlotConfig("p2p", "🤝 P2P Trading (kWh)", "Energy (kWh)",
                      ["p2p_buy_amount", "p2p_sell_amount"], ["#3498db", "#9b59b6"],
                      ["P2P Buy", "P2P Sell"], show_ci=True),
        ]
        
        # Create toggles and panels
        for config in self.plot_configs:
            var = tk.BooleanVar(value=True)
            self.plot_vars[config.name] = var
            
            cb = ttk.Checkbutton(self.toggle_frame, text=config.title, variable=var,
                                command=lambda n=config.name: self._toggle_plot(n))
            cb.pack(anchor=tk.W, pady=2)
            
            if HAS_MATPLOTLIB:
                panel = CIPlotPanel(self.plots_container, config)
                self.plot_panels[config.name] = panel
        
        self._layout_plots()
    
    def _layout_plots(self):
        """Layout visible plots in grid."""
        for panel in self.plot_panels.values():
            panel.grid_forget()
        
        visible = [name for name, var in self.plot_vars.items() if var.get()]
        for i, name in enumerate(visible):
            row, col = divmod(i, 2)
            self.plot_panels[name].grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
        
        n_rows = (len(visible) + 1) // 2
        for i in range(n_rows):
            self.plots_container.rowconfigure(i, weight=1)
        self.plots_container.columnconfigure(0, weight=1)
        self.plots_container.columnconfigure(1, weight=1)
    
    def _toggle_plot(self, name: str):
        enabled = self.plot_vars[name].get()
        if name in self.plot_panels:
            self.plot_panels[name].set_enabled(enabled)
        self._layout_plots()
        self._update_display()
    
    def _toggle_ci(self):
        """Toggle confidence intervals on all plots."""
        show_ci = self.show_ci_var.get()
        for panel in self.plot_panels.values():
            panel.config.show_ci = show_ci
        self._update_display()
    
    def _load_simulation_folder(self):
        """Load simulation from folder."""
        folder = filedialog.askdirectory(title="Select Simulation Results Folder")
        if not folder:
            return
        
        folder_path = Path(folder)
        baseline_path = folder_path / "baseline_results.csv"
        optimized_path = folder_path / "optimized_results.csv"
        
        baseline_df = pd.DataFrame()
        optimized_df = pd.DataFrame()
        loaded = []
        
        try:
            if baseline_path.exists():
                baseline_df = pd.read_csv(baseline_path)
                loaded.append("baseline")
            
            if optimized_path.exists():
                optimized_df = pd.read_csv(optimized_path)
                loaded.append("optimized")
            
            if not loaded:
                messagebox.showwarning("Warning", 
                    "No simulation files found.\nExpected: baseline_results.csv and/or optimized_results.csv")
                return
            
            self.player = SimulationPlayer(baseline_df, optimized_df)
            self.step_slider.config(to=self.player.max_step)
            self.step_var.set(0)
            self.status_var.set(f"Loaded: {folder_path.name} ({', '.join(loaded)}) - {self.player.max_step + 1} steps")
            self._update_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")
    
    def _load_file(self, file_type: str):
        """Load single file."""
        filepath = filedialog.askopenfilename(
            title=f"Select {file_type} results",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filepath:
            return
        
        try:
            df = pd.read_csv(filepath)
            
            if self.player is None:
                self.player = SimulationPlayer(
                    baseline_df=df if file_type == "baseline" else pd.DataFrame(),
                    optimized_df=df if file_type == "optimized" else pd.DataFrame()
                )
            else:
                if file_type == "baseline":
                    self.player = SimulationPlayer(df, self.player.optimized_df)
                else:
                    self.player = SimulationPlayer(self.player.baseline_df, df)
            
            self.step_slider.config(to=self.player.max_step)
            self.status_var.set(f"Loaded {file_type}: {Path(filepath).name}")
            self._update_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")
    
    def _toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_btn.config(text="⏸ Pause" if self.is_playing else "▶ Play")
        if self.is_playing:
            self._play_step()
    
    def _play_step(self):
        if not self.is_playing or self.player is None:
            return
        
        current = self.step_var.get()
        if current >= self.player.max_step:
            self.is_playing = False
            self.play_btn.config(text="▶ Play")
            return
        
        self.step_var.set(current + 1)
        self._update_display()
        
        interval = int(self.update_interval / self.play_speed)
        self.after(interval, self._play_step)
    
    def _reset(self):
        self.step_var.set(0)
        self.is_playing = False
        self.play_btn.config(text="▶ Play")
        self._update_display()
    
    def _step_back(self):
        current = self.step_var.get()
        if current > 0:
            self.step_var.set(current - 1)
            self._update_display()
    
    def _step_forward(self):
        if self.player and self.step_var.get() < self.player.max_step:
            self.step_var.set(self.step_var.get() + 1)
            self._update_display()
    
    def _on_speed_change(self, event=None):
        self.play_speed = self.speed_var.get()
    
    def _on_step_change(self, value):
        if not self.is_playing:
            self._update_display()
    
    def _update_display(self):
        """Update all visualizations."""
        if self.player is None:
            return
        
        step = self.step_var.get()
        self.step_label.config(text=f"{step} / {self.player.max_step}")
        
        # Get data
        baseline_agg, baseline_stats = self.player.get_data_up_to_step(step, "baseline")
        optimized_agg, optimized_stats = self.player.get_data_up_to_step(step, "optimized")
        
        # Update plots
        for panel in self.plot_panels.values():
            if panel.plot_config.enabled:
                panel.update_data(baseline_agg, optimized_agg, baseline_stats, optimized_stats)
        
        # Update network
        view_type = self.network_panel.view_var.get()
        step_data = self.player.get_step_data(step, view_type)
        if step_data.empty:
            other_type = "baseline" if view_type == "optimized" else "optimized"
            step_data = self.player.get_step_data(step, other_type)
        
        all_data = self.player.baseline_df if view_type == "baseline" else self.player.optimized_df
        self.network_panel.update_network(step_data, all_data)
        
        # Update analysis tabs when at final step or first load
        if step == self.player.max_step or step == 0:
            self._update_analysis_tab()
            self._update_household_tab()


def main():
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available")
    
    app = VisualizerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
