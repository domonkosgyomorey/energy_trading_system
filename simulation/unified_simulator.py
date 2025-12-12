"""
Unified simulation scheduler that runs baseline and optimized simulations.
Supports synchronized execution and data collection for visualization.
"""
import pandas as pd
from collections import defaultdict
from typing import Generator
from dataclasses import dataclass

from simulation.params import SimulationParams, DEFAULT_PARAMS
from simulation.data_collector import SimulationDataCollector, StepData
from simulation.grid_capacity_data import GridCapacityData
from simulation.household import Household
from simulation.blockchain import Blockchain
from simulation.forecaster.perfect_forecaster import PerfectForecaster
from simulation.optimizer.convex_optimizer import ConvexOptimizer
from simulation.optimizer.greedy_optimizer import GreedyOptimizer
from simulation.battery.simple_battery import SimpleBattery
from simulation.battery.central_battery import CentralBattery
from simulation.battery.shared_battery import SharedBattery
from simulation.token_to_battery import TokenToBattery
from simulation.local_price_estimator.price_estimator import PriceEstimator
from simulation.city_grid_price_forecaster.simple_city_grid_price_forecaster import SimpleCityGridPriceForecaster
from simulation.utils.logger import logger


@dataclass
class SimulationState:
    """Captures state at each step for comparison."""
    step: int
    households_data: list[StepData]
    total_production: float
    total_consumption: float
    total_wallet: float


class UnifiedSimulator:
    """
    Runs both baseline and optimized simulations with shared configuration.
    Yields control after each step to allow synchronized visualization.
    """
    
    def __init__(self, 
                 household_data: pd.DataFrame,
                 params: SimulationParams | None = None,
                 grid_capacity: GridCapacityData | None = None,
                 data_collector: SimulationDataCollector | None = None):
        
        self.params = params or DEFAULT_PARAMS
        self.household_data = household_data
        self.data_collector = data_collector or SimulationDataCollector()
        
        # Validate that we have enough data for the requested simulation
        self._validate_data_availability()
        
        # Create grid capacity data based on settings
        if not self.params.grid_capacity.use_capacity_limits:
            # No capacity limits - use very large values
            self.grid_capacity = GridCapacityData.create_constant(
                steps=self.params.simulation_steps,
                import_capacity_kw=float('inf'),
                export_capacity_kw=float('inf')
            )
        elif grid_capacity is not None:
            self.grid_capacity = grid_capacity
        elif self.params.paths.grid_capacity_file:
            self.grid_capacity = GridCapacityData.from_file(self.params.paths.grid_capacity_file)
        else:
            # Use default capacity values
            self.grid_capacity = GridCapacityData.create_constant(
                steps=self.params.simulation_steps,
                import_capacity_kw=self.params.grid_capacity.default_import_capacity_kw,
                export_capacity_kw=self.params.grid_capacity.default_export_capacity_kw
            )
        
        # Initialize components for both simulations
        self._init_baseline_components()
        self._init_optimized_components()
    
    def _validate_data_availability(self) -> None:
        """Validate that household data has enough samples for the simulation."""
        if self.household_data.empty:
            raise ValueError("Household data is empty")
        
        # Calculate required samples based on time_step_hours
        samples_per_step = max(1, int(self.params.time_step_hours * 4))  # 4 samples per hour (15min data)
        required_samples = self.params.simulation_steps * samples_per_step
        
        # Check first household to see data length
        first_household = self.household_data.groupby("id").first()
        if len(first_household) == 0:
            raise ValueError("No household data available")
        
        # Count available samples (assume all households have same length)
        sample_household = self.household_data.groupby("id").get_group(first_household.index[0])
        available_samples = len(sample_household)
        
        if available_samples < required_samples:
            max_possible_steps = available_samples // samples_per_step
            logger.warning(
                f"Insufficient data: {available_samples} samples available, "
                f"{required_samples} required for {self.params.simulation_steps} steps "
                f"with {self.params.time_step_hours}h per step. "
                f"Maximum possible steps: {max_possible_steps}. "
                f"Adjusting simulation_steps to {max_possible_steps}."
            )
            self.params.simulation_steps = max_possible_steps
        
    def _prepare_household_data(self) -> dict[str, dict]:
        """Process raw household data into usable format."""
        grouped = self.household_data.groupby("id")
        processed = {}
        
        # Calculate samples per step based on time_step_hours
        # Raw data is 15-minute intervals (96 samples = 24 hours)
        samples_per_step = max(1, int(self.params.time_step_hours * 4))  # 4 samples per hour
        
        for household_id, group in grouped:
            group = group.drop(columns=["id", "timestamp", "category", "season", "period"], errors='ignore')
            raw_dict = {col: group[col].to_list() for col in group.columns}
            
            data_dict = defaultdict(list)
            for key, val in raw_dict.items():
                if key == 'production':
                    val = [x * self.params.household.production_scale_factor for x in val]
                
                # Aggregate data based on time_step_hours parameter
                # For 15min steps: samples_per_step=1, for 1h: 4, for 24h: 96
                if samples_per_step == 1:
                    # No aggregation needed - use raw 15-minute data
                    data_dict[key] = val
                else:
                    # Aggregate into time steps
                    for i in range(0, len(val) - samples_per_step + 1, samples_per_step):
                        step_sum = sum(val[i:i + samples_per_step])
                        data_dict[key].append(step_sum)
            
            processed[str(household_id)] = dict(data_dict)
            
            if len(processed) >= self.params.household.max_households:
                break
        
        return processed
    
    def _create_households(self, central_battery: CentralBattery) -> list[Household]:
        """Create household objects with appropriate batteries."""
        processed_data = self._prepare_household_data()
        households = []
        
        shared_battery_count = int(len(processed_data) * self.params.household.shared_battery_probability)
        
        for i, (hh_id, data) in enumerate(processed_data.items()):
            if i < shared_battery_count:
                battery = SharedBattery(
                    central_battery=central_battery,
                    household_id=hh_id
                )
            else:
                battery = SimpleBattery(
                    capacity_in_kwh=self.params.battery.simple_capacity_kwh,
                    charge_efficiency=self.params.battery.simple_charge_efficiency,
                    discharge_efficiency=self.params.battery.simple_discharge_efficiency,
                    initial_charge_kwh=self.params.battery.simple_initial_charge_kwh
                )
            
            hh = Household(id=hh_id, data=data, battery=battery)
            hh.wallet = self.params.household.initial_wallet
            households.append(hh)
        
        return households
    
    def _init_baseline_components(self) -> None:
        """Initialize baseline simulation components."""
        self.baseline_central_battery = CentralBattery(
            capacity_in_kwh=self.params.battery.central_capacity_kwh,
            charge_efficiency=self.params.battery.central_charge_efficiency,
            discharge_efficiency=self.params.battery.central_discharge_efficiency,
            tax_per_kwh=self.params.battery.central_tax_per_kwh
        )
        self.baseline_households = self._create_households(self.baseline_central_battery)
        self.baseline_price_forecaster = SimpleCityGridPriceForecaster(self.params.grid_price)
    
    def _init_optimized_components(self) -> None:
        """Initialize optimized simulation components."""
        self.optimized_central_battery = CentralBattery(
            capacity_in_kwh=self.params.battery.central_capacity_kwh,
            charge_efficiency=self.params.battery.central_charge_efficiency,
            discharge_efficiency=self.params.battery.central_discharge_efficiency,
            tax_per_kwh=self.params.battery.central_tax_per_kwh
        )
        self.optimized_households = self._create_households(self.optimized_central_battery)
        self.blockchain = Blockchain(history_size=self.params.forecaster.history_size)
        self.forecaster = PerfectForecaster()
        
        # Create optimizer based on type
        if self.params.optimizer.optimizer_type.lower() == "greedy":
            self.optimizer = GreedyOptimizer(
                p2p_transaction_cost=self.params.optimizer.p2p_transaction_cost,
                min_trade_threshold=self.params.optimizer.min_trade_threshold,
                max_neighbors=self.params.optimizer.max_neighbors,
                battery_target_pct=self.params.optimizer.battery_target_pct,
                p2p_price_factor=self.params.optimizer.p2p_price_factor,
            )
        else:  # "convex"
            self.optimizer = ConvexOptimizer(
                p2p_transaction_cost=self.params.optimizer.p2p_transaction_cost,
                min_trade_threshold=self.params.optimizer.min_trade_threshold,
                solver=self.params.optimizer.solver,
                warm_start=self.params.optimizer.warm_start,
                max_neighbors=self.params.optimizer.max_neighbors,
                battery_charge_rate_factor=self.params.optimizer.battery_charge_rate_factor,
                wallet_penalty_weight=self.params.optimizer.wallet_penalty_weight
            )
        
        self.optimized_price_forecaster = SimpleCityGridPriceForecaster(self.params.grid_price)
        self.token_to_battery = TokenToBattery()
        self.local_price_estimator = PriceEstimator(
            p2p_price_factor=self.params.optimizer.p2p_price_factor
        )
    
    def run_baseline_step(self, step: int) -> SimulationState:
        """Execute one step of baseline simulation."""
        step_data: list[StepData] = []
        total_prod = 0.0
        total_cons = 0.0
        
        price_forecast = self.baseline_price_forecaster.forecast([], forecast_size=1)
        buy_price = price_forecast["buy"][0]
        sell_price = price_forecast["sell"][0]
        
        # Get grid capacity for this step
        grid_cap = self.grid_capacity.get_capacity(step)
        total_grid_buy = 0.0
        total_grid_sell = 0.0
        
        for hh in self.baseline_households:
            data = hh.get_sensor_data(step)
            production = data["production"]
            consumption = data["consumption"]
            
            total_prod += production
            total_cons += consumption
            
            # Calculate available battery energy WITHOUT withdrawing it
            # get_stored_kwh() returns effective energy after discharge efficiency
            available_battery = hh.battery.get_stored_kwh()
            
            # Net energy: production + battery - consumption
            net_energy = production + available_battery - consumption
            
            grid_buy = 0.0
            grid_sell = 0.0
            
            if net_energy < 0:
                # Need more energy - first use battery, then buy from grid
                # Actually withdraw from battery what we can
                battery_used = hh.battery.retrieve_energy(available_battery)
                remaining_deficit = consumption - production - battery_used
                
                if remaining_deficit > 0:
                    # Need to buy from grid (limited by capacity)
                    available = grid_cap["import"] - total_grid_buy
                    bought = min(remaining_deficit, available)
                    
                    hh.wallet -= bought * buy_price
                    total_grid_buy += bought
                    grid_buy = bought
            else:
                # Have excess energy - store in battery, sell remainder to grid
                excess = production - consumption
                
                # Store what we can in battery
                old_stored = hh.battery.get_fields()["stored_kwh"]
                hh.battery.store_energy(excess)
                new_stored = hh.battery.get_fields()["stored_kwh"]
                stored_amount = new_stored - old_stored
                
                # Sell remainder to grid (limited by capacity)
                sellable = excess - stored_amount / hh.battery.get_fields().get("charge_efficiency", 1.0)
                if sellable > 0:
                    available = grid_cap["export"] - total_grid_sell
                    sold = min(sellable, available)
                    
                    hh.wallet += sold * sell_price
                    total_grid_sell += sold
                    grid_sell = sold
            
            # Calculate battery percentage for visualization
            battery_capacity = hh.battery.get_capacity_in_kwh()
            battery_stored_raw = hh.battery.get_fields()["stored_kwh"]
            battery_pct = (battery_stored_raw / battery_capacity * 100) if battery_capacity > 0 else 0
            
            step_data.append(StepData(
                step=step,
                household_id=hh.id,
                production=round(production, 4),
                consumption=round(consumption, 4),
                stored_kwh=round(battery_stored_raw, 4),
                battery_pct=round(battery_pct, 2),
                wallet=round(hh.wallet, 4),
                grid_buy=round(grid_buy, 4),
                grid_sell=round(grid_sell, 4),
                grid_import_capacity=round(grid_cap["import"], 4),
                grid_export_capacity=round(grid_cap["export"], 4)
            ))
        
        self.data_collector.record_step("baseline", step, step_data)
        
        return SimulationState(
            step=step,
            households_data=step_data,
            total_production=total_prod,
            total_consumption=total_cons,
            total_wallet=sum(d.wallet for d in step_data)
        )
    
    def run_optimized_step(self, step: int) -> SimulationState:
        """Execute one step of optimized simulation with P2P trading."""
        step_data: list[StepData] = []
        
        households_forecasts = {}
        current_consumption = {}
        current_production = {}
        overall = {"production": 0.0, "consumption": 0.0}
        
        for hh in self.optimized_households:
            info = hh.get_sensor_data(step)
            current_consumption[hh.id] = info["consumption"]
            current_production[hh.id] = info["production"]
            overall["production"] += info["production"]
            overall["consumption"] += info["consumption"]
            
            self.blockchain.add_household_data(
                id=hh.id,
                production=info["production"],
                consumption=info["consumption"],
                wallet=info["wallet"],
                stored_kwh=info["stored_kwh"]
            )
            
            households_forecasts[hh.id] = self.forecaster.forecast(
                household=hh,
                iteration=step,
                forecast_size=self.params.forecaster.prediction_size
            )
        
        city_price_forecast = self.optimized_price_forecaster.forecast(
            [], 
            forecast_size=self.params.forecaster.city_grid_price_prediction_size
        )
        
        # Get grid capacity forecasts for optimizer
        grid_import_forecast = [
            self.grid_capacity.get_import_capacity(step + t) 
            for t in range(self.params.forecaster.city_grid_price_prediction_size)
        ]
        grid_export_forecast = [
            self.grid_capacity.get_export_capacity(step + t) 
            for t in range(self.params.forecaster.city_grid_price_prediction_size)
        ]
        
        # Run optimizer with grid capacity constraints
        trade_plan = self.optimizer.optimize(
            self.optimized_households,
            households_forecasts,
            city_price_forecast["buy"],
            city_price_forecast["sell"],
            current_consumption,
            current_production,
            self.local_price_estimator,
            grid_import_forecast,
            grid_export_forecast
        )
        
        finalized_offers = self._finalize_offers(trade_plan)
        
        # Record trades in blockchain
        taxes = self.optimized_central_battery.get_households_tax_in_kwh()
        for hh_id in finalized_offers:
            finalized_offers[hh_id]["central_battery_tax"] = taxes.get(hh_id, 0.0)
        
        self.blockchain.add_trades(finalized_offers)
        
        # Execute trades
        grid_buy_price = city_price_forecast["buy"][0]
        local_price = self.local_price_estimator.calculate_price(
            overall["production"], overall["consumption"], grid_buy_price
        )
        self.blockchain.trade_event(
            local_energy_price=local_price,
            city_buy_price=grid_buy_price,
            city_sell_price=city_price_forecast["sell"][0]
        )
        
        # Update batteries and wallets
        for hh in self.optimized_households:
            self.token_to_battery.handle(hh.battery, self.blockchain.households[hh.id].token)
            hh.wallet = self.blockchain.households[hh.id].wallet
            
            offer = finalized_offers.get(hh.id, {})
            trades_str = "-".join([
                f"{t['seller']}:{round(t['amount'], 4)}" 
                for t in offer.get("trades_from", [])
            ])
            
            # Calculate P2P amounts
            p2p_buy = sum(t["amount"] for t in offer.get("trades_from", []))
            p2p_sell = sum(
                t["amount"] for other_offer in finalized_offers.values() 
                for t in other_offer.get("trades_from", []) 
                if t.get("seller") == hh.id
            )
            
            # Calculate battery percentage
            battery_capacity = hh.battery.get_capacity_in_kwh()
            battery_stored_raw = hh.battery.get_fields()["stored_kwh"]
            battery_pct = (battery_stored_raw / battery_capacity * 100) if battery_capacity > 0 else 0
            
            step_data.append(StepData(
                step=step,
                household_id=hh.id,
                production=round(current_production[hh.id], 4),
                consumption=round(current_consumption[hh.id], 4),
                stored_kwh=round(battery_stored_raw, 4),
                battery_pct=round(battery_pct, 2),
                wallet=round(hh.wallet, 4),
                grid_buy=round(offer.get("amount_from_city", 0.0), 4),
                grid_sell=round(offer.get("amount_to_city", 0.0), 4),
                p2p_trades=trades_str,
                p2p_buy_amount=round(p2p_buy, 4),
                p2p_sell_amount=round(p2p_sell, 4),
                central_battery_tax=round(offer.get("central_battery_tax", 0.0), 4),
                grid_import_capacity=round(grid_import_forecast[0], 4),
                grid_export_capacity=round(grid_export_forecast[0], 4)
            ))
        
        self.data_collector.record_step("optimized", step, step_data)
        
        return SimulationState(
            step=step,
            households_data=step_data,
            total_production=overall["production"],
            total_consumption=overall["consumption"],
            total_wallet=sum(d.wallet for d in step_data)
        )
    
    def _finalize_offers(self, trade_plan: list[dict]) -> dict[str, dict]:
        """Transform optimizer output to finalized offers format."""
        finalized = {}
        for offer in trade_plan:
            hh_stats = {
                "trades_from": [
                    {"seller": buy["from"], "amount": buy["amount"]}
                    for buy in offer["buys"]
                ],
                "amount_from_city": offer["buy_from_city"],
                "amount_to_city": offer["sell_to_city"]
            }
            finalized[offer["id"]] = hh_stats
        return finalized
    
    def run_synchronized(self) -> Generator[tuple[SimulationState, SimulationState], None, None]:
        """
        Generator that yields both simulation states after each step.
        Allows GUI to update between steps.
        """
        for step in range(self.params.simulation_steps):
            logger.info(f"------Step: {step + 1}-------")
            
            baseline_state = self.run_baseline_step(step)
            optimized_state = self.run_optimized_step(step)
            
            yield baseline_state, optimized_state
        
        self.data_collector.notify_complete("baseline")
        self.data_collector.notify_complete("optimized")
    
    def run_all(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run all steps and return final DataFrames."""
        for _ in self.run_synchronized():
            pass
        
        return (
            self.data_collector.get_dataframe("baseline"),
            self.data_collector.get_dataframe("optimized")
        )
