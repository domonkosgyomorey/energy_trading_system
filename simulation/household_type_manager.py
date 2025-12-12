"""
Household type selection system.
Allows filtering households by type (prefix pattern in ID).
"""
from dataclasses import dataclass, field
import re
from typing import Dict, List
import random


@dataclass
class HouseholdTypeConfig:
    """Configuration for selecting households by type."""
    type_counts: Dict[str, int] = field(default_factory=dict)  # e.g., {"kh": 5, "ik": 3}
    random_selection: bool = True  # If True, randomly select from available IDs
    
    def get_total_count(self) -> int:
        """Get total number of households to select."""
        return sum(self.type_counts.values())
    
    def is_empty(self) -> bool:
        """Check if any types are selected."""
        return len(self.type_counts) == 0 or self.get_total_count() == 0


class HouseholdTypeManager:
    """Manages household type analysis and selection."""
    
    def __init__(self):
        self.available_types: Dict[str, List[str]] = {}
        self.all_household_ids: List[str] = []
    
    def analyze_household_ids(self, household_ids: List[str]) -> Dict[str, List[str]]:
        """
        Analyze household IDs and group by type prefix.
        
        Args:
            household_ids: List of household IDs
            
        Returns:
            Dictionary mapping type prefix to list of IDs
        """
        self.all_household_ids = list(household_ids)
        self.available_types = {}
        
        for hh_id in household_ids:
            # Extract prefix (letters before numbers)
            match = re.match(r'^([a-zA-Z_]+)(\d+)$', str(hh_id))
            if match:
                prefix = match.group(1)
                if prefix not in self.available_types:
                    self.available_types[prefix] = []
                self.available_types[prefix].append(str(hh_id))
            else:
                # Handle IDs without clear pattern
                if '_other' not in self.available_types:
                    self.available_types['_other'] = []
                self.available_types['_other'].append(str(hh_id))
        
        return self.available_types
    
    def get_type_summary(self) -> List[tuple[str, int]]:
        """
        Get summary of available types.
        
        Returns:
            List of (type_name, count) tuples, sorted by type name
        """
        return sorted([(t, len(ids)) for t, ids in self.available_types.items()])
    
    def select_households(self, config: HouseholdTypeConfig) -> List[str]:
        """
        Select household IDs based on configuration.
        
        Args:
            config: Configuration specifying how many of each type to select
            
        Returns:
            List of selected household IDs
        """
        if config.is_empty():
            return []
        
        selected = []
        
        for type_prefix, count in config.type_counts.items():
            if type_prefix not in self.available_types:
                continue
            
            available = self.available_types[type_prefix]
            
            if config.random_selection:
                # Randomly select up to 'count' from available
                selected_from_type = random.sample(
                    available, 
                    min(count, len(available))
                )
            else:
                # Take first 'count' IDs
                selected_from_type = available[:count]
            
            selected.extend(selected_from_type)
        
        return selected
    
    def create_balanced_config(self, total_count: int) -> HouseholdTypeConfig:
        """
        Create a balanced configuration with approximately equal distribution.
        
        Args:
            total_count: Total number of households to select
            
        Returns:
            HouseholdTypeConfig with balanced type distribution
        """
        if not self.available_types:
            return HouseholdTypeConfig()
        
        type_counts = {}
        num_types = len(self.available_types)
        base_count = total_count // num_types
        remainder = total_count % num_types
        
        for i, (type_prefix, available_ids) in enumerate(sorted(self.available_types.items())):
            # Distribute remainder to first few types
            count = base_count + (1 if i < remainder else 0)
            # Don't exceed available count
            count = min(count, len(available_ids))
            if count > 0:
                type_counts[type_prefix] = count
        
        return HouseholdTypeConfig(type_counts=type_counts, random_selection=True)
