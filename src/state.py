"""
State classes for race simulation.
Defines DriverState, RaceState, and Strategy dataclasses.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List
import copy


@dataclass
class DriverState:
    """Represents the state of a single driver."""
    driver_id: str
    position: int
    gap_to_leader_s: float
    compound: str
    tyre_age_laps: int
    pace_offset_s: float = 0.0  # Offset from nominal pace
    pit_stops: List[int] = field(default_factory=list)
    
    def copy(self):
        """Create a deep copy of this driver state."""
        return replace(
            self,
            pit_stops=copy.copy(self.pit_stops)
        )


@dataclass
class RaceState:
    """Represents the current state of the race."""
    track_id: str
    current_lap: int
    total_laps: int
    drivers: Dict[str, DriverState] = field(default_factory=dict)
    
    def copy(self):
        """Create a deep copy of this race state."""
        drivers_copy = {
            driver_id: driver_state.copy()
            for driver_id, driver_state in self.drivers.items()
        }
        return RaceState(
            track_id=self.track_id,
            current_lap=self.current_lap,
            total_laps=self.total_laps,
            drivers=drivers_copy
        )


@dataclass
class Strategy:
    """Represents a pit stop strategy."""
    name: str
    pit_laps: List[int]  # Lap numbers for pit stops
    tyre_plan: List[str]  # Corresponding tyre compounds (e.g., ['soft', 'hard'])
    
    def __post_init__(self):
        """Validate strategy consistency."""
        if len(self.pit_laps) != len(self.tyre_plan):
            raise ValueError(
                f"pit_laps ({len(self.pit_laps)}) and tyre_plan "
                f"({len(self.tyre_plan)}) must have same length"
            )
