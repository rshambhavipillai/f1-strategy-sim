"""Tests for state classes (DriverState, RaceState, Strategy)."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from state import DriverState, RaceState


class TestDriverState:
    """Test DriverState dataclass."""
    
    def test_create_driver_state(self):
        """Test creating a driver state."""
        driver = DriverState(
            driver_id="Hamilton",
            position=1,
            gap_to_leader_s=0.0,
            compound="soft",
            tyre_age_laps=10,
            pace_offset_s=-0.5,
        )
        assert driver.driver_id == "Hamilton"
        assert driver.position == 1
        assert driver.tyre_age_laps == 10
    
    def test_driver_state_copy(self):
        """Test copying driver state."""
        driver = DriverState(
            driver_id="Player",
            position=3,
            gap_to_leader_s=5.0,
            compound="medium",
            tyre_age_laps=8,
        )
        
        driver_copy = driver.copy()
        
        # Modify original
        driver.position = 2
        
        # Copy should be unchanged
        assert driver_copy.position == 3
        assert driver.position == 2
    
    def test_driver_state_pit_stops_tracking(self):
        """Test tracking pit stops."""
        driver = DriverState(
            driver_id="Player",
            position=1,
            gap_to_leader_s=0.0,
            compound="soft",
            tyre_age_laps=0,
            pit_stops=[25, 45],  # Pit at laps 25 and 45
        )
        
        assert len(driver.pit_stops) == 2
        assert 25 in driver.pit_stops


class TestRaceState:
    """Test RaceState dataclass."""
    
    def test_create_race_state(self, mock_race_state):
        """Test creating race state."""
        assert mock_race_state.track_id == "Bahrain"
        assert mock_race_state.current_lap == 20
        assert mock_race_state.total_laps == 57
        assert len(mock_race_state.drivers) == 3
    
    def test_race_state_copy(self, mock_race_state):
        """Test copying race state maintains independence."""
        race_copy = mock_race_state.copy()
        
        # Modify original driver
        mock_race_state.drivers["Player"].position = 1
        
        # Copy should be unchanged
        assert race_copy.drivers["Player"].position == 3
        assert mock_race_state.drivers["Player"].position == 1
    
    def test_race_state_lap_progression(self, mock_race_state):
        """Test race state tracks lap progression."""
        assert mock_race_state.current_lap == 20
        laps_remaining = mock_race_state.total_laps - mock_race_state.current_lap
        assert laps_remaining == 37
    
    def test_multiple_drivers_in_race(self, mock_race_state):
        """Test race with multiple drivers."""
        player = mock_race_state.drivers["Player"]
        rival1 = mock_race_state.drivers["Rival1"]
        
        assert player.position > rival1.position  # Player behind rival1
        assert rival1.gap_to_leader_s == 0.0  # Rival1 is leading
