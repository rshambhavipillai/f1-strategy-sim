"""Tests for strategy validation and execution."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from state import Strategy


class TestStrategyValidation:
    """Test strategy creation and validation."""
    
    def test_valid_strategy_creation(self):
        """Test creating a valid strategy."""
        strategy = Strategy(
            name="1-Stop",
            pit_laps=[30],
            tyre_plan=["hard"]
        )
        assert strategy.name == "1-Stop"
        assert strategy.pit_laps == [30]
        assert strategy.tyre_plan == ["hard"]
    
    def test_two_stop_strategy(self):
        """Test creating a two-stop strategy."""
        strategy = Strategy(
            name="2-Stop",
            pit_laps=[20, 40],
            tyre_plan=["medium", "hard"]
        )
        assert len(strategy.pit_laps) == 2
        assert len(strategy.tyre_plan) == 2
    
    def test_pit_laps_and_tyre_plan_mismatch(self):
        """Test that mismatched pit_laps and tyre_plan raises error."""
        with pytest.raises(ValueError):
            Strategy(
                name="Invalid",
                pit_laps=[25, 45],
                tyre_plan=["hard"]  # Only 1 tyre, 2 pit laps
            )
    
    def test_empty_pit_strategy(self):
        """Test zero-stop (no pit) strategy is valid."""
        strategy = Strategy(
            name="0-Stop",
            pit_laps=[],
            tyre_plan=[]
        )
        assert strategy.pit_laps == []
        assert strategy.tyre_plan == []


class TestPitLapValidation:
    """Test pit lap validation logic."""
    
    def test_valid_pit_lap_within_race(self):
        """Test pit lap is within race distance."""
        from utils import validate_pit_laps
        
        pit_laps = [25, 45]
        total_laps = 57
        
        assert validate_pit_laps(pit_laps, total_laps) == True
    
    def test_pit_lap_exceeds_race(self):
        """Test pit lap beyond total laps is invalid."""
        from utils import validate_pit_laps
        
        pit_laps = [60]  # Beyond 57 laps
        total_laps = 57
        
        with pytest.raises(ValueError):
            validate_pit_laps(pit_laps, total_laps)
    
    def test_pit_lap_below_race_start(self):
        """Test pit lap before race starts is invalid."""
        from utils import validate_pit_laps
        
        pit_laps = [0]  # Before lap 1
        total_laps = 57
        
        with pytest.raises(ValueError):
            validate_pit_laps(pit_laps, total_laps)
    
    def test_duplicate_pit_laps(self):
        """Test duplicate pit laps are invalid."""
        from utils import validate_pit_laps
        
        pit_laps = [25, 25]  # Duplicate
        total_laps = 57
        
        with pytest.raises(ValueError):
            validate_pit_laps(pit_laps, total_laps)
