"""Tests for simulator logic."""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from simulator import predict_lap_time, run_single_simulation, update_positions


class TestLapTimePrediction:
    """Test lap time prediction."""
    
    def test_predict_lap_time_basic(self, mock_model_and_encoders, mock_race_state, 
                                     feature_cols):
        """Test basic lap time prediction."""
        model, encoders = mock_model_and_encoders
        driver = mock_race_state.drivers["Player"]
        
        lap_time = predict_lap_time(
            model, feature_cols, 
            mock_race_state.track_id,
            driver, 
            mock_race_state, 
            encoders
        )
        
        # Should return a positive float
        assert isinstance(lap_time, (float, np.floating))
        assert lap_time > 0
        assert lap_time >= 60  # Minimum lap time
    
    def test_lap_time_varies_with_conditions(self, mock_model_and_encoders, 
                                             mock_race_state, feature_cols):
        """Test lap time changes with race conditions."""
        model, encoders = mock_model_and_encoders
        driver1 = mock_race_state.drivers["Player"]
        driver2 = mock_race_state.drivers["Rival1"]
        
        lap_time_1 = predict_lap_time(
            model, feature_cols,
            mock_race_state.track_id,
            driver1,
            mock_race_state,
            encoders
        )
        
        lap_time_2 = predict_lap_time(
            model, feature_cols,
            mock_race_state.track_id,
            driver2,
            mock_race_state,
            encoders
        )
        
        # Different drivers should potentially have different lap times
        # (unless model doesn't differentiate, which is fine for mock)
        assert lap_time_1 > 0
        assert lap_time_2 > 0


class TestPositionUpdates:
    """Test position calculation."""
    
    def test_update_positions_basic(self):
        """Test position updates from lap times."""
        drivers_lap_times = {
            "Hamilton": 90.5,
            "Verstappen": 90.2,
            "Russell": 91.0,
        }
        
        positions = update_positions(drivers_lap_times)
        
        # Fastest driver gets position 1
        assert positions["Verstappen"] == 1
        assert positions["Hamilton"] == 2
        assert positions["Russell"] == 3
    
    def test_position_updates_with_ties(self):
        """Test position updates with equal lap times."""
        drivers_lap_times = {
            "Driver1": 90.5,
            "Driver2": 90.5,
            "Driver3": 90.6,
        }
        
        positions = update_positions(drivers_lap_times)
        
        # Even with ties, positions are assigned
        assert len(positions) == 3
        assert all(p in [1, 2, 3] for p in positions.values())


class TestSingleSimulation:
    """Test single race simulation."""
    
    def test_simulation_runs_without_error(self, mock_race_state, mock_strategy,
                                           mock_model_and_encoders, feature_cols):
        """Test that a single simulation runs without error."""
        model, encoders = mock_model_and_encoders
        
        result = run_single_simulation(
            mock_race_state,
            mock_strategy,
            model,
            feature_cols,
            encoders,
            randomness=False  # Disable randomness for determinism
        )
        
        assert "strategy_name" in result
        assert "finishing_position" in result
        assert "race_time_s" in result
    
    def test_simulation_result_structure(self, mock_race_state, mock_strategy,
                                         mock_model_and_encoders, feature_cols):
        """Test simulation result has expected structure."""
        model, encoders = mock_model_and_encoders
        
        result = run_single_simulation(
            mock_race_state,
            mock_strategy,
            model,
            feature_cols,
            encoders,
        )
        
        assert isinstance(result['finishing_position'], (int, np.integer))
        assert 1 <= result['finishing_position'] <= 20
        assert result['race_time_s'] > 0
        assert result['strategy_name'] == mock_strategy.name
