"""Pytest configuration and fixtures."""

import sys
from pathlib import Path
import pytest
import numpy as np
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from state import RaceState, DriverState, Strategy
from config import COMPOUNDS


@pytest.fixture
def mock_race_state():
    """Create a mock race state for testing."""
    drivers = {
        "Player": DriverState(
            driver_id="Player",
            position=3,
            gap_to_leader_s=5.0,
            compound="soft",
            tyre_age_laps=5,
            pace_offset_s=0.0,
        ),
        "Rival1": DriverState(
            driver_id="Rival1",
            position=1,
            gap_to_leader_s=0.0,
            compound="medium",
            tyre_age_laps=8,
            pace_offset_s=-0.2,
        ),
        "Rival2": DriverState(
            driver_id="Rival2",
            position=2,
            gap_to_leader_s=2.5,
            compound="hard",
            tyre_age_laps=15,
            pace_offset_s=0.1,
        ),
    }
    
    return RaceState(
        track_id="Bahrain",
        current_lap=20,
        total_laps=57,
        drivers=drivers,
    )


@pytest.fixture
def mock_strategy():
    """Create a mock pit strategy for testing."""
    return Strategy(
        name="Test Strategy",
        pit_laps=[25, 45],
        tyre_plan=["medium", "hard"],
    )


@pytest.fixture
def mock_model_and_encoders():
    """Create mock model and encoders for testing."""
    from sklearn.preprocessing import LabelEncoder
    
    # Minimal mock model with predict method
    class MockModel:
        def predict(self, X):
            return np.array([90.0] * len(X))
    
    encoders = {
        'compound': LabelEncoder().fit(COMPOUNDS),
        'track_id': LabelEncoder().fit(['Bahrain', 'Monza', 'Monaco']),
    }
    
    return MockModel(), encoders


@pytest.fixture
def feature_cols():
    """Return standard feature columns."""
    return [
        'lap_number',
        'compound',
        'tyre_age_laps',
        'laps_remaining_proxy',
        'under_safety_car',
        'under_vsc',
        'position',
        'track_temp_norm',
        'air_temp_norm',
        'fuel_load_norm',
        'likely_wet',
    ]
