"""
Configuration and constants for F1 Strategy Simulator.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
NOTEBOOKS_DIR.mkdir(exist_ok=True)

# Model paths
LAP_TIME_MODEL_PATH = MODELS_DIR / "lap_time_model.pkl"
CACHED_DATA_PATH = DATA_DIR / "cached_laps.parquet"

# Training seasons and tracks
TRAINING_SEASONS = [2022, 2023, 2024]
TRAINING_TRACKS = [
    "Bahrain",
    "Saudi Arabia",
    "Australia",
    "Japan",
    "China",
    "Miami",
    "Emilia Romagna",
    "Monaco",
    "Canada",
    "Spain",
    "Austria",
    "Great Britain",
    "Hungary",
    "Belgium",
    "Netherlands",
    "Italy",
    "Singapore",
    "Japan",
    "United States",
    "Mexico",
    "Brazil",
    "Abu Dhabi",
]

# Default simulation track
DEFAULT_TRACK = "Bahrain"

# Simulation parameters
DEFAULT_NUM_DRIVERS = 4  # For v1, simple setup with 4-6 cars
DEFAULT_TOTAL_LAPS = 57  # Adjust based on track

# Pit stop delta (seconds) - approximate
PIT_STOP_DELTA = 21.0

# Tyre compound info
COMPOUNDS = ["soft", "medium", "hard"]
COMPOUND_PACE_DELTA = {
    "soft": 0.0,  # baseline
    "medium": 0.25,
    "hard": 0.50,
}

# Tyre degradation (seconds per lap)
TYRE_DEGRADATION = {
    "soft": 0.08,
    "medium": 0.05,
    "hard": 0.03,
}

# Simulation noise/randomness (standard deviation in seconds)
SIMULATION_NOISE_STD = 0.5

# Number of simulations per strategy (default)
DEFAULT_NUM_SIMULATIONS = 200

# ===== Weather & Environmental Parameters =====

# Track temperature effects (normalized 0-1, where 0.5 = optimal)
# Lower temp = cold tyres (slower), higher = risk of overheating
OPTIMAL_TRACK_TEMP = 0.5

# Air temperature effects on engine power and fuel consumption
OPTIMAL_AIR_TEMP = 0.5

# Fuel load effects on pace
# Affects: acceleration, braking, grip in high-speed corners
FUEL_CONSUMPTION_PER_LAP = 1.6  # kg/lap

# Wet weather impact on lap times (multiplicative)
WET_WEATHER_PACE_MULT = 1.15  # 15% slower in wet
