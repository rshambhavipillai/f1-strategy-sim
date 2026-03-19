"""
Simulation engine for F1 races.
Runs Monte Carlo simulations with strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from state import RaceState, DriverState, Strategy
from config import (
    PIT_STOP_DELTA, 
    TYRE_DEGRADATION, 
    COMPOUND_PACE_DELTA,
    SIMULATION_NOISE_STD
)


def safe_encode_value(encoder, value, default_code=0):
    """
    Safely encode a value using a LabelEncoder, handling unseen labels.
    
    Args:
        encoder: fitted LabelEncoder
        value: value to encode
        default_code: code to use if value is unseen (default: 0, first class)
    
    Returns:
        encoded value (int)
    """
    value_str = str(value).lower()  # Normalize to lowercase
    
    try:
        # Try to find matching class (case-insensitive)
        for i, class_val in enumerate(encoder.classes_):
            if str(class_val).lower() == value_str:
                return i
        
        # If no match found, use default
        return default_code
    except Exception:
        return default_code



def predict_lap_time(model, feature_cols, track_id, driver_state, race_state, encoders):
    """
    Predict lap time for a driver in current conditions.
    
    Args:
        model: trained regression model
        feature_cols: list of feature column names
        track_id: track identifier
        driver_state: DriverState object
        race_state: RaceState object
        encoders: dict of LabelEncoders for categorical features
    
    Returns:
        predicted lap time in seconds
    """
    # Build feature dict
    features = {}
    
    for col in feature_cols:
        if col == 'lap_number':
            features[col] = race_state.current_lap
        elif col == 'compound':
            features[col] = driver_state.compound
        elif col == 'tyre_age_laps':
            features[col] = driver_state.tyre_age_laps
        elif col == 'laps_remaining_proxy':
            features[col] = race_state.total_laps - race_state.current_lap
        elif col == 'under_safety_car':
            features[col] = 0  # Placeholder, could be updated
        elif col == 'under_vsc':
            features[col] = 0  # Placeholder
        elif col == 'position':
            features[col] = driver_state.position
        elif col == 'track_id':
            features[col] = track_id
        # NEW: Handle weather/environmental features
        elif col == 'track_temp_norm':
            features[col] = 0.5  # Default: neutral temp (can be updated from race_state)
        elif col == 'air_temp_norm':
            features[col] = 0.5  # Default: neutral temp
        elif col == 'fuel_load_norm':
            # Approximate: fuel depletes over race
            features[col] = max(0.2, 1.0 - (race_state.current_lap / race_state.total_laps))
        elif col == 'likely_wet':
            features[col] = 0  # Default: dry (can be updated from weather data)
        else:
            features[col] = 0  # Default
    
    # Encode categorical features with safe encoding
    df_row = pd.DataFrame([features])
    for col in ['compound', 'track_id']:
        if col in feature_cols and col in encoders:
            # Use safe encoding to handle unseen labels
            encoded_val = safe_encode_value(encoders[col], features[col])
            df_row[col] = encoded_val
    
    # Get model prediction
    X = df_row[feature_cols]
    base_lap_time = model.predict(X)[0]
    
    # Apply tyre degradation
    tyre_deg = TYRE_DEGRADATION.get(driver_state.compound, 0.0)
    tyre_deg_delta = tyre_deg * driver_state.tyre_age_laps
    
    # Apply compound pace delta
    compound_delta = COMPOUND_PACE_DELTA.get(driver_state.compound, 0.0)
    
    # Apply driver-specific pace offset
    pace_delta = driver_state.pace_offset_s
    
    # Total lap time
    lap_time = base_lap_time + tyre_deg_delta + compound_delta + pace_delta
    
    return max(lap_time, 60.0)  # Ensure minimum lap time


def update_positions(drivers_lap_times: Dict[str, float]) -> Dict[str, int]:
    """
    Update driver positions based on lap times.
    
    Args:
        drivers_lap_times: dict {driver_id: lap_time_s}
    
    Returns:
        dict {driver_id: position}
    """
    sorted_drivers = sorted(
        drivers_lap_times.items(),
        key=lambda x: x[1]
    )
    return {
        driver_id: position + 1
        for position, (driver_id, _) in enumerate(sorted_drivers)
    }


def run_single_simulation(race_state, strategy, model, feature_cols, encoders, randomness=True):
    """
    Run a single race simulation with given strategy.
    
    Args:
        race_state: RaceState object
        strategy: Strategy object for the player's driver
        model: trained lap time model
        feature_cols: feature column names
        encoders: dict of LabelEncoders
        randomness: whether to add Gaussian noise to lap times
    
    Returns:
        dict with simulation results for player's driver
    """
    # Clone state for simulation
    sim_state = race_state.copy()
    player_driver_id = 'Player'  # Player is always the primary driver we're evaluating
    
    # Initialize tracking
    player_total_time = 0.0
    lap_times_history = {driver_id: [] for driver_id in sim_state.drivers}
    pit_lap_count = {driver_id: 0 for driver_id in sim_state.drivers}
    
    # Simulation loop
    for lap in range(sim_state.current_lap, sim_state.total_laps + 1):
        sim_state.current_lap = lap
        drivers_lap_times = {}
        
        for driver_id, driver_state in sim_state.drivers.items():
            # Check for pit stop
            if driver_id == player_driver_id and lap in strategy.pit_laps:
                pit_index = strategy.pit_laps.index(lap)
                driver_state.compound = strategy.tyre_plan[pit_index]
                driver_state.tyre_age_laps = 0
                lap_time = PIT_STOP_DELTA
            else:
                # Predict lap time
                lap_time = predict_lap_time(
                    model, feature_cols, 
                    sim_state.track_id, 
                    driver_state, 
                    sim_state, 
                    encoders
                )
                # Increment tyre age
                driver_state.tyre_age_laps += 1
            
            # Add noise
            if randomness:
                noise = np.random.normal(0, SIMULATION_NOISE_STD)
                lap_time += noise
            
            drivers_lap_times[driver_id] = lap_time
            lap_times_history[driver_id].append(lap_time)
            
            if driver_id == player_driver_id:
                player_total_time += lap_time
        
        # Update positions
        positions = update_positions(drivers_lap_times)
        for driver_id, position in positions.items():
            sim_state.drivers[driver_id].position = position
    
    # Compile results
    player_driver = sim_state.drivers[player_driver_id]
    
    return {
        "strategy_name": strategy.name,
        "finishing_position": player_driver.position,
        "race_time_s": player_total_time,
        "total_laps": sim_state.total_laps,
    }
