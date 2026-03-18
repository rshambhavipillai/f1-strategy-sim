"""
Data preparation module for F1 lap data.
Loads and preprocesses FastF1 lap data.
"""

import pandas as pd
import numpy as np
import fastf1
from pathlib import Path
from config import CACHED_DATA_PATH, TRAINING_SEASONS, TRAINING_TRACKS


def load_lap_data(years, gp_names=None):
    """
    Load lap data from FastF1 for specified years and races.
    
    Args:
        years: list of years (e.g., [2022, 2023, 2024])
        gp_names: list of GP names (e.g., ["Bahrain", "Saudi Arabia"]), 
                  if None use all available
    
    Returns:
        pd.DataFrame with per-lap features and target lap time
    """
    laps_list = []
    
    for year in years:
        season = fastf1.get_session(year, 1, 'R')  # Start with first race
        season.load()
        
        for gp_num in range(1, 23):  # F1 has up to 22 races
            try:
                session = fastf1.get_session(year, gp_num, 'R')
                session.load()
                
                laps = session.laps
                if laps is not None and len(laps) > 0:
                    laps['year'] = year
                    laps['gp_number'] = gp_num
                    laps['gp_name'] = session.event['EventName']
                    laps_list.append(laps)
            except Exception as e:
                print(f"Could not load {year} GP {gp_num}: {e}")
    
    if not laps_list:
        raise ValueError("No lap data loaded. Check year/GP parameters.")
    
    df_laps = pd.concat(laps_list, ignore_index=True)
    return df_laps


def build_lap_features(df_laps):
    """
    Build engineered features from raw lap data.
    
    Args:
        df_laps: pd.DataFrame with raw lap data from FastF1
    
    Returns:
        pd.DataFrame with feature matrix and target column
    """
    df = df_laps.copy()
    
    # Target: lap time in seconds
    df['lap_time_s'] = df['LapTime'].dt.total_seconds()
    
    # Remove laps with NaN lap times (includes in/outlaps)
    df = df.dropna(subset=['lap_time_s'])
    
    # Basic features
    df['lap_number'] = df['LapNumber']
    df['compound'] = df['Compound'].fillna('unknown')
    df['driver_id'] = df['Driver']
    df['track_id'] = df['gp_name'].fillna('unknown')
    
    # Tyre age: number of laps on current tyre
    df['tyre_age_laps'] = df.groupby(['year', 'gp_number', 'Driver']).cumcount() + 1
    
    # Fuel proxy: approximate remaining laps (assume race - current lap)
    # This is simplified; better would be actual fuel data if available
    df['laps_remaining_proxy'] = 57  # Placeholder, should vary by track
    
    # Safety Car / VSC flags (simplified - check available columns)
    df['under_safety_car'] = 0  # Placeholder
    df['under_vsc'] = 0  # Placeholder
    
    # Driver position info (if available)
    df['position'] = df['Position'].fillna(-1).astype(int)
    
    # ==== NEW: Weather & Environmental Features ====
    
    # Track temperature (affects tyre and track performance)
    # FastF1 provides track temperature in session.weather
    # Normalize: lower temp = slower (cold tyres), higher = faster (optimal window)
    if 'TrackTemp' in df.columns:
        df['track_temp'] = df['TrackTemp'].fillna(df['TrackTemp'].median())
        # Normalize to 0-1 range (typical F1 track temps: 20-60°C)
        df['track_temp_norm'] = (df['track_temp'] - 20) / 40
        df['track_temp_norm'] = df['track_temp_norm'].clip(0, 1)
    else:
        df['track_temp_norm'] = 0.5  # Default: neutral
    
    # Ambient air temperature (affects engine power, tyre performance)
    if 'AirTemp' in df.columns:
        df['air_temp'] = df['AirTemp'].fillna(df['AirTemp'].median())
        # Normalize: typical range 5-30°C
        df['air_temp_norm'] = (df['air_temp'] - 5) / 25
        df['air_temp_norm'] = df['air_temp_norm'].clip(0, 1)
    else:
        df['air_temp_norm'] = 0.5  # Default: neutral
    
    # Fuel load proxy: estimate from fuel consumed
    # Cars consume ~1.5-2 kg per lap, start with ~110 kg
    if 'FuelLoad' in df.columns:
        df['fuel_load'] = df['FuelLoad'].fillna(df['FuelLoad'].median())
        # Normalize fuel: 0 = nearly empty, 1 = full
        df['fuel_load_norm'] = df['fuel_load'] / 110
        df['fuel_load_norm'] = df['fuel_load_norm'].clip(0, 1)
    else:
        # Estimate from fuel consumed
        df['fuel_load_norm'] = (57 - df['lap_number']) / 57  # Decreases over race
        df['fuel_load_norm'] = df['fuel_load_norm'].clip(0, 1)
    
    # Wet/dry conditions flag
    # If track temp < 15°C or rain data available, likely wet
    df['likely_wet'] = ((df['track_temp_norm'] < 0.3) | 
                         (df.get('TrackStatus', '').str.contains('WET', case=False, na=False))).astype(int)
    
    # ==== END: Weather Features ====
    
    # Select and return feature columns
    feature_cols = [
        'lap_number',
        'compound',
        'tyre_age_laps',
        'laps_remaining_proxy',
        'under_safety_car',
        'under_vsc',
        'position',
        'track_temp_norm',      # NEW
        'air_temp_norm',        # NEW
        'fuel_load_norm',       # NEW
        'likely_wet',           # NEW
    ]
    
    target_col = 'lap_time_s'
    
    # Ensure no missing values in features
    df = df.dropna(subset=feature_cols + [target_col])
    
    return df[feature_cols + [target_col]], feature_cols, target_col


def prepare_dataset(years, gp_names=None, use_cache=True):
    """
    Convenience function to load and prepare dataset.
    
    Args:
        years: list of years
        gp_names: list of GP names
        use_cache: whether to load from cache if available
    
    Returns:
        tuple: (df_features, feature_cols, target_col)
    """
    if use_cache and CACHED_DATA_PATH.exists():
        print(f"Loading cached data from {CACHED_DATA_PATH}")
        return pd.read_parquet(CACHED_DATA_PATH)
    
    print("Loading fresh lap data from FastF1...")
    df_laps = load_lap_data(years, gp_names)
    
    print("Building features...")
    df_features, feature_cols, target_col = build_lap_features(df_laps)
    
    # Cache for future use
    df_features.to_parquet(CACHED_DATA_PATH, index=False)
    print(f"Dataset cached to {CACHED_DATA_PATH}")
    
    return df_features, feature_cols, target_col
