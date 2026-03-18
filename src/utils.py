"""
Utility functions for F1 Strategy Simulator.
"""

import pandas as pd
from datetime import timedelta
from typing import Union


def seconds_to_laptime(seconds: float) -> str:
    """
    Convert seconds to MM:SS.mmm format.
    
    Args:
        seconds: time in seconds
    
    Returns:
        formatted string
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:06.3f}"


def validate_pit_laps(pit_laps, total_laps):
    """
    Validate pit lap numbers are within race.
    
    Args:
        pit_laps: list of pit lap numbers
        total_laps: total race laps
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    for lap in pit_laps:
        if lap < 1 or lap > total_laps:
            raise ValueError(
                f"Pit lap {lap} is outside race range [1, {total_laps}]"
            )
    
    # Check no duplicate pit laps
    if len(pit_laps) != len(set(pit_laps)):
        raise ValueError("Duplicate pit laps not allowed")
    
    return True


def calculate_pit_delta_laps(pit_stop_time_s, avg_lap_time_s):
    """
    Convert pit stop time to equivalent laps lost.
    
    Args:
        pit_stop_time_s: pit stop duration in seconds
        avg_lap_time_s: average lap time in seconds
    
    Returns:
        equivalent laps lost as float
    """
    return pit_stop_time_s / avg_lap_time_s


def format_strategy_table(strategy_results):
    """
    Format strategy results for display in Streamlit.
    
    Args:
        strategy_results: list of dicts from evaluate_strategies
    
    Returns:
        pd.DataFrame formatted for display
    """
    df = pd.DataFrame(strategy_results)
    
    # Round numeric columns
    numeric_cols = [
        'expected_position', 
        'position_std', 
        'podium_probability',
        'win_probability',
        'mean_race_time_s',
        'std_race_time_s'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            if 'probability' in col:
                df[col] = df[col].apply(lambda x: f"{x:.1%}")
            elif 'position' in col or 'time' in col:
                df[col] = df[col].apply(lambda x: f"{x:.2f}")
    
    return df[['strategy_name', 'expected_position', 'podium_probability', 
               'win_probability', 'mean_race_time_s']]
