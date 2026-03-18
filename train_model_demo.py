"""
Quick demo/mock training script for F1 Strategy Simulator.
Uses synthetic data to train a model quickly for testing purposes.
For production, use train_model.py which downloads real FastF1 data.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from model_train import train_lap_time_model, save_model
from config import LAP_TIME_MODEL_PATH, COMPOUNDS

def generate_synthetic_data(n_samples=2000):
    """
    Generate synthetic lap data for quick testing.
    
    Args:
        n_samples: number of synthetic lap records
    
    Returns:
        tuple: (df_features, feature_cols, target_col)
    """
    np.random.seed(42)
    
    # Generate synthetic features
    data = {
        'lap_number': np.random.randint(1, 60, n_samples),
        'compound': np.random.choice(COMPOUNDS, n_samples),
        'tyre_age_laps': np.random.randint(0, 40, n_samples),
        'laps_remaining_proxy': np.random.randint(10, 60, n_samples),
        'under_safety_car': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'under_vsc': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
        'position': np.random.randint(1, 21, n_samples),
        'track_temp_norm': np.random.uniform(0.2, 0.8, n_samples),  # NEW
        'air_temp_norm': np.random.uniform(0.2, 0.8, n_samples),    # NEW
        'fuel_load_norm': np.random.uniform(0.2, 1.0, n_samples),   # NEW (starts full, depletes)
        'likely_wet': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),  # NEW
    }
    
    df = pd.DataFrame(data)
    
    # Generate synthetic target (lap time in seconds)
    # Base lap time around 90 seconds (approximate F1 lap)
    base_time = 90.0
    
    # Add effects from features
    compound_effect = df['compound'].map({
        'soft': -0.5,
        'medium': 0.0,
        'hard': 0.8
    })
    
    tyre_age_effect = df['tyre_age_laps'] * 0.03  # degradation
    
    position_effect = (df['position'] - 1) * 0.2  # trailing cars slightly faster
    
    # NEW: Temperature effects
    # Optimal at 0.5, worse at extremes
    temp_effect = (
        0.5 * np.abs(df['track_temp_norm'] - 0.5) +
        0.3 * np.abs(df['air_temp_norm'] - 0.5)
    )
    
    # NEW: Fuel load effect (heavier = slower)
    fuel_effect = (1.0 - df['fuel_load_norm']) * 0.3  # Up to 0.3s slower when heavy
    
    # NEW: Wet weather effect
    wet_effect = df['likely_wet'] * 1.5  # 1.5s slower when wet
    
    # Noise
    noise = np.random.normal(0, 0.5, n_samples)
    
    # Target
    df['lap_time_s'] = (
        base_time + 
        compound_effect + 
        tyre_age_effect + 
        position_effect + 
        temp_effect +
        fuel_effect +
        wet_effect +
        noise
    )
    
    feature_cols = [
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
    
    target_col = 'lap_time_s'
    
    return df[feature_cols + [target_col]], feature_cols, target_col


def main():
    print("=" * 60)
    print("F1 Strategy Simulator - Quick Demo Training")
    print("(Using synthetic data for fast testing)")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print("\n[1/3] Generating synthetic lap data...")
    
    try:
        df_features, feature_cols, target_col = generate_synthetic_data(n_samples=2000)
        print(f"✓ Generated {len(df_features)} synthetic lap records")
        print(f"✓ Features: {feature_cols}")
        print(f"✓ Target: {target_col}")
    except Exception as e:
        print(f"✗ Error generating data: {e}")
        return 1
    
    # Step 2: Train model
    print("\n[2/3] Training lap time prediction model...")
    
    try:
        model, feature_cols, target_col, encoders = train_lap_time_model(
            df_features,
            feature_cols,
            target_col,
            model_type='gradientboosting'
        )
        print(f"✓ Model training complete")
    except Exception as e:
        print(f"✗ Error training model: {e}")
        return 1
    
    # Step 3: Save model
    print("\n[3/3] Saving trained model and metadata...")
    
    try:
        save_model(model, feature_cols, target_col, encoders)
        print(f"✓ Model saved to {LAP_TIME_MODEL_PATH}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("✓ Quick demo training complete!")
    print("=" * 60)
    print("\nYou can now run the Streamlit app:")
    print(f"  streamlit run app_streamlit.py")
    print("\n" + "=" * 60)
    print("IMPORTANT: For production use with real F1 data:")
    print("  python train_model.py")
    print("  (Downloads ~1-2 GB of FastF1 data, takes 30+ minutes)")
    print("=" * 60)
    print("")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
