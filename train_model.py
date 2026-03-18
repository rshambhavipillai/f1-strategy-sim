"""
Training script for F1 Strategy Simulator lap time model.
Loads FastF1 data, builds features, trains model, and saves.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_prep import prepare_dataset
from model_train import train_lap_time_model, save_model
from config import TRAINING_SEASONS, LAP_TIME_MODEL_PATH

def main():
    print("=" * 60)
    print("F1 Strategy Simulator - Model Training Pipeline")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print("\n[1/3] Loading and preparing lap data from FastF1...")
    print(f"      Seasons: {TRAINING_SEASONS}")
    
    try:
        df_features, feature_cols, target_col = prepare_dataset(
            TRAINING_SEASONS,
            use_cache=False  # Force fresh download
        )
        print(f"✓ Loaded {len(df_features)} lap records")
        print(f"✓ Features: {feature_cols}")
        print(f"✓ Target: {target_col}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("\nNote: FastF1 data download may require internet connection.")
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
    print("✓ Training pipeline complete!")
    print("=" * 60)
    print(f"\nYou can now run the Streamlit app:")
    print(f"  streamlit run app_streamlit.py")
    print("")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
