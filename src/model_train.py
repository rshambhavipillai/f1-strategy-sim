"""
Model training module for lap time prediction.
Train and save regression models.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
from config import LAP_TIME_MODEL_PATH


def encode_categorical_features(df, feature_cols, encoders=None):
    """
    Encode categorical features.
    
    Args:
        df: DataFrame with features
        feature_cols: list of feature column names
        encoders: dict of LabelEncoders for categorical features
    
    Returns:
        df_encoded: DataFrame with encoded features
        encoders: dict of fitted encoders
    """
    df_encoded = df.copy()
    if encoders is None:
        encoders = {}
    
    categorical_cols = ['compound', 'track_id']
    
    for col in categorical_cols:
        if col in feature_cols:
            if col not in encoders:
                encoders[col] = LabelEncoder()
                df_encoded[col] = encoders[col].fit_transform(df[col].astype(str))
            else:
                df_encoded[col] = encoders[col].transform(df[col].astype(str))
    
    return df_encoded, encoders


def train_lap_time_model(df_features, feature_cols, target_col, model_type='gradientboosting'):
    """
    Train a lap time prediction model.
    
    Args:
        df_features: pd.DataFrame with features and target
        feature_cols: list of feature columns
        target_col: name of target column
        model_type: 'gradientboosting' or 'randomforest'
    
    Returns:
        model: trained regression model
        feature_cols: list of feature columns used
        target_col: name of target column
        encoders: dict of LabelEncoders for categorical features
    """
    # Encode categorical features
    df_encoded, encoders = encode_categorical_features(df_features, feature_cols)
    
    # Prepare X and y
    X = df_encoded[feature_cols]
    y = df_encoded[target_col]
    
    # Train/val split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print(f"Training {model_type} model...")
    if model_type == 'gradientboosting':
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=1
        )
    else:  # randomforest
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    
    print(f"Training MAE: {mae_train:.4f} s")
    print(f"Validation MAE: {mae_val:.4f} s")
    print(f"Validation RMSE: {rmse_val:.4f} s")
    
    return model, feature_cols, target_col, encoders


def save_model(model, feature_cols, target_col, encoders, path=None):
    """
    Save trained model and metadata to pickle.
    
    Args:
        model: trained model object
        feature_cols: list of feature columns
        target_col: target column name
        encoders: dict of LabelEncoders
        path: path to save model (default: LAP_TIME_MODEL_PATH)
    """
    if path is None:
        path = LAP_TIME_MODEL_PATH
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    package = {
        'model': model,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'encoders': encoders,
    }
    
    with open(path, 'wb') as f:
        pickle.dump(package, f)
    
    print(f"Model saved to {path}")


def load_model(path=None):
    """
    Load trained model and metadata from pickle.
    
    Args:
        path: path to model file (default: LAP_TIME_MODEL_PATH)
    
    Returns:
        model: trained model object
        feature_cols: list of feature columns
        target_col: target column name
        encoders: dict of LabelEncoders
    """
    if path is None:
        path = LAP_TIME_MODEL_PATH
    
    with open(path, 'rb') as f:
        package = pickle.load(f)
    
    return (package['model'], 
            package['feature_cols'], 
            package['target_col'],
            package['encoders'])
