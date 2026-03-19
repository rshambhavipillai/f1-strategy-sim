# F1 Strategy Simulator

A comprehensive Formula 1 race strategy simulation and optimization tool using machine learning and Monte Carlo analysis.

## Features

- **Lap Time Prediction**: Machine learning model trained on FastF1 historical data to predict lap times
- **Strategy Evaluation**: Monte Carlo simulation to evaluate pit stop strategies
- **Interactive Dashboard**: Streamlit web app for easy strategy comparison
- **Customizable Simulations**: Configure race conditions, driver states, and custom pit strategies

## Project Structure

```
f1-strategy-sim/
├── data/                      # Cached lap data
├── models/                    # Trained model files
├── notebooks/                 # Jupyter notebooks for EDA
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration and constants
│   ├── data_prep.py          # Load and preprocess FastF1 data
│   ├── model_train.py        # Train lap time prediction model
│   ├── state.py              # RaceState, DriverState, Strategy classes
│   ├── simulator.py          # Monte Carlo simulation engine
│   ├── strategy_eval.py      # Strategy evaluation wrapper
│   └── utils.py              # Utility functions
├── app_streamlit.py          # Streamlit dashboard
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd f1-strategy-sim
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train the Lap Time Model

First, you need to train the machine learning model on historical F1 data:

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from data_prep import prepare_dataset
from model_train import train_lap_time_model, save_model
from config import TRAINING_SEASONS

# Load and prepare data
df_features, feature_cols, target_col = prepare_dataset(TRAINING_SEASONS)

# Train model
model, feature_cols, target_col, encoders = train_lap_time_model(
    df_features, feature_cols, target_col
)

# Save model
save_model(model, feature_cols, target_col, encoders)
"
```

Alternatively, create a simple training script or notebook in `notebooks/`.

### 2. Run the Streamlit Dashboard

```bash
streamlit run app_streamlit.py
```

The app will open in your browser at `http://localhost:8501/`.

### 3. Evaluate Strategies

Use the interactive dashboard to:
- Select a track and race configuration
- Set your car's current state (position, gap, tyres)
- Define pit stop strategies (predefined or custom)
- Run simulations and view results

## Module Overview

### `config.py`
Central configuration file containing:
- File paths (data, models)
- Training parameters (seasons, tracks)
- Simulation constants (pit stop delta, tyre degradation)

### `data_prep.py`
Data loading and feature engineering:
- `load_lap_data()`: Fetch FastF1 lap data
- `build_lap_features()`: Engineer lap-level features
- `prepare_dataset()`: Complete pipeline with caching

### `model_train.py`
Model training and persistence:
- `train_lap_time_model()`: Train GradientBoosting or RandomForest regressor
- `save_model()`: Pickle model and metadata
- `load_model()`: Load trained model for inference

### `state.py`
Dataclasses for simulation:
- `DriverState`: Driver position, tyres, pace
- `RaceState`: Track, lap, all drivers
- `Strategy`: Pit schedule and tyre plan

### `simulator.py`
Core simulation logic:
- `predict_lap_time()`: Use model to predict lap time
- `run_single_simulation()`: Execute one race simulation
- `update_positions()`: Update driver standings based on lap times

### `strategy_eval.py`
Strategy evaluation:
- `evaluate_strategies()`: Run N simulations per strategy
- Aggregate metrics: expected position, podium/win probability

### `app_streamlit.py`
Interactive web dashboard:
- Sidebar for race configuration
- Strategy definition (predefined + custom)
- Results visualization and comparison

## Usage Examples

### Programmatic Usage

```python
from src.model_train import load_model
from src.state import RaceState, DriverState, Strategy
from src.strategy_eval import evaluate_strategies

# Load the trained model
model, feature_cols, target_col, encoders = load_model()

# Define race state
race_state = RaceState(
    track_id="Monza",
    current_lap=30,
    total_laps=53,
    drivers={
        "Player": DriverState(
            driver_id="Player",
            position=3,
            gap_to_leader_s=2.5,
            compound="soft",
            tyre_age_laps=8,
        ),
        # ... other drivers
    }
)

# Define strategies
strategies = [
    Strategy(
        name="One Stop (Lap 30)",
        pit_laps=[30],
        tyre_plan=["hard"]
    ),
    Strategy(
        name="Two Stop (Laps 20, 40)",
        pit_laps=[20, 40],
        tyre_plan=["medium", "hard"]
    ),
]

# Evaluate
results = evaluate_strategies(
    race_state, strategies, model, feature_cols, encoders, n_sim=500
)

for result in results:
    print(f"{result['strategy_name']}: Position {result['expected_position']:.2f}")
```

## Development

### Adding New Features

1. **Custom Tyre Models**: Modify `TYRE_DEGRADATION` in `config.py`
2. **Safety Car Logic**: Extend `simulator.py` with track-specific rules
3. **Advanced Features**: Add fuel load, weather, setup changes in `state.py`

### Testing

Create a test script or notebook to validate:
- Lap time predictions
- Simulation logic
- Strategy rankings consistency

## Configuration

Key parameters in `src/config.py`:

```python
# Training data
TRAINING_SEASONS = [2022, 2023, 2024]
TRAINING_TRACKS = [...]

# Simulation
DEFAULT_NUM_DRIVERS = 4
DEFAULT_TOTAL_LAPS = 57
PIT_STOP_DELTA = 21.0  # seconds
SIMULATION_NOISE_STD = 0.5  # seconds

# Tyre degradation (seconds per lap)
TYRE_DEGRADATION = {
    "soft": 0.08,
    "medium": 0.05,
    "hard": 0.03,
}
```

## Limitations & Future Work

- **v1 Alpha**: Simplified 4-car grid, no safety car/VSC
- **Tyre Model**: Basic degradation; could use more sophisticated models
- **Driver Performance**: Constant pace offset; could add lap-by-lap variation
- **Track-Specific Logic**: Pit strategy logic is generic; could optimize per track
- **Real-Time Data**: Could integrate live race data feeds

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- fastf1 (formula 1 data)
- streamlit (web app)
- Optional: xgboost, lightgbm (advanced models)

## License

[Your License Here]

## Contributing

Contributions welcome! Please fork and submit pull requests.

## Acknowledgments

- **FastF1**: For Formula 1 data and telemetry
- **Streamlit**: For interactive dashboard framework
- **scikit-learn**: For machine learning models

---

**Author**: Shambhavi Pillai  
**Last Updated**: 2026
