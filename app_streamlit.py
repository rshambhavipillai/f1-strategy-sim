"""
Streamlit dashboard for F1 Strategy Simulator.
Interactive web app to evaluate pit stop strategies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from model_train import load_model
from state import RaceState, DriverState, Strategy
from strategy_eval import evaluate_strategies
from utils import format_strategy_table
from config import (
    LAP_TIME_MODEL_PATH,
    COMPOUNDS,
    DEFAULT_TRACK,
    DEFAULT_TOTAL_LAPS,
    DEFAULT_NUM_SIMULATIONS,
)

# Page config
st.set_page_config(
    page_title="F1 Strategy Simulator",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ F1 Strategy Simulator")
st.markdown(
    "Evaluate pit stop strategies using Monte Carlo simulation and machine learning predictions."
)

# ===== SIDEBAR: Input Controls =====
with st.sidebar:
    st.header("Race Setup")
    
    # Track selection
    track = st.selectbox(
        "Select Track",
        ["Bahrain", "Saudi Arabia", "Australia", "Japan", "Miami", "Monza"],
        index=0
    )
    
    # Total laps
    total_laps = st.slider(
        "Total Laps",
        min_value=20,
        max_value=80,
        value=DEFAULT_TOTAL_LAPS,
        step=1
    )
    
    st.divider()
    st.header("Current Race State")
    
    # Current lap
    current_lap = st.slider(
        "Current Lap",
        min_value=1,
        max_value=total_laps,
        value=max(1, total_laps // 2),
        step=1
    )
    
    st.divider()
    st.header("Your Car (Player)")
    
    # Player position
    player_position = st.slider(
        "Current Position",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )
    
    # Gap to leader
    gap_to_leader = st.slider(
        "Gap to Leader (seconds)",
        min_value=0.0,
        max_value=60.0,
        value=15.0,
        step=0.5
    )
    
    # Tyre compound
    player_compound = st.selectbox(
        "Current Tyre Compound",
        COMPOUNDS,
        index=0
    )
    
    # Tyre age
    player_tyre_age = st.slider(
        "Tyre Age (laps)",
        min_value=0,
        max_value=60,
        value=5,
        step=1
    )
    
    st.divider()
    st.header("Simulation Settings")
    
    n_simulations = st.slider(
        "Number of Simulations",
        min_value=50,
        max_value=1000,
        value=DEFAULT_NUM_SIMULATIONS,
        step=50
    )

# ===== MAIN CONTENT =====

# Try to load model
try:
    model, feature_cols, target_col, encoders = load_model(LAP_TIME_MODEL_PATH)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error(
        f"❌ Lap time model not found at {LAP_TIME_MODEL_PATH}\n\n"
        "Please train a model first using model_train.py"
    )

if model_loaded:
    # Build race state
    player_driver = DriverState(
        driver_id="Player",
        position=player_position,
        gap_to_leader_s=gap_to_leader,
        compound=player_compound,
        tyre_age_laps=player_tyre_age,
        pace_offset_s=0.0,
    )
    
    # Dummy rival drivers
    rival_drivers = {
        "Rival1": DriverState(
            driver_id="Rival1",
            position=1,
            gap_to_leader_s=0.0,
            compound="medium",
            tyre_age_laps=8,
            pace_offset_s=-0.2,  # Little faster
        ),
        "Rival2": DriverState(
            driver_id="Rival2",
            position=3,
            gap_to_leader_s=5.0,
            compound="soft",
            tyre_age_laps=3,
            pace_offset_s=0.1,
        ),
        "Rival3": DriverState(
            driver_id="Rival3",
            position=4,
            gap_to_leader_s=7.5,
            compound="hard",
            tyre_age_laps=15,
            pace_offset_s=0.3,
        ),
    }
    
    all_drivers = {"Player": player_driver, **rival_drivers}
    
    race_state = RaceState(
        track_id=track,
        current_lap=current_lap,
        total_laps=total_laps,
        drivers=all_drivers,
    )
    
    # ===== STRATEGY DEFINITION =====
    st.header("Strategy Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predefined Strategies")
        
        # Strategy 1: One stop mid-race
        pit_lap_1 = round(total_laps * 0.5)
        
        # Strategy 2: Two-stop early+late
        pit_lap_2a = round(total_laps * 0.3)
        pit_lap_2b = round(total_laps * 0.7)
        
        # Strategy 3: Early pit
        pit_lap_3 = round(total_laps * 0.25)
    
    with col2:
        st.subheader("Custom Strategy")
        custom_pit_input = st.text_input(
            "Enter pit laps (comma-separated)",
            placeholder="e.g., 25,50",
            help="Leave blank to use predefined strategies only"
        )
    
    # Parse custom strategy
    custom_pit_laps = None
    if custom_pit_input.strip():
        try:
            custom_pit_laps = [
                int(x.strip()) for x in custom_pit_input.split(",")
            ]
        except ValueError:
            st.warning("Invalid custom pit laps format")
            custom_pit_laps = None
    
    # Create strategy list
    strategies = [
        Strategy(
            name=f"1-Stop (Lap {pit_lap_1})",
            pit_laps=[pit_lap_1],
            tyre_plan=["hard"]
        ),
        Strategy(
            name=f"2-Stop Early+Late (Laps {pit_lap_2a},{pit_lap_2b})",
            pit_laps=[pit_lap_2a, pit_lap_2b],
            tyre_plan=["medium", "hard"]
        ),
        Strategy(
            name=f"Early Pit (Lap {pit_lap_3})",
            pit_laps=[pit_lap_3],
            tyre_plan=["hard"]
        ),
    ]
    
    if custom_pit_laps:
        strategies.append(
            Strategy(
                name="Custom Strategy",
                pit_laps=custom_pit_laps,
                tyre_plan=["hard"] * len(custom_pit_laps)
            )
        )
    
    # ===== RUN SIMULATION =====
    st.divider()
    
    col_run, col_info = st.columns([1, 2])
    
    with col_run:
        run_button = st.button(
            "🚀 Run Simulation",
            type="primary",
            use_container_width=True
        )
    
    with col_info:
        st.info(
            f"Running {n_simulations} simulations per strategy. "
            f"This may take a minute..."
        )
    
    if run_button:
        with st.spinner("Running simulations..."):
            try:
                results = evaluate_strategies(
                    race_state,
                    strategies,
                    model,
                    feature_cols,
                    encoders,
                    n_sim=n_simulations,
                    parallel=False
                )
                
                # Display results
                st.success("Simulation complete!")
                
                st.divider()
                st.header("Results")
                
                # Results table
                st.subheader("Strategy Comparison")
                df_results = pd.DataFrame(results)
                
                # Format for display
                display_df = df_results[[
                    'strategy_name',
                    'expected_position',
                    'position_std',
                    'podium_probability',
                    'win_probability',
                    'mean_race_time_s'
                ]].copy()
                
                display_df.columns = [
                    'Strategy',
                    'Expected Pos',
                    'Std Dev',
                    'Podium %',
                    'Win %',
                    'Race Time'
                ]
                
                st.dataframe(display_df, use_container_width=True)
                
                # Charts
                st.divider()
                st.subheader("📊 Strategy Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Expected Finishing Position")
                    chart_data = df_results[['strategy_name', 'expected_position']].set_index('strategy_name')
                    st.bar_chart(chart_data, color='#1f77b4')
                
                with col2:
                    st.subheader("Podium Probability")
                    chart_data = df_results[['strategy_name', 'podium_probability']].set_index('strategy_name')
                    st.bar_chart(chart_data, color='#2ca02c')
                
                # NEW: Risk-Reward scatter plot
                st.subheader("Risk vs. Reward")
                scatter_data = df_results[['strategy_name', 'expected_position', 'podium_probability']].copy()
                scatter_data.columns = ['Strategy', 'Expected Position', 'Podium Probability']
                
                chart = (
                    alt.Chart(scatter_data)
                    .mark_circle(size=200, opacity=0.8)
                    .encode(
                        x=alt.X('Expected Position', scale=alt.Scale(domain=[1, 20])),
                        y=alt.Y('Podium Probability', scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color('Strategy', legend=alt.Legend(title='Strategy')),
                        tooltip=['Strategy', 'Expected Position', 'Podium Probability']
                    )
                    .properties(width=600, height=400)
                )
                st.altair_chart(chart, use_container_width=True)
                
                # NEW: Position distribution histogram
                st.subheader("Finishing Position Distribution")
                dist_data = []
                for result in results:
                    for pos in result['position_distribution']:
                        dist_data.append({
                            'strategy': result['strategy_name'],
                            'finishing_position': int(pos)
                        })
                
                df_dist = pd.DataFrame(dist_data)
                
                hist_chart = (
                    alt.Chart(df_dist)
                    .mark_bar(opacity=0.7)
                    .encode(
                        x='finishing_position',
                        y='count()',
                        color='strategy',
                        xOffset='strategy'
                    )
                    .properties(width=600, height=300)
                )
                st.altair_chart(hist_chart, use_container_width=True)
                
                # NEW: Win/Podium probabilities comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Win Probability")
                    win_data = df_results[['strategy_name', 'win_probability']].set_index('strategy_name')
                    st.bar_chart(win_data, color='#ff7f0e')
                
                with col2:
                    st.subheader("Race Time Consistency")
                    time_data = df_results[['strategy_name', 'std_race_time_s']].set_index('strategy_name')
                    st.bar_chart(time_data, color='#d62728')
                
                # Best strategy
                best_strategy = results[0]
                st.divider()
                st.success(
                    f"🏆 **Best Strategy: {best_strategy['strategy_name']}**\n\n"
                    f"Expected Finishing Position: {best_strategy['expected_position']:.2f}\n\n"
                    f"Podium Probability: {best_strategy['podium_probability']:.1%}\n\n"
                    f"Win Probability: {best_strategy['win_probability']:.1%}"
                )
            
            except Exception as e:
                st.error(f"Error during simulation: {e}")
                st.exception(e)

st.divider()
st.markdown(
    "---\n"
    "**F1 Strategy Simulator v0.1** | "
    "[GitHub](https://github.com) | "
    "Data: FastF1"
)
