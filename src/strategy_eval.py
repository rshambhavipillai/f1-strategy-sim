"""
Strategy evaluation module.
Runs simulations and aggregates strategy metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from state import RaceState, Strategy
from simulator import run_single_simulation


def _run_simulation_worker(args):
    """
    Worker function for parallel simulation execution.
    
    Args:
        args: tuple of (race_state, strategy, model, feature_cols, encoders)
    
    Returns:
        simulation result dict
    """
    race_state, strategy, model, feature_cols, encoders = args
    return run_single_simulation(race_state, strategy, model, feature_cols, encoders, randomness=True)


def evaluate_strategies(race_state, strategies, model, feature_cols, encoders, n_sim=200, parallel=True):
    """
    Evaluate multiple strategies through Monte Carlo simulation.
    
    Args:
        race_state: RaceState object
        strategies: list of Strategy objects
        model: trained lap time model
        feature_cols: feature column names
        encoders: dict of LabelEncoders
        n_sim: number of simulations per strategy
        parallel: whether to use parallel processing (default: True)
    
    Returns:
        list of dicts with aggregated metrics per strategy
    """
    results = []
    
    for strategy in strategies:
        print(f"Evaluating strategy: {strategy.name}")
        
        sim_results = []
        use_parallel = parallel and n_sim > 10
        
        if use_parallel:
            # Try parallel processing with error handling
            max_workers = min(4, n_sim // 5)  # Don't spawn too many processes
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    futures = [
                        executor.submit(_run_simulation_worker, 
                                       (race_state, strategy, model, feature_cols, encoders))
                        for _ in range(n_sim)
                    ]
                    # Collect results as they complete
                    for future in as_completed(futures):
                        result = future.result()
                        sim_results.append(result)
            except Exception as e:
                # Fall back to sequential execution if parallel fails
                print(f"Parallel execution failed ({e}). Falling back to sequential execution...")
                sim_results = []
                for i in range(n_sim):
                    result = run_single_simulation(
                        race_state, 
                        strategy, 
                        model, 
                        feature_cols,
                        encoders,
                        randomness=True
                    )
                    sim_results.append(result)
        else:
            # Sequential execution for small n_sim or parallel disabled
            for i in range(n_sim):
                result = run_single_simulation(
                    race_state, 
                    strategy, 
                    model, 
                    feature_cols,
                    encoders,
                    randomness=True
                )
                sim_results.append(result)
        
        # Aggregate metrics
        positions = [r['finishing_position'] for r in sim_results]
        race_times = [r['race_time_s'] for r in sim_results]
        
        mean_position = np.mean(positions)
        std_position = np.std(positions)
        
        podium_count = sum(1 for p in positions if p <= 3)
        podium_prob = podium_count / n_sim
        
        win_count = sum(1 for p in positions if p == 1)
        win_prob = win_count / n_sim
        
        mean_race_time = np.mean(race_times)
        std_race_time = np.std(race_times)
        
        summary = {
            "strategy_name": strategy.name,
            "expected_position": mean_position,
            "position_std": std_position,
            "podium_probability": podium_prob,
            "win_probability": win_prob,
            "mean_race_time_s": mean_race_time,
            "std_race_time_s": std_race_time,
            "n_simulations": n_sim,
            "position_distribution": positions,  # NEW: for detailed analysis
        }
        
        results.append(summary)
        print(f"  Expected position: {mean_position:.2f} +/- {std_position:.2f}")
        print(f"  Podium prob: {podium_prob:.2%}, Win prob: {win_prob:.2%}")
    
    # Sort by expected finishing position
    results_sorted = sorted(results, key=lambda x: x['expected_position'])
    
    return results_sorted
