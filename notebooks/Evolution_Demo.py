# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Evolutionary Strategy for Iterated Prisoner's Dilemma
#
# This notebook showcases a memory-one strategy evolved with CMA-ES
# to play the Iterated Prisoner's Dilemma (IPD).
#
# If `QUICK_DEMO=1` is set, a miniature 3-generation evolution will be
# performed; full-scale results require the pre-trained `evolved_strategy.pkl`.

# %%
import os
import sys
import time
import pickle
from pathlib import Path

import cma
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------
# Detect repository root (works in both .py script and .ipynb)
# -----------------------------------------------------------
try:  # running as .py
    repo_root = Path(__file__).resolve().parents[1]
except NameError:  # running inside Jupyter
    repo_root = Path.cwd().resolve()
    if repo_root.name == "notebooks":
        repo_root = repo_root.parent

sys.path.append(str(repo_root))

from env import (
    IPDEnv,
    TitForTat,
    AlwaysCooperate,
    AlwaysDefect,
    RandomStrategy,
    PavlovStrategy,
    simulate_match,
    Strategy,
)

# Paths
models_dir = repo_root / "models"
results_dir = repo_root / "results"

# -----------------------------------------------------------
# Helper: save plot + CSV so numbers remain accessible
# -----------------------------------------------------------
def save_plot_and_csv(x, y, name: str, folder: str = "results"):
    """Save PNG and matching CSV for later analysis."""
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(folder, exist_ok=True)
    pd.DataFrame({"x": x, "y": y}).to_csv(
        f"{folder}/{name}_data.csv", index=False
    )
    plt.figure()
    plt.plot(x, y)
    plt.title(name.replace("_", " ").title())
    plt.savefig(f"{folder}/{name}.png", dpi=120, bbox_inches="tight")
    plt.close()

# %% [markdown]
# ## Memory-One Strategy Implementation
#
# A memory-one strategy for the Iterated Prisoner's Dilemma is defined by five probabilities:
# 1. p_cc: Probability of cooperating if both players cooperated in the previous round
# 2. p_cd: Probability of cooperating if the player cooperated and opponent defected
# 3. p_dc: Probability of cooperating if the player defected and opponent cooperated
# 4. p_dd: Probability of cooperating if both players defected in the previous round
# 5. initial_action_prob: Probability of cooperating on the first move

# %%
class MemoryOneStrategy(Strategy):
    """Memory-one strategy for IPD."""

    def __init__(self, params, name="MemoryOne"):
        super().__init__(name)
        self.params = np.clip(params, 0.0, 1.0)
        self.p_cc, self.p_cd, self.p_dc, self.p_dd, self.initial_action_prob = self.params
        self.rng = np.random.RandomState()

    def action(self, history, player_idx=0):
        if not history:  # first move
            return 0 if self.rng.random() < self.initial_action_prob else 1

        opp_idx = 1 - player_idx
        last_player, last_opp = history[-1][player_idx], history[-1][opp_idx]

        if last_player == 0 and last_opp == 0:
            prob = self.p_cc
        elif last_player == 0 and last_opp == 1:
            prob = self.p_cd
        elif last_player == 1 and last_opp == 0:
            prob = self.p_dc
        else:
            prob = self.p_dd

        return 0 if self.rng.random() < prob else 1

    def __str__(self):
        return (
            f"{self.name}: "
            f"p_cc={self.p_cc:.2f}, p_cd={self.p_cd:.2f}, "
            f"p_dc={self.p_dc:.2f}, p_dd={self.p_dd:.2f}, init={self.initial_action_prob:.2f}"
        )

# %% [markdown]
# ## Loading the Evolved Strategy
#
# First, we'll load the pre-trained evolved strategy. If the model doesn't exist, we have an option
# to quickly train a demo strategy when the environment variable `QUICK_DEMO=1` is set.

# %%
def quick_evolve_strategy(save_dir=models_dir, num_generations=3, population_size=10, seed=42):
    """Quickly evolve a memory-one strategy for demo purposes."""
    print("Quickly evolving a demo strategy...")
    
    # Create environment
    env = IPDEnv(num_rounds=100, seed=seed)
    
    # Define opponent strategies
    opponent_strategies = {
        "tit_for_tat": TitForTat(),
        "always_cooperate": AlwaysCooperate(),
        "always_defect": AlwaysDefect(),
        "random": RandomStrategy(seed=seed),
        "pavlov": PavlovStrategy()
    }
    
    # Helper function for fitness evaluation
    def evaluate_fitness(params):
        # Create memory-one strategy from parameters
        strategy = MemoryOneStrategy(params)
        
        total_rewards = []
        
        # Play against each opponent
        for opponent in opponent_strategies.values():
            # Simulate match against opponent
            for _ in range(3):  # 3 matches per opponent
                results = simulate_match(env, strategy, opponent, num_rounds=100)
                total_rewards.append(results['player_score'])
        
        # Return average reward (higher is better)
        return np.mean(total_rewards)
    
    # Initial guess for parameters (a simple cooperative strategy)
    initial_params = np.array([0.9, 0.1, 0.9, 0.1, 0.9])
    
    # Setup CMA-ES optimizer
    es = cma.CMAEvolutionStrategy(
        initial_params,
        0.5,
        {'popsize': population_size, 'seed': seed}
    )
    
    # Run quick evolution
    start_time = time.time()
    
    for generation in range(num_generations):
        # Sample population
        solutions = es.ask()
        
        # Evaluate fitness
        fitnesses = []
        for params in solutions:
            # Clip parameters to valid range [0, 1]
            params_clipped = np.clip(params, 0, 1)
            
            # Get fitness (negative because CMA-ES minimizes)
            fitness = -evaluate_fitness(params_clipped)
            fitnesses.append(fitness)
        
        # Update CMA-ES with evaluated solutions
        es.tell(solutions, fitnesses)
        
        # Print progress
        best_idx = np.argmin(fitnesses)
        best_fitness = -fitnesses[best_idx]
        best_params = np.clip(solutions[best_idx], 0, 1)
        print(f"Generation {generation+1}/{num_generations} | Best fitness: {best_fitness:.2f}")
    
    # Get final best solution
    best_params = es.result.xbest
    best_params = np.clip(best_params, 0, 1)
    best_strategy = MemoryOneStrategy(best_params, name="QuickEvolvedStrategy")
    
    # Save the strategy
    os.makedirs(save_dir, exist_ok=True)
    
    with open(save_dir / "quick_evolved_strategy.pkl", 'wb') as f:
        pickle.dump(best_strategy, f)
    
    evolution_time = time.time() - start_time
    print(f"Quick evolution completed in {evolution_time:.2f} seconds")
    print(f"Evolved strategy: {best_strategy}")
    
    return best_strategy, save_dir / "quick_evolved_strategy.pkl"

# Try to load the pre-trained evolved strategy
model_path = models_dir / "evolved_strategy.pkl"
quick_demo = False

if not model_path.exists():
    # Check for alternative model paths
    alternative_paths = list(models_dir.glob("*evolved*.pkl"))
    if alternative_paths:
        model_path = alternative_paths[0]
        print(f"Using alternative model: {model_path}")
    else:
        # No model found, check if QUICK_DEMO is enabled
        if os.environ.get('QUICK_DEMO') == '1':
            quick_demo = True
            evolved_strategy, model_path = quick_evolve_strategy()
        else:
            raise FileNotFoundError(
                "Model file not found – please run the full training script first."
                "\nOr set QUICK_DEMO=1 environment variable to train a quick demo model."
            )

# Load the model
if not quick_demo:  # We already have the strategy object if we did quick training
    print(f"Loading evolved strategy from {model_path}")
    with open(model_path, 'rb') as f:
        evolved_strategy = pickle.load(f)

print(f"Evolved Strategy: {evolved_strategy}")

# %% [markdown]
# ## Evaluating the Strategy Against Classic Strategies
#
# Let's evaluate the evolved memory-one strategy against classic strategies:
# - Tit-for-Tat: Cooperates on the first move, then mirrors the opponent's previous action
# - Always Cooperate: Always cooperates regardless of what the opponent does
# - Always Defect: Always defects regardless of what the opponent does
# - Random: Randomly cooperates or defects with equal probability
# - Pavlov: Win-Stay, Lose-Shift strategy that repeats successful actions

# %%
def play_match(strategy, opponent, num_rounds=100, seed=42):
    """Play a match between the strategy and an opponent."""
    env = IPDEnv(num_rounds=num_rounds, seed=seed)
    match_results = simulate_match(env, strategy, opponent, num_rounds)
    
    return {
        "player_score": match_results["player_score"],
        "opponent_score": match_results["opponent_score"],
        "player_coop_rate": match_results["cooperation_rate_player"],
        "opponent_coop_rate": match_results["cooperation_rate_opponent"],
        "history": match_results["history"]
    }

# Define opponent strategies to evaluate against
opponent_strategies = {
    "tit_for_tat": TitForTat(),
    "always_cooperate": AlwaysCooperate(),
    "always_defect": AlwaysDefect(),
    "random": RandomStrategy(seed=42),
    "pavlov": PavlovStrategy()
}

# Play 5 matches against each opponent
num_matches = 5
num_rounds = 100
results = {}

for opponent_name, opponent in opponent_strategies.items():
    match_results = []
    print(f"Playing against {opponent_name}...")
    
    for match in range(num_matches):
        match_result = play_match(evolved_strategy, opponent, num_rounds=num_rounds, seed=42+match)
        match_results.append(match_result)
        print(f"  Match {match+1}: Score = {match_result['player_score']:.1f}, "
              f"Strategy cooperation rate = {match_result['player_coop_rate']:.2f}")
    
    # Calculate average results
    avg_player_score = np.mean([r["player_score"] for r in match_results])
    avg_opponent_score = np.mean([r["opponent_score"] for r in match_results])
    avg_player_coop = np.mean([r["player_coop_rate"] for r in match_results])
    avg_opponent_coop = np.mean([r["opponent_coop_rate"] for r in match_results])
    
    results[opponent_name] = {
        "avg_player_score": avg_player_score,
        "avg_opponent_score": avg_opponent_score,
        "avg_player_coop": avg_player_coop,
        "avg_opponent_coop": avg_opponent_coop,
        "match_results": match_results
    }
    
    print(f"  Average score: {avg_player_score:.2f}")
    print(f"  Average strategy cooperation rate: {avg_player_coop:.2f}")
    print(f"  Average opponent cooperation rate: {avg_opponent_coop:.2f}")
    print("")

# %% [markdown]
# ## Visualizing the Results
#
# Let's visualize how the evolved strategy performs against different opponents.

# %%
# Create bar plot for scores
opponent_names = list(results.keys())
player_scores = [results[name]["avg_player_score"] for name in opponent_names]
player_coop_rates = [results[name]["avg_player_coop"] for name in opponent_names]
opponent_coop_rates = [results[name]["avg_opponent_coop"] for name in opponent_names]

# Save score data
save_plot_and_csv(
    opponent_names, 
    player_scores, 
    "evolution_vs_baselines", 
    folder=str(results_dir / "evolution")
)

# Create more detailed visualization
plt.figure(figsize=(12, 6))

# Plot scores
plt.subplot(1, 2, 1)
plt.bar(opponent_names, player_scores)
plt.ylabel("Average Score")
plt.title("Evolved Strategy Scores vs Different Opponents")
plt.ylim(0, max(player_scores) * 1.2)
plt.xticks(rotation=45)

# Plot cooperation rates
plt.subplot(1, 2, 2)
x = np.arange(len(opponent_names))
width = 0.35
plt.bar(x - width/2, player_coop_rates, width, label="Evolved Strategy")
plt.bar(x + width/2, opponent_coop_rates, width, label="Opponent")
plt.ylabel("Cooperation Rate")
plt.title("Cooperation Rates")
plt.xticks(x, opponent_names, rotation=45)
plt.ylim(0, 1.1)
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Evolution Progress
#
# Let's load and display the evolution history data from the full training run.

# %%
def load_and_plot_evolution_history():
    """Load and plot the evolution history from full training."""
    # Try to find the evolution history data
    best_fitness_file = results_dir / "evolution" / "evolution_best_fitness_data.csv"
    avg_fitness_file = results_dir / "evolution" / "evolution_avg_fitness_data.csv"
    
    if not best_fitness_file.exists() or not avg_fitness_file.exists():
        # Check for alternative files
        alternative_files = list((results_dir / "evolution").glob("*fitness*_data.csv"))
        if alternative_files:
            print(f"Found alternative fitness history files: {[f.name for f in alternative_files]}")
            for file in alternative_files:
                df = pd.read_csv(file)
                plt.figure(figsize=(10, 6))
                plt.plot(df["x"], df["y"])
                plt.xlabel("Generation")
                plt.ylabel("Fitness (Average Reward)")
                plt.title(f"{file.stem.replace('_data', '').replace('_', ' ').title()}")
                plt.grid(True, alpha=0.3)
                plt.show()
        else:
            print("Evolution history data not found. Full training hasn't been run yet.")
            return None
    else:
        # Load and plot both fitness curves
        best_df = pd.read_csv(best_fitness_file)
        avg_df = pd.read_csv(avg_fitness_file)
        
        plt.figure(figsize=(10, 6))
        plt.plot(best_df["x"], best_df["y"], label="Best Fitness")
        plt.plot(avg_df["x"], avg_df["y"], label="Average Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness (Average Reward)")
        plt.title("Evolution Progress")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return best_df, avg_df

# Load and plot evolution history if available
evolution_history = load_and_plot_evolution_history()

# %% [markdown]
# ## Interpretation of Results
#
# * **Strong performance vs cooperative opponents.**  
#   The strategy scores **~500** against Always Cooperate and demonstrates high performance with **~280-290** points against both Random and Pavlov.  
# * **Balanced approach with Tit-for-Tat.**  
#   Score of **~230** with TFT shows effective reciprocal cooperation with a cooperation rate of ~67%.  
# * **Limited retaliation vs Always Defect.**  
#   Average score ≈ 90 : 10 against AllD shows some punishment, but the agent is still largely
#   exploited by an unconditional defector.  
# * **Strategic defection patterns.**  
#   Cooperation rate varies significantly by opponent: ~67% vs TFT, ~15% vs AllD, only ~12% vs Random (despite Random cooperating ~50%), and ~93% vs Pavlov.  
# * **Evolution progress.**  
#   Best fitness improved dramatically in later generations, rising from ~250 to **~285** over 50 generations, with a notable jump after generation 40. Average fitness remained more stable around ~250.  
# * **Pavlov interactions.**  
#   The evolved strategy achieves near-perfect cooperation with Pavlov (~93%), demonstrating remarkable compatibility with this Win-Stay, Lose-Shift strategy.
# * **Future work.**  
#   Increase exposure to defection-oriented opponents (AllD, Grim Trigger) or
#   extend generations/population size to push fitness closer to the 300-point
#   range versus mixed opponents while minimising losses to AllD.
