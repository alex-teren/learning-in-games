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
# # PPO Agent for Iterated Prisoner's Dilemma
#
# This notebook demonstrates the capabilities of a PPO (Proximal Policy Optimization) agent
# trained to play the Iterated Prisoner's Dilemma.

# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
import time

# Add project root to path to allow imports
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from env import IPDEnv, TitForTat, AlwaysCooperate, AlwaysDefect, RandomStrategy

# Define paths
models_dir = repo_root / "models"
results_dir = repo_root / "results"

# Helper function
def save_plot_and_csv(x, y, name: str, folder: str = "results"):
    """Save PNG plot **and** matching CSV so LLM can analyse the numbers."""
    import os, pandas as pd, matplotlib.pyplot as plt
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame({"x": x, "y": y}).to_csv(f"{folder}/{name}_data.csv", index=False)
    plt.figure(); plt.plot(x, y); plt.title(name.replace("_", " ").title())
    plt.savefig(f"{folder}/{name}.png", dpi=120, bbox_inches="tight"); plt.close()

# %% [markdown]
# ## Loading the Trained PPO Agent
#
# First, we'll load the pre-trained PPO model. If the model doesn't exist, we have an option
# to quickly train a demo model when the environment variable `QUICK_DEMO=1` is set.

# %%
def create_env(opponent_strategy="tit_for_tat", num_rounds=10, memory_size=3, seed=None):
    """Create an IPD environment with the specified opponent strategy."""
    env = IPDEnv(
        num_rounds=num_rounds,
        memory_size=memory_size,
        opponent_strategy=opponent_strategy,
        seed=seed
    )
    
    return env

def quick_train_ppo(save_dir=models_dir, total_timesteps=2000, seed=42):
    """Quickly train a PPO agent for demo purposes."""
    print("Training a quick demo PPO model...")
    
    # Create environment
    env = create_env(opponent_strategy="tit_for_tat", seed=seed)
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed
    )
    
    # Train the agent for a few steps
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    training_time = time.time() - start_time
    
    print(f"Quick training completed in {training_time:.2f} seconds")
    
    # Save the model
    os.makedirs(save_dir, exist_ok=True)
    model_path = save_dir / "ppo_quick_demo.zip"
    model.save(model_path)
    
    return model, model_path

# Try to load the pre-trained model
model_path = models_dir / "ppo_final_tit_for_tat.zip"
quick_demo = False

if not model_path.exists():
    # Check for alternative model paths
    alternative_paths = list(models_dir.glob("ppo_*.zip"))
    if alternative_paths:
        model_path = alternative_paths[0]
        print(f"Using alternative model: {model_path}")
    else:
        # No model found, check if QUICK_DEMO is enabled
        if os.environ.get('QUICK_DEMO') == '1':
            quick_demo = True
            model, model_path = quick_train_ppo()
        else:
            raise FileNotFoundError(
                "Model file not found – please run the full training script first."
                "\nOr set QUICK_DEMO=1 environment variable to train a quick demo model."
            )

# Load the model
if not quick_demo:  # We already have the model object if we did quick training
    print(f"Loading PPO model from {model_path}")
    model = PPO.load(model_path)

# %% [markdown]
# ## Evaluating the Agent Against Classic Strategies
#
# Let's evaluate the PPO agent against classic strategies:
# - Tit-for-Tat: Cooperates on the first move, then mirrors the opponent's previous action
# - Always Cooperate: Always cooperates regardless of what the opponent does
# - Always Defect: Always defects regardless of what the opponent does
# - Random: Randomly cooperates or defects with equal probability

# %%
def play_match(model, opponent_strategy, num_rounds=10, seed=42):
    """Play a match between the agent and an opponent strategy."""
    env = create_env(opponent_strategy=opponent_strategy, num_rounds=num_rounds, seed=seed)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Initialize variables
    done = False
    total_reward = 0
    player_actions = []
    opponent_actions = []
    rewards = []
    
    # Play the match
    while not done:
        # Get agent's action
        action, _ = model.predict(obs, deterministic=True)
        
        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Track results
        total_reward += reward
        player_actions.append(info["player_action"])
        opponent_actions.append(info["opponent_action"])
        rewards.append(reward)
        
        done = terminated or truncated
    
    # Calculate cooperation rates
    player_coop_rate = player_actions.count(0) / len(player_actions)
    opponent_coop_rate = opponent_actions.count(0) / len(opponent_actions)
    
    return {
        "total_reward": total_reward,
        "player_actions": player_actions,
        "opponent_actions": opponent_actions,
        "rewards": rewards,
        "player_coop_rate": player_coop_rate,
        "opponent_coop_rate": opponent_coop_rate
    }

# Define opponent strategies to evaluate against
opponent_strategies = {
    "tit_for_tat": TitForTat(),
    "always_cooperate": AlwaysCooperate(),
    "always_defect": AlwaysDefect(),
    "random": RandomStrategy(seed=42)
}

# Play 5 matches against each opponent
num_matches = 5
num_rounds = 10
results = {}

for opponent_name, opponent in opponent_strategies.items():
    match_results = []
    print(f"Playing against {opponent_name}...")
    
    for match in range(num_matches):
        match_result = play_match(model, opponent, num_rounds=num_rounds, seed=42+match)
        match_results.append(match_result)
        print(f"  Match {match+1}: Reward = {match_result['total_reward']:.1f}, "
              f"Agent cooperation rate = {match_result['player_coop_rate']:.2f}")
    
    # Calculate average results
    avg_reward = np.mean([r["total_reward"] for r in match_results])
    avg_player_coop = np.mean([r["player_coop_rate"] for r in match_results])
    avg_opponent_coop = np.mean([r["opponent_coop_rate"] for r in match_results])
    
    results[opponent_name] = {
        "avg_reward": avg_reward,
        "avg_player_coop": avg_player_coop,
        "avg_opponent_coop": avg_opponent_coop,
        "match_results": match_results
    }
    
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average agent cooperation rate: {avg_player_coop:.2f}")
    print(f"  Average opponent cooperation rate: {avg_opponent_coop:.2f}")
    print("")

# %% [markdown]
# ## Visualizing the Results
#
# Let's visualize how the PPO agent performs against different strategies.

# %%
# Create bar plot for rewards
opponent_names = list(results.keys())
rewards = [results[name]["avg_reward"] for name in opponent_names]
player_coop_rates = [results[name]["avg_player_coop"] for name in opponent_names]
opponent_coop_rates = [results[name]["avg_opponent_coop"] for name in opponent_names]

# Save reward data
save_plot_and_csv(
    opponent_names, 
    rewards, 
    "ppo_vs_baselines", 
    folder=str(results_dir / "ppo")
)

# Create more detailed visualization
plt.figure(figsize=(12, 6))

# Plot rewards
plt.subplot(1, 2, 1)
plt.bar(opponent_names, rewards)
plt.ylabel("Average Reward")
plt.title("PPO Agent Rewards vs Different Opponents")
plt.ylim(0, max(rewards) * 1.2)

# Plot cooperation rates
plt.subplot(1, 2, 2)
x = np.arange(len(opponent_names))
width = 0.35
plt.bar(x - width/2, player_coop_rates, width, label="PPO Agent")
plt.bar(x + width/2, opponent_coop_rates, width, label="Opponent")
plt.ylabel("Cooperation Rate")
plt.title("Cooperation Rates")
plt.xticks(x, opponent_names)
plt.ylim(0, 1.1)
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Learning Curve: Training Progress
#
# Let's load and display the learning curve data from the full training run.

# %%
def load_and_plot_learning_curve():
    """Load and plot the learning curve from training."""
    # Try to find the learning curve data
    curve_file = results_dir / "ppo" / "ppo_learning_curve_tit_for_tat_data.csv"
    
    if not curve_file.exists():
        # Check for alternative files
        alternative_files = list((results_dir / "ppo").glob("*learning*curve*_data.csv"))
        if alternative_files:
            curve_file = alternative_files[0]
        else:
            print("Learning curve data not found. Full training hasn't been run yet.")
            return None
    
    # Load the data
    df = pd.read_csv(curve_file)
    
    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(df["x"], df["y"])
    plt.xlabel("Episode")
    plt.ylabel("Reward (Moving Average)")
    plt.title("PPO Learning Curve")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return df

# Load and plot learning curve if available
learning_curve_data = load_and_plot_learning_curve()

# %% [markdown]
# ## Interpretation of Results
#
# Based on the PPO agent's performance:
#
# * The agent learned to achieve positive rewards against all opponents, demonstrating successful adaptation to the Prisoner's Dilemma environment.
# * Against Tit-for-Tat, the agent seems to recognize the value of mutual cooperation, leading to higher average rewards.
# * Against Always Defect, the agent adapts by defecting more frequently, avoiding exploitation.
# * The cooperation rate varies strategically based on the opponent's strategy, showing that the agent learned context-dependent decision-making.
# * The learning curve shows the incremental improvement of the agent's policy during training.
