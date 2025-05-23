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
# This notebook demonstrates a PPO (Proximal Policy Optimization) agent
# trained to play the Iterated Prisoner's Dilemma (IPD).
#
# **Note:** if `QUICK_DEMO=1`, a miniature model is trained on-the-fly;
# numerical results will differ from the full 200 k-step run.

# %%
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# %%
# Determine repository root in both .py and Jupyter contexts
try:
    # Running as a .py script
    repo_root = Path(__file__).resolve().parents[1]
except NameError:
    # Running in Jupyter: __file__ is undefined
    repo_root = Path.cwd().resolve()
    if repo_root.name == "notebooks":
        repo_root = repo_root.parent

# Add repo root to import path
sys.path.append(str(repo_root))

from env import IPDEnv, TitForTat, AlwaysCooperate, AlwaysDefect, RandomStrategy, PavlovStrategy

# Paths
models_dir = repo_root / "models"
results_dir = repo_root / "results"

# Helper
def save_plot_and_csv(x, y, name: str, folder: str = "results"):
    """Save PNG plot and matching CSV"""
    import os, pandas as pd, matplotlib.pyplot as plt

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
# ## Loading the Trained PPO Agent
#
# We first try to load the full-scale model (`ppo_final_*.zip`).
# If no model is found **and** the environment variable `QUICK_DEMO=1`
# is set, a tiny 2 k-step agent is trained for illustration.

# %%
def create_env(opponent_strategy="tit_for_tat", num_rounds=10, memory_size=3, seed=None):
    return IPDEnv(
        num_rounds=num_rounds,
        memory_size=memory_size,
        opponent_strategy=opponent_strategy,
        seed=seed,
    )

def quick_train_ppo(save_dir=models_dir, total_timesteps=2_000, seed=42):
    print("Training a quick demo PPO model...")
    env = create_env(opponent_strategy="tit_for_tat", seed=seed)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed)
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps)
    print(f"Quick training finished in {time.time() - t0:.1f}s")
    os.makedirs(save_dir, exist_ok=True)
    path = save_dir / "ppo_quick_demo.zip"
    model.save(path)
    return model, path

model_path = models_dir / "ppo_final_tit_for_tat.zip"
quick_demo = False

if not model_path.exists():
    alt = list(models_dir.glob("ppo_*.zip"))
    if alt:
        model_path = alt[0]
        print(f"Using alternative model: {model_path.name}")
    elif os.environ.get("QUICK_DEMO") == "1":
        quick_demo = True
        model, model_path = quick_train_ppo()
    else:
        raise FileNotFoundError(
            "No PPO model found. Run full training or set QUICK_DEMO=1."
        )

if not quick_demo:
    print(f"Loading PPO model from {model_path.name}")
    model = PPO.load(model_path)

# %% [markdown]
# ## Evaluating the Agent Against Classic Strategies

# %%
def play_match(model, opponent_strategy, num_rounds=10, seed=42):
    env = create_env(opponent_strategy=opponent_strategy, num_rounds=num_rounds, seed=seed)
    obs, _ = env.reset()
    done = False
    total_reward, player_actions, opponent_actions = 0, [], []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        player_actions.append(info["player_action"])
        opponent_actions.append(info["opponent_action"])
        done = term or trunc

    player_coop = player_actions.count(0) / len(player_actions)
    opp_coop = opponent_actions.count(0) / len(opponent_actions)
    return total_reward, player_coop, opp_coop

opponents = {
    "tit_for_tat": TitForTat(),
    "always_cooperate": AlwaysCooperate(),
    "always_defect": AlwaysDefect(),
    "random": RandomStrategy(seed=42),
    "pavlov": PavlovStrategy(),
}

stats = {}
for name, strat in opponents.items():
    rewards, pc, oc = [], [], []
    for i in range(5):
        r, p_c, o_c = play_match(model, strat, seed=42 + i)
        rewards.append(r)
        pc.append(p_c)
        oc.append(o_c)
    stats[name] = {
        "avg_reward": np.mean(rewards),
        "agent_coop": np.mean(pc),
        "opp_coop": np.mean(oc),
    }

# %% [markdown]
# ## Visualising Rewards and Cooperation Rates

# %%
names = list(stats.keys())
avg_rewards = [stats[n]["avg_reward"] for n in names]
agent_c = [stats[n]["agent_coop"] for n in names]
opp_c = [stats[n]["opp_coop"] for n in names]

save_plot_and_csv(
    names, avg_rewards, "ppo_vs_baselines", folder=str(results_dir / "ppo")
)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(names, avg_rewards)
plt.ylabel("Average Reward")
plt.title("PPO Rewards vs Opponents")
plt.ylim(0, max(avg_rewards) * 1.2)
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
x = np.arange(len(names))
w = 0.35
plt.bar(x - w / 2, agent_c, w, label="PPO Agent")
plt.bar(x + w / 2, opp_c, w, label="Opponent")
plt.ylabel("Cooperation Rate")
plt.title("Cooperation Rates")
plt.xticks(x, names, rotation=45)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Learning Curve

# %%
def load_curve():
    p = results_dir / "ppo" / "ppo_learning_curve_tit_for_tat_data.csv"
    if not p.exists():
        alt = list((results_dir / "ppo").glob("*learning*curve*_data.csv"))
        if alt:
            p = alt[0]
        else:
            print("Learning curve CSV not found.")
            return None
    df = pd.read_csv(p)
    plt.figure(figsize=(10, 5))
    plt.plot(df["x"], df["y"])
    plt.xlabel("Episode")
    plt.ylabel("Reward (moving average)")
    plt.title("PPO Learning Curve")
    plt.grid(alpha=0.3)
    plt.show()
    return df

load_curve()

# %% [markdown]
# ## Interpretation of Results
#
# * The learning curve (plotted as a rolling average) stabilizes around 15–16 due to early exploration and the windowed average; it converges quickly, showing fast—but not perfect—learning.  
# * **Consistently cooperative policy:** The agent cooperates every round (cooperation = 1.0) with all opponents, even in the face of defection.
# * **Reward structure reflects opponent strategy:** The agent earns the maximum 30 points against TFT, Always Cooperate, and Pavlov (all cooperative strategies), around 17 points against Random (which cooperates ~56% of the time), but 0 points against Always Defect.
# * A noticeable dip around episode 120 reflects a brief exploration of defection before the agent reverted to full cooperation.  
# * **Vulnerability to exploitation:** The strategy is completely exploited by unconditional defectors, receiving the minimum possible payoff.
# * **Performance against Pavlov:** The agent achieves maximum cooperation and payoff with Pavlov, demonstrating compatibility with this Win-Stay, Lose-Shift adaptive strategy.
# * **Next steps** for a more robust PPO: train against a varied opponent pool (AllD, Random, Grim Trigger) or introduce self-play fine-tuning to encourage retaliatory defection when needed.  

