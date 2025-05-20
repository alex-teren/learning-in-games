import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
import time
import gym
import sys

# Add project root to path to allow imports from other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from env import IPDEnv, TitForTat, AlwaysCooperate, AlwaysDefect, RandomStrategy


def save_plot_and_csv(x, y, name: str, folder: str = "results"):
    """Save PNG plot **and** matching CSV so LLM can analyse the numbers."""
    import os, pandas as pd, matplotlib.pyplot as plt
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame({"x": x, "y": y}).to_csv(f"{folder}/{name}_data.csv", index=False)
    plt.figure(); plt.plot(x, y); plt.title(name.replace("_", " ").title())
    plt.savefig(f"{folder}/{name}.png", dpi=120, bbox_inches="tight"); plt.close()


def create_env(opponent_strategy="tit_for_tat", num_rounds=10, memory_size=3, seed=None):
    """
    Create and configure the IPD environment
    
    Args:
        opponent_strategy: Opponent strategy to use
        num_rounds: Number of rounds per episode
        memory_size: History memory size
        seed: Random seed
        
    Returns:
        Configured environment
    """
    env = IPDEnv(
        num_rounds=num_rounds,
        memory_size=memory_size,
        opponent_strategy=opponent_strategy,
        seed=seed
    )
    
    # Wrap with Monitor to record episode statistics
    env = Monitor(env)
    
    return env


def train_ppo_agent(
    opponent_strategy="tit_for_tat",
    total_timesteps=500000,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    ent_coef=0.01,
    clip_range=0.2,
    n_epochs=10,
    num_rounds=10,
    memory_size=3,
    seed=42,
    save_dir="../../models",
    log_dir="../../results",
    eval_freq=10000,
    n_eval_episodes=50
):
    """
    Train a PPO agent against a specific opponent
    
    Args:
        opponent_strategy: Opponent strategy to train against
        total_timesteps: Total training timesteps
        n_steps: Number of steps to run for each environment per update
        batch_size: Minibatch size
        learning_rate: Learning rate
        gamma: Discount factor
        ent_coef: Entropy coefficient
        clip_range: Clipping parameter for PPO
        n_epochs: Number of epoch when optimizing the surrogate loss
        num_rounds: Number of rounds per episode
        memory_size: History memory size
        seed: Random seed
        save_dir: Directory to save models
        log_dir: Directory to save logs
        eval_freq: Evaluate the agent every eval_freq timesteps
        n_eval_episodes: Number of episodes to evaluate
        
    Returns:
        Trained PPO model, results dataframe
    """
    print(f"Training PPO agent against {opponent_strategy} opponent...")
    
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/ppo", exist_ok=True)
    
    # Set random seeds
    np.random.seed(seed)
    
    # Create train and eval environments with different seeds
    train_env = create_env(
        opponent_strategy=opponent_strategy,
        num_rounds=num_rounds,
        memory_size=memory_size,
        seed=seed
    )
    
    eval_env = create_env(
        opponent_strategy=opponent_strategy,
        num_rounds=num_rounds,
        memory_size=memory_size,
        seed=seed+100
    )
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/ppo_best_{opponent_strategy}",
        log_path=f"{log_dir}/ppo",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=n_eval_episodes
    )
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"{save_dir}/ppo_checkpoints_{opponent_strategy}",
        name_prefix="ppo_ipd",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
        clip_range=clip_range,
        n_epochs=n_epochs,
        verbose=1,
        tensorboard_log=f"{log_dir}/ppo_tensorboard",
        seed=seed
    )
    
    # Train the agent
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback]
    )
    total_training_time = time.time() - start_time
    
    # Save the final model
    model.save(f"{save_dir}/ppo_final_{opponent_strategy}")
    
    print(f"Training completed in {total_training_time:.2f} seconds")
    
    # Evaluate the trained agent against different opponents
    evaluate_against_all_opponents(model, num_rounds, memory_size, seed, log_dir)
    
    # Plot and save the learning curve
    plot_results(log_dir, opponent_strategy)
    
    return model


def evaluate_against_all_opponents(model, num_rounds=10, memory_size=3, seed=42, log_dir="../../results"):
    """
    Evaluate a trained model against different opponent strategies
    
    Args:
        model: Trained PPO model
        num_rounds: Number of rounds per episode
        memory_size: History memory size
        seed: Random seed
        log_dir: Directory to save evaluation results
    
    Returns:
        DataFrame with evaluation results
    """
    print("Evaluating against different opponents...")
    
    # Define opponent strategies to evaluate against
    opponent_strategies = {
        "tit_for_tat": TitForTat(),
        "always_cooperate": AlwaysCooperate(),
        "always_defect": AlwaysDefect(),
        "random": RandomStrategy(seed=seed+200)
    }
    
    # Create results dataframe
    results = []
    
    # Evaluate against each opponent
    for opponent_name, opponent_strategy in opponent_strategies.items():
        env = create_env(
            opponent_strategy=opponent_strategy,
            num_rounds=num_rounds,
            memory_size=memory_size,
            seed=seed+300
        )
        
        # Run evaluation
        mean_reward, std_reward = evaluate_policy(
            model, 
            env, 
            n_eval_episodes=100,
            deterministic=True
        )
        
        # Get cooperation rates
        cooperation_rates = get_cooperation_rates(model, env, n_episodes=100)
        
        # Store results
        results.append({
            "opponent": opponent_name,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "agent_cooperation_rate": cooperation_rates["agent"],
            "opponent_cooperation_rate": cooperation_rates["opponent"]
        })
    
    # Create dataframe and save to CSV
    results_df = pd.DataFrame(results)
    os.makedirs(f"{log_dir}/ppo", exist_ok=True)
    results_df.to_csv(f"{log_dir}/ppo/evaluation_results.csv", index=False)
    
    print("Evaluation results:")
    print(results_df)
    
    return results_df


def get_cooperation_rates(model, env, n_episodes=100):
    """
    Calculate cooperation rates for agent and opponent
    
    Args:
        model: Trained model
        env: Environment to evaluate in
        n_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary with cooperation rates
    """
    agent_actions = []
    opponent_actions = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()  # Unpack observation and info
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)  # Updated step API
            done = terminated or truncated
            
            agent_actions.append(info["player_action"])
            opponent_actions.append(info["opponent_action"])
    
    agent_coop_rate = agent_actions.count(0) / len(agent_actions)
    opponent_coop_rate = opponent_actions.count(0) / len(opponent_actions)
    
    return {
        "agent": agent_coop_rate,
        "opponent": opponent_coop_rate
    }


def plot_results(log_dir="../../results", opponent_strategy="tit_for_tat"):
    """
    Plot the training results
    
    Args:
        log_dir: Directory with logs
        opponent_strategy: Opponent strategy used in training
    """
    # Find monitor files
    monitor_files = []
    for dirpath, dirnames, filenames in os.walk(f"{log_dir}/ppo"):
        for filename in filenames:
            if filename.startswith("monitor.csv"):
                monitor_files.append(os.path.join(dirpath, filename))
    
    if not monitor_files:
        print("No monitor files found, skipping plotting")
        return
    
    # Use the most recent file for plotting
    monitor_file = sorted(monitor_files, key=os.path.getmtime)[-1]
    
    # Load and process monitor data
    monitor_data = pd.read_csv(monitor_file, skiprows=1)
    
    if monitor_data.empty:
        print("Monitor data is empty, skipping plotting")
        return
    
    # Extract relevant columns
    rewards = monitor_data["r"]
    timesteps = monitor_data["l"].cumsum()
    episodes = np.arange(len(rewards))
    
    # Calculate rolling average
    window = min(50, len(rewards))
    if window > 0:
        rolling_rewards = rewards.rolling(window=window).mean()
    else:
        rolling_rewards = rewards
    
    # Create plot and save data using helper function
    save_plot_and_csv(
        episodes.tolist(), 
        rolling_rewards.tolist(), 
        f"ppo_learning_curve_{opponent_strategy}",
        folder=f"{log_dir}/ppo"
    )
    
    # Also save raw rewards
    save_plot_and_csv(
        episodes.tolist(), 
        rewards.tolist(), 
        f"ppo_raw_rewards_{opponent_strategy}",
        folder=f"{log_dir}/ppo"
    )
    
    print(f"Learning curves saved to {log_dir}/ppo/")


if __name__ == "__main__":
    # Train against Tit-for-Tat opponent
    model_tft = train_ppo_agent(
        opponent_strategy="tit_for_tat",
        total_timesteps=200000,  # Reduced for faster execution
        num_rounds=10,
        memory_size=3,
        seed=42
    )
    
    # Optionally train against other opponents
    # model_allc = train_ppo_agent(
    #     opponent_strategy="always_cooperate",
    #     total_timesteps=200000,
    #     num_rounds=10,
    #     memory_size=3,
    #     seed=43
    # )
    # 
    # model_alld = train_ppo_agent(
    #     opponent_strategy="always_defect",
    #     total_timesteps=200000,
    #     num_rounds=10,
    #     memory_size=3,
    #     seed=44
    # )
    # 
    # model_random = train_ppo_agent(
    #     opponent_strategy="random",
    #     total_timesteps=200000,
    #     num_rounds=10,
    #     memory_size=3,
    #     seed=45
    # ) 