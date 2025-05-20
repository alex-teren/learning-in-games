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

from env import IPDEnv, TitForTat, AlwaysCooperate, AlwaysDefect, RandomStrategy, PavlovStrategy


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
    opponent_strategy="mixed",
    total_timesteps=1000000,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    ent_coef=0.05,
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
    Train a PPO agent against a specific opponent or a mix of opponents
    
    Args:
        opponent_strategy: Opponent strategy to train against. Use "mixed" for training against multiple strategies.
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
    
    # For mixed training, we'll create a list of different environments
    if opponent_strategy == "mixed":
        train_envs = []
        opponent_types = ["tit_for_tat", "always_cooperate", "always_defect", "random", "pavlov"]
        
        for i, opponent_type in enumerate(opponent_types):
            env = create_env(
                opponent_strategy=opponent_type,
                num_rounds=num_rounds,
                memory_size=memory_size,
                seed=seed + i
            )
            train_envs.append((env, opponent_type))
            
        # We'll use the first environment for model initialization
        train_env = train_envs[0][0]
    else:
        # Create a single training environment
        train_env = create_env(
            opponent_strategy=opponent_strategy,
            num_rounds=num_rounds,
            memory_size=memory_size,
            seed=seed
        )
        train_envs = [(train_env, opponent_strategy)]
    
    # Create evaluation environments for each strategy
    eval_envs = {
        "tit_for_tat": create_env("tit_for_tat", num_rounds, memory_size, seed+100),
        "always_cooperate": create_env("always_cooperate", num_rounds, memory_size, seed+101),
        "always_defect": create_env("always_defect", num_rounds, memory_size, seed+102),
        "random": create_env("random", num_rounds, memory_size, seed+103),
        "pavlov": create_env("pavlov", num_rounds, memory_size, seed+104)
    }
    
    # Use TFT as the main evaluation environment
    eval_env = eval_envs["tit_for_tat"]
    
    # Create evaluation callback with extended metrics
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
    
    # Train the agent, switching environments if using mixed strategy
    start_time = time.time()
    
    if opponent_strategy == "mixed":
        # Train against different opponents sequentially in multiple rounds
        steps_per_env = total_timesteps // (len(train_envs) * 2)  # Divide time equally
        
        for training_round in range(2):  # Multiple rounds of training
            for i, (env, env_name) in enumerate(train_envs):
                print(f"Training round {training_round+1}, opponent: {env_name} for {steps_per_env} steps...")
                model.set_env(env)
                model.learn(
                    total_timesteps=steps_per_env,
                    callback=[eval_callback, checkpoint_callback],
                    reset_num_timesteps=False  # Continue counting timesteps
                )
                
                # Save intermediate model after each opponent
                model.save(f"{save_dir}/ppo_intermediate_{opponent_strategy}_{env_name}_{training_round}")
                
                # Evaluate current progress against all opponents
                print(f"Intermediate evaluation after training against {env_name}:")
                intermediate_results = evaluate_against_all_opponents(
                    model, 
                    num_rounds, 
                    memory_size, 
                    seed + 1000 + (training_round * 100) + i,
                    log_dir,
                    save_suffix=f"_intermediate_{training_round}_{env_name}"
                )
    else:
        # Train against a single opponent
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback]
        )
    
    total_training_time = time.time() - start_time
    
    # Save the final model
    model.save(f"{save_dir}/ppo_final_{opponent_strategy}")
    
    print(f"Training completed in {total_training_time:.2f} seconds")
    
    # Evaluate the trained agent against different opponents
    final_results = evaluate_against_all_opponents(model, num_rounds, memory_size, seed, log_dir)
    
    # Plot and save the learning curve
    plot_results(log_dir, opponent_strategy)
    
    return model


def evaluate_against_all_opponents(model, num_rounds=10, memory_size=3, seed=42, log_dir="../../results", save_suffix=""):
    """
    Evaluate a trained model against different opponent strategies
    
    Args:
        model: Trained PPO model
        num_rounds: Number of rounds per episode
        memory_size: History memory size
        seed: Random seed
        log_dir: Directory to save evaluation results
        save_suffix: Suffix to add to save files for intermediate evaluations
    
    Returns:
        DataFrame with evaluation results
    """
    print("Evaluating against different opponents...")
    
    # Define opponent strategies to evaluate against
    opponent_strategies = {
        "tit_for_tat": TitForTat(),
        "always_cooperate": AlwaysCooperate(),
        "always_defect": AlwaysDefect(),
        "random": RandomStrategy(seed=seed+200),
        "pavlov": PavlovStrategy()  # Added Pavlov strategy
    }
    
    # Create results dataframe
    results = []
    
    # For detailed action analysis by round
    action_dynamics = {}
    
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
        cooperation_data = get_cooperation_rates(model, env, n_episodes=100, detailed=True)
        
        # Store action dynamics for this opponent
        action_dynamics[opponent_name] = cooperation_data["round_data"]
        
        # Store results
        results.append({
            "opponent": opponent_name,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "agent_cooperation_rate": cooperation_data["agent"],
            "opponent_cooperation_rate": cooperation_data["opponent"]
        })
    
    # Create dataframe and save to CSV
    results_df = pd.DataFrame(results)
    os.makedirs(f"{log_dir}/ppo", exist_ok=True)
    results_df.to_csv(f"{log_dir}/ppo/evaluation_results{save_suffix}.csv", index=False)
    
    print("Evaluation results:")
    print(results_df)
    
    # Save action dynamics data for visualization
    np.save(f"{log_dir}/ppo/action_dynamics{save_suffix}.npy", action_dynamics)
    
    # Create visualizations of action dynamics
    plot_action_dynamics(action_dynamics, f"{log_dir}/ppo", suffix=save_suffix)
    
    # Create bar chart visualization of cooperation rates
    plot_cooperation_comparison(results_df, f"{log_dir}/ppo", suffix=save_suffix)
    
    return results_df


def get_cooperation_rates(model, env, n_episodes=100, detailed=False):
    """
    Calculate cooperation rates for agent and opponent
    
    Args:
        model: Trained model
        env: Environment to evaluate in
        n_episodes: Number of episodes to evaluate
        detailed: Whether to return detailed cooperation data
        
    Returns:
        Dictionary with cooperation rates
    """
    agent_actions = []
    opponent_actions = []
    
    if detailed:
        round_data = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()  # Unpack observation and info
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)  # Updated step API
            done = terminated or truncated
            
            agent_actions.append(info["player_action"])
            opponent_actions.append(info["opponent_action"])
            
            if detailed:
                round_data.append({
                    "round": info["round"],
                    "player_action": info["player_action"],
                    "opponent_action": info["opponent_action"],
                    "player_payoff": info["player_payoff"],
                    "opponent_payoff": info["opponent_payoff"]
                })
    
    agent_coop_rate = agent_actions.count(0) / len(agent_actions)
    opponent_coop_rate = opponent_actions.count(0) / len(opponent_actions)
    
    if detailed:
        return {
            "agent": agent_coop_rate,
            "opponent": opponent_coop_rate,
            "round_data": round_data
        }
    else:
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
    
    # Create plot
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, rewards, alpha=0.3, label='Episode Reward')
    plt.plot(episodes, rolling_rewards, linewidth=2, label=f'{window}-Episode Moving Average')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'PPO Training Rewards against {opponent_strategy} opponent')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig(f"{log_dir}/ppo/learning_curve_{opponent_strategy}.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Learning curve saved to {log_dir}/ppo/learning_curve_{opponent_strategy}.png")


def plot_action_dynamics(action_dynamics, save_dir, suffix=""):
    """
    Plot the dynamics of actions over rounds for different opponents
    
    Args:
        action_dynamics: Dictionary with action data
        save_dir: Directory to save plots
        suffix: Optional suffix for saved files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for opponent_name, round_data in action_dynamics.items():
        # Group data by round
        rounds = []
        agent_actions = []
        opponent_actions = []
        agent_rewards = []
        
        max_round = max([d["round"] for d in round_data])
        
        for round_num in range(1, max_round + 1):
            # Extract data for this round
            round_entries = [d for d in round_data if d["round"] == round_num]
            
            # Calculate cooperation rate for this round
            agent_coop_rate = sum(1 for d in round_entries if d["player_action"] == 0) / len(round_entries)
            opponent_coop_rate = sum(1 for d in round_entries if d["opponent_action"] == 0) / len(round_entries)
            
            rounds.append(round_num)
            agent_actions.append(agent_coop_rate)  # Changed to cooperation rate for clarity
            opponent_actions.append(opponent_coop_rate)
            
            # Calculate average reward for agent
            agent_reward = sum(d["player_payoff"] for d in round_entries) / len(round_entries)
            agent_rewards.append(agent_reward)
        
        # Create plot with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot cooperation rates
        ax1.plot(rounds, agent_actions, 'o-', color='blue', linewidth=2, label='PPO Agent')
        ax1.plot(rounds, opponent_actions, 'o-', color='orange', linewidth=2, label=opponent_name)
        ax1.set_ylabel('Рівень кооперації')
        ax1.set_title(f'Динаміка кооперації по раундам проти {opponent_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)  # 0-100% cooperation rate
        
        # Plot rewards
        ax2.plot(rounds, agent_rewards, 'o-', color='blue', linewidth=2)
        ax2.set_xlabel('Раунд')
        ax2.set_ylabel('Середня винагорода')
        ax2.set_title(f'Винагороди PPO агента проти {opponent_name}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/dynamics_{opponent_name}{suffix}.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    # Create summary plot showing agent cooperation rate against all opponents
    plt.figure(figsize=(14, 8))
    
    for opponent_name, round_data in action_dynamics.items():
        rounds = []
        agent_coop_rates = []
        
        max_round = max([d["round"] for d in round_data])
        
        for round_num in range(1, max_round + 1):
            round_entries = [d for d in round_data if d["round"] == round_num]
            agent_coop_rate = sum(1 for d in round_entries if d["player_action"] == 0) / len(round_entries)
            
            rounds.append(round_num)
            agent_coop_rates.append(agent_coop_rate)
        
        plt.plot(rounds, agent_coop_rates, 'o-', linewidth=2, label=opponent_name)
    
    plt.xlabel('Раунд')
    plt.ylabel('Рівень кооперації агента')
    plt.title('Рівень кооперації PPO-агента проти різних опонентів')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)  # 0-100% cooperation rate
    plt.savefig(f"{save_dir}/agent_cooperation_by_opponent{suffix}.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # Create heatmap of agent actions by round for each opponent
    for opponent_name, round_data in action_dynamics.items():
        # Create matrix of probabilities: rows are rounds, columns are [CC, CD, DC, DD] outcomes
        max_round = max([d["round"] for d in round_data])
        outcome_matrix = np.zeros((max_round, 4))
        
        for round_num in range(1, max_round + 1):
            round_entries = [d for d in round_data if d["round"] == round_num]
            total_count = len(round_entries)
            
            # Count outcomes [CC, CD, DC, DD]
            cc_count = sum(1 for d in round_entries if d["player_action"] == 0 and d["opponent_action"] == 0)
            cd_count = sum(1 for d in round_entries if d["player_action"] == 0 and d["opponent_action"] == 1)
            dc_count = sum(1 for d in round_entries if d["player_action"] == 1 and d["opponent_action"] == 0)
            dd_count = sum(1 for d in round_entries if d["player_action"] == 1 and d["opponent_action"] == 1)
            
            outcome_matrix[round_num-1, 0] = cc_count / total_count
            outcome_matrix[round_num-1, 1] = cd_count / total_count
            outcome_matrix[round_num-1, 2] = dc_count / total_count
            outcome_matrix[round_num-1, 3] = dd_count / total_count
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(outcome_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(label='Ймовірність')
        plt.xlabel('Результат гри (CC, CD, DC, DD)')
        plt.ylabel('Раунд')
        plt.title(f'Розподіл результатів гри проти {opponent_name}')
        plt.xticks([0, 1, 2, 3], ['CC', 'CD', 'DC', 'DD'])
        plt.yticks(np.arange(0, max_round, 2), np.arange(1, max_round+1, 2))
        plt.savefig(f"{save_dir}/outcome_heatmap_{opponent_name}{suffix}.png", dpi=100, bbox_inches='tight')
        plt.close()


def plot_cooperation_comparison(results_df, save_dir, suffix=""):
    """
    Create a bar chart comparing cooperation rates against different opponents
    
    Args:
        results_df: DataFrame with evaluation results
        save_dir: Directory to save the plot
        suffix: Optional suffix for saved files
    """
    plt.figure(figsize=(12, 7))
    
    opponents = results_df['opponent']
    agent_coop = results_df['agent_cooperation_rate']
    opponent_coop = results_df['opponent_cooperation_rate']
    rewards = results_df['mean_reward']
    
    # Set up bar width
    bar_width = 0.25
    r1 = np.arange(len(opponents))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create bars
    plt.bar(r1, agent_coop, width=bar_width, label='PPO Agent', color='blue', alpha=0.7)
    plt.bar(r2, opponent_coop, width=bar_width, label='Opponent', color='orange', alpha=0.7)
    plt.bar(r3, rewards/10, width=bar_width, label='Reward/10', color='green', alpha=0.7)
    
    # Add labels and legend
    plt.xlabel('Стратегія опонента')
    plt.ylabel('Рівень кооперації / Винагорода/10')
    plt.title('Порівняння рівнів кооперації та винагород проти різних опонентів')
    plt.xticks([r + bar_width for r in range(len(opponents))], opponents)
    plt.legend()
    
    # Save figure
    plt.savefig(f"{save_dir}/cooperation_comparison{suffix}.png", dpi=100, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Train against mixed opponents 
    model_mixed = train_ppo_agent(
        opponent_strategy="mixed",
        total_timesteps=1000000,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.05,
        num_rounds=10,
        memory_size=3,
        seed=42
    )
    
    # Alternatively, train against specific opponents
    # model_tft = train_ppo_agent(
    #     opponent_strategy="tit_for_tat",
    #     total_timesteps=500000,
    #     ent_coef=0.05,
    #     num_rounds=10,
    #     memory_size=3,
    #     seed=42
    # ) 