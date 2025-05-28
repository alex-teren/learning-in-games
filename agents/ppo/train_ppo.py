import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Optional
import argparse
from pathlib import Path

# Import RL libraries
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from env import IPDEnv, TitForTat, AlwaysCooperate, AlwaysDefect, RandomStrategy, PavlovStrategy, GrudgerStrategy, GTFTStrategy


def save_plot_and_csv(x, y, name: str, folder: str = "results"):
    """Save PNG plot and matching CSV"""
    import pandas as pd
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame({"x": x, "y": y}).to_csv(f"{folder}/{name}_data.csv", index=False)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linewidth=2)
    plt.title(name.replace("_", " ").title())
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{folder}/{name}.png", dpi=120, bbox_inches="tight")
    plt.close()


def create_env(opponent_strategy=None, num_rounds: int = 100, seed: int = 42):
    """Create IPD environment with specified opponent"""
    if opponent_strategy is None:
        opponent_strategy = TitForTat()
    
    def _make_env():
        return IPDEnv(opponent_strategy=opponent_strategy, num_rounds=num_rounds, seed=seed)
    
    return _make_env


class ProgressCallback(BaseCallback):
    """Custom callback to track training progress"""
    
    def __init__(self, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.rewards = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Collect episode rewards
        if len(self.locals.get('rewards', [])) > 0:
            self.rewards.extend(self.locals['rewards'])
        
        return True


def train_ppo_agent(
    opponent_strategies: List = None,
    total_timesteps: int = 200000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    num_rounds: int = 100,
    seed: int = 42,
    save_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    model_name: str = "ppo_ipd"
) -> PPO:
    """
    Train PPO agent against multiple opponents or single opponent
    
    Args:
        opponent_strategies: List of strategies to train against, or single strategy
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for PPO
        n_steps: Number of steps to run for each environment per update
        batch_size: Minibatch size
        n_epochs: Number of epochs when optimizing the surrogate loss
        clip_range: Clipping parameter for PPO
        ent_coef: Entropy coefficient for exploration
        num_rounds: Number of rounds per IPD game
        seed: Random seed
        save_dir: Directory to save models
        log_dir: Directory to save logs and results
        model_name: Name for saved model
    
    Returns:
        Trained PPO model
    """
    
    print("🤖 Training PPO Agent for Iterated Prisoner's Dilemma")
    
    # Set up directories
    repo_root = Path(__file__).resolve().parents[2]
    if save_dir is None:
        save_dir = repo_root / "models"
    if log_dir is None:
        log_dir = repo_root / "results" / "ppo"
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up opponent strategies
    if opponent_strategies is None:
        opponent_strategies = [TitForTat()]
    elif not isinstance(opponent_strategies, list):
        opponent_strategies = [opponent_strategies]
    
    print(f"📊 Training against {len(opponent_strategies)} opponent(s):")
    for strategy in opponent_strategies:
        print(f"   - {strategy.name}")
    
    # Multi-opponent training setup
    if len(opponent_strategies) > 1:
        print("🔄 Multi-opponent training mode")
        training_results = {}
        
        # Train against each opponent for portion of timesteps
        timesteps_per_opponent = total_timesteps // len(opponent_strategies)
        final_model = None
        
        for i, opponent in enumerate(opponent_strategies):
            print(f"\n🎯 Training phase {i+1}/{len(opponent_strategies)}: vs {opponent.name}")
            
            # Create environment
            env = make_vec_env(
                create_env(opponent, num_rounds, seed), 
                n_envs=1,
                seed=seed
            )
            env = VecMonitor(env)
            
            # Create or continue model
            if final_model is None:
                model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    verbose=1,
                    seed=seed,
                    tensorboard_log=f"{log_dir}/tensorboard"
                )
            else:
                # Continue training with new environment
                model.set_env(env)
            
            # Training callback
            callback = ProgressCallback()
            
            # Train for this phase
            start_time = time.time()
            model.learn(
                total_timesteps=timesteps_per_opponent,
                callback=callback,
                progress_bar=True
            )
            training_time = time.time() - start_time
            
            # Evaluate against this opponent
            mean_reward, std_reward = evaluate_policy(
                model, env, n_eval_episodes=20, deterministic=True
            )
            
            training_results[opponent.name] = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'training_time': training_time
            }
            
            print(f"   📈 Mean reward vs {opponent.name}: {mean_reward:.3f} ± {std_reward:.3f}")
            
            final_model = model
            env.close()
        
        model = final_model
        
    else:
        # Single opponent training
        opponent = opponent_strategies[0]
        print(f"🎯 Single opponent training: vs {opponent.name}")
        
        # Create environment
        env = make_vec_env(
            create_env(opponent, num_rounds, seed), 
            n_envs=1,
            seed=seed
        )
        env = VecMonitor(env)
        
        # Create model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=1,
            seed=seed,
            tensorboard_log=f"{log_dir}/tensorboard"
        )
        
        # Training callback
        callback = ProgressCallback()
        
        # Train model
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        training_time = time.time() - start_time
        
        # Single opponent results
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=20, deterministic=True
        )
        
        training_results = {
            opponent.name: {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'training_time': training_time
            }
        }
        
        env.close()
    
    # Save model
    model_path = f"{save_dir}/{model_name}"
    model.save(model_path)
    print(f"💾 Model saved: {model_path}")
    
    # Comprehensive evaluation against all standard strategies
    print("\n📊 Comprehensive Evaluation:")
    all_strategies = [
        TitForTat(), AlwaysCooperate(), AlwaysDefect(), 
        RandomStrategy(seed=seed), PavlovStrategy(),
        GrudgerStrategy(), GTFTStrategy(seed=seed)
    ]
    
    evaluation_results = evaluate_against_all_opponents(
        model, all_strategies, num_rounds, seed, log_dir
    )
    
    # Plot training results if we have progress data
    if hasattr(callback, 'rewards') and len(callback.rewards) > 0:
        save_plot_and_csv(
            list(range(len(callback.rewards))), 
            callback.rewards, 
            "ppo_training_rewards", 
            folder=str(log_dir)
        )
    
    return model


def evaluate_against_all_opponents(
    model: PPO, 
    strategies: List, 
    num_rounds: int = 100,
    seed: int = 42,
    log_dir: Optional[str] = None,
    n_episodes: int = 50
) -> Dict:
    """Evaluate trained model against all opponent strategies"""
    
    print("🎯 Evaluating against all opponents:")
    results = {}
    
    for strategy in strategies:
        print(f"   Testing vs {strategy.name}...")
        
        # Create environment for this opponent
        env = make_vec_env(
            create_env(strategy, num_rounds, seed), 
            n_envs=1,
            seed=seed
        )
        env = VecMonitor(env)
        
        # Evaluate
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=n_episodes, deterministic=True
        )
        
        # Get cooperation rate
        coop_rate = get_cooperation_rate(model, env, n_episodes=10)
        
        results[strategy.name] = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'cooperation_rate': coop_rate
        }
        
        print(f"     📈 Reward: {mean_reward:.3f} ± {std_reward:.3f}")
        print(f"     🤝 Cooperation: {coop_rate:.1%}")
        
        env.close()
    
    # Save results
    if log_dir:
        import pandas as pd
        results_df = pd.DataFrame(results).T
        results_df.to_csv(f"{log_dir}/evaluation_results.csv")
        
        # Plot results
        plot_evaluation_results(results, log_dir)
    
    return results


def get_cooperation_rate(model: PPO, env, n_episodes: int = 10) -> float:
    """Calculate cooperation rate of the model"""
    cooperation_count = 0
    total_actions = 0
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Action 0 = Cooperate, Action 1 = Defect
            if action[0] == 0:
                cooperation_count += 1
            total_actions += 1
    
    return cooperation_count / total_actions if total_actions > 0 else 0.0


def plot_evaluation_results(results: Dict, log_dir: str):
    """Plot evaluation results"""
    strategies = list(results.keys())
    rewards = [results[s]['mean_reward'] for s in strategies]
    reward_stds = [results[s]['std_reward'] for s in strategies]
    coop_rates = [results[s]['cooperation_rate'] for s in strategies]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Rewards plot
    bars1 = ax1.bar(strategies, rewards, yerr=reward_stds, capsize=5, alpha=0.8)
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('PPO Performance vs Different Opponents')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, reward in zip(bars1, rewards):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{reward:.2f}', ha='center', va='bottom')
    
    # Cooperation rates plot
    bars2 = ax2.bar(strategies, coop_rates, alpha=0.8, color='green')
    ax2.set_ylabel('Cooperation Rate')
    ax2.set_title('PPO Cooperation Rate vs Different Opponents')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Add percentage labels on bars
    for bar, rate in zip(bars2, coop_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{log_dir}/ppo_evaluation.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for IPD")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Total training timesteps (default: 200000)")
    parser.add_argument("--num_rounds", type=int, default=100,
                        help="Number of rounds per game (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--multi_opponent", action="store_true",
                        help="Train against multiple opponents")
    
    args = parser.parse_args()
    
    print("=== PPO Training for Iterated Prisoner's Dilemma ===")
    
    # Set up opponent strategies
    if args.multi_opponent:
        opponents = [
            TitForTat(), AlwaysCooperate(), AlwaysDefect(), 
            RandomStrategy(seed=args.seed), PavlovStrategy(),
            GrudgerStrategy(), GTFTStrategy(seed=args.seed)
        ]
        model_name = "ppo_multi_opponent"
    else:
        opponents = [TitForTat()]  # Default opponent
        model_name = "ppo_single_opponent"
    
    # Train model
    model = train_ppo_agent(
        opponent_strategies=opponents,
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        num_rounds=args.num_rounds,
        seed=args.seed,
        model_name=model_name
    )
    
    print("\n✅ PPO Training completed!")
    print(f"🎯 Model trained for {args.timesteps:,} timesteps")
    print(f"📊 Results saved in results/ppo/")
    print(f"💾 Model saved in models/{model_name}") 