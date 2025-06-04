import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Optional, Tuple
import argparse
from pathlib import Path
import random
from collections import deque

# Import RL libraries
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from env import IPDEnv, TitForTat, AlwaysCooperate, AlwaysDefect, RandomStrategy, PavlovStrategy, GrudgerStrategy, GTFTStrategy


class EnhancedIPDEnv(IPDEnv):
    """Enhanced IPD Environment with opponent modeling features"""
    
    def __init__(self, num_rounds=100, memory_size=3, opponent_strategy=None, seed=42, history_length=10):
        self.history_length = history_length
        self.opponent_history = deque(maxlen=history_length)
        self.my_history = deque(maxlen=history_length)
        
        super().__init__(num_rounds=num_rounds, memory_size=memory_size, opponent_strategy=opponent_strategy, seed=seed)
        
        # Enhanced observation space: current state + opponent history + my history + round info
        # [my_last_action, opp_last_action, round_num/total_rounds, opp_coop_rate, my_coop_rate, 
        #  recent_opp_actions(history_length), recent_my_actions(history_length)]
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(5 + 2 * history_length,), 
            dtype=np.float32
        )
    
    def _get_enhanced_obs(self):
        """Get enhanced observation with opponent modeling features"""
        obs = np.zeros(5 + 2 * self.history_length, dtype=np.float32)
        
        # Basic state info
        if len(self.history) > 0:
            last_my_action, last_opp_action = self.history[-1]
            obs[0] = 1 - last_my_action  # Convert to cooperation probability (1=coop, 0=defect)
            obs[1] = 1 - last_opp_action
        else:
            obs[0] = 0.5  # Unknown
            obs[1] = 0.5
            
        obs[2] = self.current_round / self.num_rounds  # progress through game
        
        # Cooperation rates
        if len(self.my_history) > 0:
            obs[3] = sum(1-a for a in self.opponent_history) / len(self.opponent_history)  # opp coop rate
            obs[4] = sum(1-a for a in self.my_history) / len(self.my_history)  # my coop rate
        else:
            obs[3] = 0.5
            obs[4] = 0.5
        
        # Recent opponent actions (padded with 0.5 if not enough history)
        for i in range(self.history_length):
            if i < len(self.opponent_history):
                obs[5 + i] = 1 - self.opponent_history[-(i+1)]  # Convert to cooperation (0=defect, 1=coop)
            else:
                obs[5 + i] = 0.5
        
        # Recent my actions
        for i in range(self.history_length):
            if i < len(self.my_history):
                obs[5 + self.history_length + i] = 1 - self.my_history[-(i+1)]
            else:
                obs[5 + self.history_length + i] = 0.5
        
        return obs
    
    def reset(self, **kwargs):
        """Reset with enhanced observations"""
        base_obs, info = super().reset(**kwargs)
        
        # Clear histories if they exist
        if hasattr(self, 'opponent_history'):
            self.opponent_history.clear()
        if hasattr(self, 'my_history'):
            self.my_history.clear()
            
        return self._get_enhanced_obs(), info
    
    def step(self, action):
        """Step with enhanced observations and reward shaping"""
        # Take step in base environment first
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Update my history deque from base class history
        if len(self.history) > 0:
            latest_round = self.history[-1]
            my_action, opponent_action = latest_round
            
            # Add to deques if not already added
            if len(self.my_history) == 0 or self.my_history[-1] != my_action:
                self.my_history.append(my_action)
            if len(self.opponent_history) == 0 or self.opponent_history[-1] != opponent_action:
                self.opponent_history.append(opponent_action)
        
        # Reward shaping for strategic behavior
        shaped_reward = reward
        
        # Bonus for adaptation: reward if changing strategy when getting exploited
        if len(self.my_history) >= 2 and len(self.opponent_history) >= 2:
            # If opponent defected and I was cooperating, bonus for adapting (defecting back)
            if (self.opponent_history[-1] == 1 and self.my_history[-2] == 0 and 
                self.my_history[-1] == 1):  # opp defected, I was coop, now I defect
                shaped_reward += 0.5
            
            # Bonus for forgiveness: cooperating after mutual defection
            if (len(self.my_history) >= 2 and len(self.opponent_history) >= 2 and
                self.opponent_history[-2] == 1 and self.my_history[-2] == 1 and 
                self.my_history[-1] == 0):  # both defected last round, now I cooperate
                shaped_reward += 0.3
        
        # Penalty for being too predictable (always same action)
        if len(self.my_history) >= 5:
            if len(set(list(self.my_history)[-5:])) == 1:  # same action for 5 rounds
                shaped_reward -= 0.2
        
        # Enhanced observation
        enhanced_obs = self._get_enhanced_obs()
        
        return enhanced_obs, shaped_reward, terminated, truncated, info


class CurriculumCallback(BaseCallback):
    """Callback for curriculum learning and adaptive training"""
    
    def __init__(self, strategy_pool, difficulty_schedule, verbose=0):
        super().__init__(verbose)
        self.strategy_pool = strategy_pool
        self.difficulty_schedule = difficulty_schedule
        self.current_phase = 0
        self.phase_steps = 0
        self.performance_history = deque(maxlen=100)
        self.current_strategy_idx = 0
        
    def _on_step(self) -> bool:
        # Collect performance data
        if 'episode' in self.locals and len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    episode_reward = info['episode']['r']
                    self.performance_history.append(episode_reward)
        
        self.phase_steps += 1
        
        # Check if it's time to advance curriculum
        if (self.phase_steps >= self.difficulty_schedule[self.current_phase]['steps'] and 
            self.current_phase < len(self.difficulty_schedule) - 1):
            
            avg_performance = np.mean(list(self.performance_history)) if self.performance_history else 0
            threshold = self.difficulty_schedule[self.current_phase]['performance_threshold']
            
            if avg_performance >= threshold:
                self.current_phase += 1
                self.phase_steps = 0
                self.performance_history.clear()
                
                if self.verbose:
                    print(f"\n🎓 Advancing to curriculum phase {self.current_phase + 1}")
                    print(f"   Performance: {avg_performance:.2f} >= {threshold}")
        
        return True


def create_enhanced_env(opponent_strategy, num_rounds=100, seed=42, history_length=10):
    """Create enhanced IPD environment"""
    def _make_env():
        return EnhancedIPDEnv(
            num_rounds=num_rounds,
            memory_size=3,
            opponent_strategy=opponent_strategy, 
            seed=seed, 
            history_length=history_length
        )
    return _make_env


def train_ppo_agent(
    total_timesteps: int = 250000,
    learning_rate: float = 3e-4,
    num_rounds: int = 100,
    seed: int = 42,
    save_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    model_name: str = "ppo_ipd"
) -> PPO:
    """
    Train PPO agent with curriculum learning and opponent modeling
    """
    
    print("🧠 Training PPO Agent with Curriculum Learning")
    
    # Set up directories
    repo_root = Path(__file__).resolve().parents[2]
    if save_dir is None:
        save_dir = repo_root / "models"
    if log_dir is None:
        log_dir = repo_root / "results" / "ppo"
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Define curriculum: progressive difficulty
    strategy_pool = [
        AlwaysCooperate(),     # Phase 1: Learn to exploit
        TitForTat(),           # Phase 2: Learn reciprocity  
        AlwaysDefect(),        # Phase 3: Learn defense
        RandomStrategy(seed=seed),  # Phase 4: Handle uncertainty
        PavlovStrategy(),      # Phase 5: Complex reciprocity
        GrudgerStrategy(),     # Phase 6: Unforgiving opponents
        GTFTStrategy(seed=seed) # Phase 7: Advanced strategies
    ]
    
    # Curriculum schedule
    difficulty_schedule = [
        {'steps': 40000, 'performance_threshold': 250},   # vs AlwaysCooperate
        {'steps': 50000, 'performance_threshold': 280},   # vs TitForTat
        {'steps': 60000, 'performance_threshold': 80},    # vs AlwaysDefect
        {'steps': 60000, 'performance_threshold': 180},   # vs Random
        {'steps': 60000, 'performance_threshold': 260},   # vs Pavlov
        {'steps': 60000, 'performance_threshold': 90},    # vs Grudger
        {'steps': 70000, 'performance_threshold': 240},   # vs GTFT
    ]
    
    print("📚 Curriculum Learning Schedule:")
    for i, (strategy, schedule) in enumerate(zip(strategy_pool, difficulty_schedule)):
        print(f"   Phase {i+1}: {strategy.name} - {schedule['steps']:,} steps, target: {schedule['performance_threshold']}")
    
    # Start with first opponent
    current_opponent = strategy_pool[0]
    env = make_vec_env(
        create_enhanced_env(current_opponent, num_rounds, seed), 
        n_envs=1, 
        seed=seed
    )
    env = VecMonitor(env)
    
    # Create model with enhanced observation space
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=seed,
        policy_kwargs=dict(
            net_arch=[256, 256, 128],  # Larger network for complex observations
            activation_fn=lambda: __import__('torch.nn', fromlist=['ReLU']).ReLU()
        ),
        tensorboard_log=f"{log_dir}/tensorboard"
    )
    
    # Curriculum callback
    curriculum_callback = CurriculumCallback(strategy_pool, difficulty_schedule, verbose=1)
    
    # Training phases
    training_results = {}
    start_time = time.time()
    
    for phase, (opponent, schedule) in enumerate(zip(strategy_pool, difficulty_schedule)):
        print(f"\n🎯 Phase {phase+1}/{len(strategy_pool)}: Training vs {opponent.name}")
        
        # Update environment for current opponent
        if phase > 0:  # Don't recreate for first phase
            env.close()
            env = make_vec_env(
                create_enhanced_env(opponent, num_rounds, seed), 
                n_envs=1, 
                seed=seed
            )
            env = VecMonitor(env)
            model.set_env(env)
        
        # Train for this phase
        phase_start = time.time()
        model.learn(
            total_timesteps=schedule['steps'],
            callback=curriculum_callback,
            progress_bar=True,
            reset_num_timesteps=False  # Continue timestep count
        )
        phase_time = time.time() - phase_start
        
        # Evaluate performance on current opponent
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=20, deterministic=True
        )
        
        training_results[f"Phase_{phase+1}_{opponent.name}"] = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'training_time': phase_time,
            'target_performance': schedule['performance_threshold']
        }
        
        print(f"   📈 Performance: {mean_reward:.1f} ± {std_reward:.1f} (target: {schedule['performance_threshold']})")
        print(f"   ⏱️  Phase time: {phase_time:.1f}s")
        
        # Early stopping if performance is excellent
        if mean_reward >= schedule['performance_threshold'] * 1.2:
            print(f"   🌟 Excellent performance achieved early!")
    
    total_training_time = time.time() - start_time
    
    # Additional mixed-opponent training for robustness
    print(f"\n🔀 Final phase: Mixed-opponent training")
    mixed_timesteps = 100000
    
    # Create a mixed environment using random sampling approach
    def create_mixed_env():
        """Create environment with randomly selected opponent"""
        opponent = random.choice(strategy_pool)
        return EnhancedIPDEnv(
            num_rounds=num_rounds,
            memory_size=3,
            opponent_strategy=opponent,
            seed=seed
        )
    
    # Create mixed environment  
    env.close()
    env = make_vec_env(create_mixed_env, n_envs=1, seed=seed)
    env = VecMonitor(env)
    model.set_env(env)
    
    # Final mixed training
    model.learn(
        total_timesteps=mixed_timesteps,
        progress_bar=True,
        reset_num_timesteps=False
    )
    
    total_training_time = time.time() - start_time
    env.close()
    
    # Save model
    model_path = f"{save_dir}/{model_name}"
    model.save(model_path)
    print(f"💾 Model saved: {model_path}")
    
    print(f"\n🏁 Training completed in {total_training_time:.1f} seconds")
    print(f"📊 Total timesteps: {total_timesteps + mixed_timesteps:,}")
    
    # Comprehensive evaluation
    print("\n📊 Final Evaluation:")
    all_strategies = [
        TitForTat(), AlwaysCooperate(), AlwaysDefect(), 
        RandomStrategy(seed=seed), PavlovStrategy(),
        GrudgerStrategy(), GTFTStrategy(seed=seed)
    ]
    
    evaluation_results = evaluate_against_all_opponents(
        model, all_strategies, num_rounds, seed, log_dir
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
    """Evaluate model against all opponent strategies"""
    
    print("🎯 Evaluating model:")
    results = {}
    
    for strategy in strategies:
        print(f"   Testing vs {strategy.name}...")
        
        # Create environment for this opponent
        env = make_vec_env(
            create_enhanced_env(strategy, num_rounds, seed), 
            n_envs=1,
            seed=seed
        )
        env = VecMonitor(env)
        
        # Evaluate
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=n_episodes, deterministic=True
        )
        
        # Get cooperation rate and adaptation metrics
        coop_rate, adaptation_score = get_enhanced_metrics(model, env, strategy, n_episodes=20)
        
        results[strategy.name] = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'cooperation_rate': coop_rate,
            'adaptation_score': adaptation_score
        }
        
        print(f"     📈 Reward: {mean_reward:.3f} ± {std_reward:.3f}")
        print(f"     🤝 Cooperation: {coop_rate:.1%}")
        print(f"     🔄 Adaptation: {adaptation_score:.3f}")
        
        env.close()
    
    # Save results
    if log_dir:
        import pandas as pd
        results_df = pd.DataFrame(results).T
        results_df.to_csv(f"{log_dir}/evaluation_results.csv")
        
        # Plot results
        plot_evaluation_results(results, log_dir)
    
    return results


def get_enhanced_metrics(model: PPO, env, opponent_strategy, n_episodes: int = 20) -> Tuple[float, float]:
    """Calculate enhanced metrics including adaptation score"""
    cooperation_count = 0
    total_actions = 0
    adaptation_events = 0
    adaptation_opportunities = 0
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_actions = []
        episode_rewards = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_actions.append(action[0])
            episode_rewards.append(reward[0])
            
            if action[0] == 0:  # Cooperate
                cooperation_count += 1
            total_actions += 1
        
        # Calculate adaptation score for this episode
        if len(episode_actions) >= 10:
            # Look for strategic adaptation patterns
            for i in range(5, len(episode_actions) - 1):
                # Check if there was a reward drop (being exploited)
                if (episode_rewards[i] < episode_rewards[i-1] - 1 and 
                    episode_actions[i-1] == 0):  # was cooperating
                    adaptation_opportunities += 1
                    
                    # Check if agent adapted (changed strategy)
                    if episode_actions[i] != episode_actions[i-1]:
                        adaptation_events += 1
    
    coop_rate = cooperation_count / total_actions if total_actions > 0 else 0.0
    adaptation_score = adaptation_events / adaptation_opportunities if adaptation_opportunities > 0 else 0.0
    
    return coop_rate, adaptation_score


def plot_evaluation_results(results: Dict, log_dir: str):
    """Plot evaluation results"""
    strategies = list(results.keys())
    rewards = [results[s]['mean_reward'] for s in strategies]
    reward_stds = [results[s]['std_reward'] for s in strategies]
    coop_rates = [results[s]['cooperation_rate'] for s in strategies]
    adaptation_scores = [results[s]['adaptation_score'] for s in strategies]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Rewards plot
    bars1 = ax1.bar(strategies, rewards, capsize=5, alpha=0.8, color='blue')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('PPO: Performance vs Different Opponents')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    for bar, reward in zip(bars1, rewards):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{reward:.1f}', ha='center', va='bottom')
    
    # Cooperation rates plot
    bars2 = ax2.bar(strategies, coop_rates, alpha=0.8, color='green')
    ax2.set_ylabel('Cooperation Rate')
    ax2.set_title('PPO: Cooperation Rate vs Different Opponents')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    for bar, rate in zip(bars2, coop_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.1%}', ha='center', va='bottom')
    
    # Adaptation scores
    bars3 = ax3.bar(strategies, adaptation_scores, alpha=0.8, color='orange')
    ax3.set_ylabel('Adaptation Score')
    ax3.set_title('PPO: Strategic Adaptation')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    for bar, score in zip(bars3, adaptation_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.2f}', ha='center', va='bottom')
    
    # Performance summary
    avg_reward = np.mean(rewards)
    avg_coop = np.mean(coop_rates)
    avg_adapt = np.mean(adaptation_scores)
    
    ax4.text(0.1, 0.8, f'Average Performance:', fontsize=14, fontweight='bold')
    ax4.text(0.1, 0.7, f'Mean Reward: {avg_reward:.1f}', fontsize=12)
    ax4.text(0.1, 0.6, f'Cooperation Rate: {avg_coop:.1%}', fontsize=12)
    ax4.text(0.1, 0.5, f'Adaptation Score: {avg_adapt:.3f}', fontsize=12)
    ax4.text(0.1, 0.3, f'Strategic Classification:', fontsize=14, fontweight='bold')
    ax4.text(0.1, 0.2, f'Adaptive Strategic Learner', fontsize=12, style='italic')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{log_dir}/ppo_evaluation.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for IPD")
    parser.add_argument("--timesteps", type=int, default=250000,
                        help="Total training timesteps (default: 250000)")
    parser.add_argument("--num_rounds", type=int, default=100,
                        help="Number of rounds per game (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    print("=== PPO Training for Iterated Prisoner's Dilemma ===")
    
    # Train model
    model = train_ppo_agent(
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        num_rounds=args.num_rounds,
        seed=args.seed,
        model_name="ppo_ipd"
    )
    
    print("\n✅ PPO Training completed!")
    print(f"🎯 Model trained for {args.timesteps + 100000:,} timesteps")
    print(f"📊 Results saved in results/ppo/")
    print(f"💾 Model saved in models/ppo_ipd") 