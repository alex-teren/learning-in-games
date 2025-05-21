import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import cma
from typing import List, Tuple, Dict, Any, Optional
import pickle
from pathlib import Path

# Add project root to path to allow imports from other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from env import IPDEnv, Strategy, TitForTat, AlwaysCooperate, AlwaysDefect, RandomStrategy, simulate_match


def save_plot_and_csv(x, y, name: str, folder: str = "results"):
    """Save PNG plot **and** matching CSV so LLM can analyse the numbers."""
    import os, pandas as pd, matplotlib.pyplot as plt
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame({"x": x, "y": y}).to_csv(f"{folder}/{name}_data.csv", index=False)
    plt.figure(); plt.plot(x, y); plt.title(name.replace("_", " ").title())
    plt.savefig(f"{folder}/{name}.png", dpi=120, bbox_inches="tight"); plt.close()


class MemoryOneStrategy(Strategy):
    """
    Memory-one strategy for the Iterated Prisoner's Dilemma.
    
    A memory-one strategy is defined by:
    - initial_action: The first action to take
    - p_cc: Probability of cooperating if both players cooperated in the previous round
    - p_cd: Probability of cooperating if the player cooperated and opponent defected
    - p_dc: Probability of cooperating if the player defected and opponent cooperated
    - p_dd: Probability of cooperating if both players defected in the previous round
    """
    
    def __init__(self, params: List[float], name: str = "MemoryOne"):
        """
        Initialize a memory-one strategy with given parameters
        
        Args:
            params: List of probabilities [p_cc, p_cd, p_dc, p_dd, initial_action_prob]
            name: Strategy name
        """
        super().__init__(name)
        
        # Ensure parameters are valid probabilities
        self.params = np.clip(params, 0.0, 1.0)
        
        # Extract parameters
        self.p_cc = self.params[0]  # Prob. of cooperating after both players cooperated
        self.p_cd = self.params[1]  # Prob. of cooperating after player cooperated, opponent defected
        self.p_dc = self.params[2]  # Prob. of cooperating after player defected, opponent cooperated
        self.p_dd = self.params[3]  # Prob. of cooperating after both players defected
        self.initial_action_prob = self.params[4]  # Prob. of cooperating on first move
        
        # Initialize random number generator
        self.rng = np.random.RandomState()
    
    def action(self, history: List[Tuple[int, int]], player_idx: int = 0) -> int:
        """
        Determine next action based on game history
        
        Args:
            history: List of tuples (player_action, opponent_action) for each past round
            player_idx: Index of the player using this strategy (0 or 1)
            
        Returns:
            int: 0 for Cooperate, 1 for Defect
        """
        # First move - use initial action probability
        if not history:
            return 0 if self.rng.random() < self.initial_action_prob else 1
        
        # Get opponent index
        opponent_idx = 1 - player_idx
        
        # Get last actions
        last_player_action = history[-1][player_idx]
        last_opponent_action = history[-1][opponent_idx]
        
        # Determine cooperation probability based on previous actions
        if last_player_action == 0 and last_opponent_action == 0:  # CC
            coop_prob = self.p_cc
        elif last_player_action == 0 and last_opponent_action == 1:  # CD
            coop_prob = self.p_cd
        elif last_player_action == 1 and last_opponent_action == 0:  # DC
            coop_prob = self.p_dc
        else:  # DD
            coop_prob = self.p_dd
        
        # Return action based on probability
        return 0 if self.rng.random() < coop_prob else 1
    
    @classmethod
    def from_params(cls, params: List[float]) -> 'MemoryOneStrategy':
        """
        Create a MemoryOneStrategy from parameters
        
        Args:
            params: List of probabilities [p_cc, p_cd, p_dc, p_dd, initial_action_prob]
            
        Returns:
            MemoryOneStrategy instance
        """
        return cls(params)
    
    def __str__(self) -> str:
        """String representation of the strategy"""
        return (f"{self.name}: "
                f"p_cc={self.p_cc:.2f}, "
                f"p_cd={self.p_cd:.2f}, "
                f"p_dc={self.p_dc:.2f}, "
                f"p_dd={self.p_dd:.2f}, "
                f"init={self.initial_action_prob:.2f}")


def evaluate_fitness(
    params: List[float],
    env: IPDEnv,
    opponent_strategies: Dict[str, Strategy],
    num_rounds: int = 100,
    num_matches: int = 5
) -> float:
    """
    Evaluate the fitness of a strategy defined by parameters
    
    Args:
        params: Strategy parameters (for MemoryOneStrategy)
        env: IPD environment for evaluation
        opponent_strategies: Dictionary of opponent strategies to play against
        num_rounds: Number of rounds per match
        num_matches: Number of matches per opponent
        
    Returns:
        Fitness score (average reward across all matches)
    """
    # Create memory-one strategy from parameters
    strategy = MemoryOneStrategy(params)
    
    total_rewards = []
    
    # Play against each opponent multiple times
    for opponent_name, opponent in opponent_strategies.items():
        for _ in range(num_matches):
            # Simulate match against opponent
            results = simulate_match(env, strategy, opponent, num_rounds)
            total_rewards.append(results['player_score'])
    
    # Return average reward (higher is better)
    return np.mean(total_rewards)


def run_cmaes_evolution(
    population_size: int = 10,
    num_generations: int = 50,
    sigma0: float = 0.5,
    num_rounds: int = 100,
    opponent_strategies: Optional[Dict[str, Strategy]] = None,
    seed: int = 42,
    save_dir: Optional[str] = None,
    log_dir: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run CMA-ES optimization to evolve a memory-one IPD strategy
    
    Args:
        population_size: Population size for CMA-ES
        num_generations: Number of generations to run
        sigma0: Initial step size
        num_rounds: Number of rounds per match in evaluation
        opponent_strategies: Opponent strategies to evaluate against
        seed: Random seed
        save_dir: Directory to save evolved strategies
        log_dir: Directory to save logs and plots
        
    Returns:
        Best parameters found and evolution history
    """
    print("Starting CMA-ES evolution of IPD strategy...")
    
    # Get repo root and set default paths if not provided
    repo_root = Path(__file__).resolve().parents[2]
    if save_dir is None:
        save_dir = repo_root / "models"
    if log_dir is None:
        log_dir = repo_root / "results"
    
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/evolution", exist_ok=True)
    
    # Set random seed
    np.random.seed(seed)
    
    # Initialize IPD environment
    env = IPDEnv(num_rounds=num_rounds, seed=seed)
    
    # Set opponent strategies if not provided
    if opponent_strategies is None:
        opponent_strategies = {
            "tit_for_tat": TitForTat(),
            "always_cooperate": AlwaysCooperate(),
            "always_defect": AlwaysDefect(),
            "random": RandomStrategy(seed=seed+100)
        }
    
    # Initialize CMA-ES optimizer
    # We optimize 5 parameters: [p_cc, p_cd, p_dc, p_dd, initial_action_prob]
    
    # Initial guess for parameters (TFT-like strategy)
    initial_mean = np.array([0.99, 0.01, 0.99, 0.01, 0.99])
    
    # Initialize optimizer
    es = cma.CMAEvolutionStrategy(
        initial_mean,
        sigma0,
        {'popsize': population_size, 'seed': seed}
    )
    
    # Setup for tracking progress
    history = {
        'generation': [],
        'best_fitness': [],
        'best_params': [],
        'avg_fitness': [],
        'std_fitness': []
    }
    
    # Main evolution loop
    start_time = time.time()
    
    for generation in range(num_generations):
        # Sample population
        solutions = es.ask()
        
        # Evaluate fitness for each individual
        fitnesses = []
        for params in solutions:
            # Clip parameters to valid range [0, 1]
            params_clipped = np.clip(params, 0, 1)
            
            # Get fitness (negative because CMA-ES minimizes)
            fitness = -evaluate_fitness(params_clipped, env, opponent_strategies, num_rounds)
            fitnesses.append(fitness)
        
        # Update CMA-ES with evaluated solutions
        es.tell(solutions, fitnesses)
        
        # Get best solution in this generation
        best_idx = np.argmin(fitnesses)
        best_fitness = -fitnesses[best_idx]
        best_params = np.clip(solutions[best_idx], 0, 1)
        
        # Log progress
        history['generation'].append(generation)
        history['best_fitness'].append(best_fitness)
        history['best_params'].append(best_params)
        history['avg_fitness'].append(-np.mean(fitnesses))
        history['std_fitness'].append(np.std(fitnesses))
        
        # Print progress
        if generation % 5 == 0 or generation == num_generations - 1:
            strategy = MemoryOneStrategy(best_params)
            elapsed_time = time.time() - start_time
            print(f"Gen {generation+1}/{num_generations} | "
                  f"Best Fitness: {best_fitness:.2f} | "
                  f"Avg Fitness: {-np.mean(fitnesses):.2f} | "
                  f"Time: {elapsed_time:.1f}s")
            print(f"Best Strategy: {strategy}")
            print("-" * 80)
    
    # Get final best solution
    best_params = es.result.xbest
    best_params = np.clip(best_params, 0, 1)  # Ensure valid probabilities
    
    # Save results
    save_evolution_results(best_params, history, save_dir, log_dir)
    
    return best_params, history


def save_evolution_results(
    best_params: np.ndarray,
    history: Dict[str, List],
    save_dir: Optional[str] = None,
    log_dir: Optional[str] = None
) -> None:
    """
    Save evolution results and generate plots
    
    Args:
        best_params: Best strategy parameters
        history: Evolution history
        save_dir: Directory to save evolved strategies
        log_dir: Directory to save logs and plots
    """
    # Get repo root and set default paths if not provided
    repo_root = Path(__file__).resolve().parents[2]
    if save_dir is None:
        save_dir = repo_root / "models"
    if log_dir is None:
        log_dir = repo_root / "results"
    
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/evolution", exist_ok=True)
    
    # Save best strategy
    best_strategy = MemoryOneStrategy(best_params)
    
    # Save as numpy array
    np.save(f"{save_dir}/evolved_strategy_params.npy", best_params)
    
    # Save as pickle for easy loading
    with open(f"{save_dir}/evolved_strategy.pkl", 'wb') as f:
        pickle.dump(best_strategy, f)
    
    # Save parameters as text file for reference
    with open(f"{save_dir}/evolved_strategy_params.txt", 'w') as f:
        f.write(f"# Evolved Memory-One Strategy Parameters\n")
        f.write(f"p_cc = {best_params[0]:.6f}\n")
        f.write(f"p_cd = {best_params[1]:.6f}\n")
        f.write(f"p_dc = {best_params[2]:.6f}\n")
        f.write(f"p_dd = {best_params[3]:.6f}\n")
        f.write(f"initial_action_prob = {best_params[4]:.6f}\n")
    
    # Create history dataframe
    history_df = pd.DataFrame({
        'Generation': history['generation'],
        'Best Fitness': history['best_fitness'],
        'Average Fitness': history['avg_fitness'],
        'Std Fitness': history['std_fitness']
    })
    
    # Save history to CSV
    history_df.to_csv(f"{log_dir}/evolution/evolution_history.csv", index=False)
    
    # Save best fitness plot using helper
    save_plot_and_csv(
        history['generation'],
        history['best_fitness'],
        "evolution_best_fitness",
        folder=f"{log_dir}/evolution"
    )
    
    # Save average fitness plot using helper
    save_plot_and_csv(
        history['generation'],
        history['avg_fitness'],
        "evolution_avg_fitness",
        folder=f"{log_dir}/evolution"
    )
    
    print(f"Best evolved strategy saved to {save_dir}/evolved_strategy.pkl")
    print(f"Evolution history saved to {log_dir}/evolution/evolution_history.csv")
    print(f"Fitness curves saved to {log_dir}/evolution/")


def evaluate_strategy(
    strategy: Strategy,
    opponent_strategies: Dict[str, Strategy],
    num_rounds: int = 100,
    num_matches: int = 10,
    seed: int = 42,
    log_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Evaluate a strategy against various opponents
    
    Args:
        strategy: Strategy to evaluate
        opponent_strategies: Dictionary of opponent strategies
        num_rounds: Number of rounds per match
        num_matches: Number of matches per opponent
        seed: Random seed
        log_dir: Directory to save results
        
    Returns:
        DataFrame with evaluation results
    """
    print(f"Evaluating {strategy.name} strategy against different opponents...")
    
    # Get repo root and set default path if not provided
    if log_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        log_dir = repo_root / "results"
    
    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/evolution", exist_ok=True)
    
    # Create environment
    env = IPDEnv(num_rounds=num_rounds, seed=seed)
    
    # Results container
    results_list = []
    
    # Evaluate against each opponent
    for opponent_name, opponent in opponent_strategies.items():
        matches_results = []
        
        for match in range(num_matches):
            # Simulate match
            match_results = simulate_match(env, strategy, opponent, num_rounds)
            matches_results.append({
                'match': match + 1,
                'player_score': match_results['player_score'],
                'opponent_score': match_results['opponent_score'],
                'player_coop_rate': match_results['cooperation_rate_player'],
                'opponent_coop_rate': match_results['cooperation_rate_opponent']
            })
        
        # Calculate average results across matches
        avg_player_score = np.mean([r['player_score'] for r in matches_results])
        avg_opponent_score = np.mean([r['opponent_score'] for r in matches_results])
        avg_player_coop = np.mean([r['player_coop_rate'] for r in matches_results])
        avg_opponent_coop = np.mean([r['opponent_coop_rate'] for r in matches_results])
        
        results_list.append({
            'opponent': opponent_name,
            'avg_player_score': avg_player_score,
            'avg_opponent_score': avg_opponent_score,
            'avg_player_coop_rate': avg_player_coop,
            'avg_opponent_coop_rate': avg_opponent_coop
        })
    
    # Create and save results dataframe
    results_df = pd.DataFrame(results_list)
    
    results_df.to_csv(f"{log_dir}/evolution/evaluation_results.csv", index=False)
    
    print("Evaluation results:")
    print(results_df)
    
    return results_df


def load_evolved_strategy(model_path: Optional[str] = None) -> Strategy:
    """
    Load a saved evolved strategy
    
    Args:
        model_path: Path to saved strategy
        
    Returns:
        Loaded strategy
    """
    # Get repo root and set default path if not provided
    if model_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        model_path = repo_root / "models" / "evolved_strategy.pkl"
    
    with open(model_path, 'rb') as f:
        strategy = pickle.load(f)
    
    return strategy


if __name__ == "__main__":
    # Set parameters for evolution
    population_size = 20
    num_generations = 50
    num_rounds = 100
    
    # Run CMA-ES evolution
    best_params, history = run_cmaes_evolution(
        population_size=population_size,
        num_generations=num_generations,
        num_rounds=num_rounds,
        seed=42
    )
    
    # Create the best evolved strategy
    best_strategy = MemoryOneStrategy(best_params, name="EvolvedStrategy")
    
    # Define opponent strategies
    opponent_strategies = {
        "tit_for_tat": TitForTat(),
        "always_cooperate": AlwaysCooperate(),
        "always_defect": AlwaysDefect(),
        "random": RandomStrategy(seed=42)
    }
    
    # Evaluate the evolved strategy against different opponents
    evaluation_results = evaluate_strategy(
        best_strategy,
        opponent_strategies,
        num_rounds=100,
        num_matches=20,
        seed=42
    )
    
    print(f"Evolution completed successfully!")
    print(f"Best evolved strategy: {best_strategy}") 