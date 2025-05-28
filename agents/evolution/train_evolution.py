import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from typing import List, Tuple, Dict, Any, Optional
import time
from tqdm import tqdm
import argparse
from pathlib import Path
import cma

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from env import IPDEnv, Strategy, TitForTat, AlwaysCooperate, AlwaysDefect, RandomStrategy, PavlovStrategy, GrudgerStrategy, GTFTStrategy, simulate_match


def save_plot_and_csv(x, y, name: str, folder: str = "results"):
    """Save PNG plot and matching CSV"""
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame({"x": x, "y": y}).to_csv(f"{folder}/{name}_data.csv", index=False)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linewidth=2)
    plt.title(name.replace("_", " ").title())
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{folder}/{name}.png", dpi=120, bbox_inches="tight")
    plt.close()


class MemoryOneStrategy(Strategy):
    """
    Memory-one strategy for IPD - responds based on last round outcome
    
    Parameters control probability of cooperation given last round:
    - p_cc: Probability to cooperate after mutual cooperation (CC)
    - p_cd: Probability to cooperate after I cooperated, opponent defected (CD)  
    - p_dc: Probability to cooperate after I defected, opponent cooperated (DC)
    - p_dd: Probability to cooperate after mutual defection (DD)
    - initial_action_prob: Probability to cooperate on first move
    """
    
    def __init__(self, params: np.ndarray, name: str = "Memory-One"):
        super().__init__(name)
        self.p_cc = max(0, min(1, params[0]))  # Clamp to [0, 1]
        self.p_cd = max(0, min(1, params[1]))
        self.p_dc = max(0, min(1, params[2]))
        self.p_dd = max(0, min(1, params[3]))
        self.initial_action_prob = max(0, min(1, params[4]))
        self.params = params.copy()
        
        # Update name to include parameters for identification
        self.name = f"Memory-One(cc={self.p_cc:.2f},cd={self.p_cd:.2f},dc={self.p_dc:.2f},dd={self.p_dd:.2f},init={self.initial_action_prob:.2f})"
    
    def action(self, history: List[Tuple[int, int]], player_idx: int = 0) -> int:
        if not history:  # First move
            return 0 if np.random.random() < self.initial_action_prob else 1
        
        # Get last round outcome
        my_last_action = history[-1][player_idx]
        opponent_last_action = history[-1][1 - player_idx]
        
        # Determine cooperation probability based on last round
        if my_last_action == 0 and opponent_last_action == 0:  # CC
            coop_prob = self.p_cc
        elif my_last_action == 0 and opponent_last_action == 1:  # CD  
            coop_prob = self.p_cd
        elif my_last_action == 1 and opponent_last_action == 0:  # DC
            coop_prob = self.p_dc
        else:  # DD
            coop_prob = self.p_dd
        
        return 0 if np.random.random() < coop_prob else 1


def evaluate_fitness(
    params: np.ndarray,
    opponent_strategies: List[Strategy],
    num_games_per_opponent: int = 50,
    num_rounds: int = 100,
    seed: Optional[int] = None
) -> float:
    """
    Evaluate fitness of a memory-one strategy against multiple opponents
    
    Args:
        params: Strategy parameters [p_cc, p_cd, p_dc, p_dd, initial_action_prob]
        opponent_strategies: List of opponent strategies to test against
        num_games_per_opponent: Number of games per opponent
        num_rounds: Number of rounds per game
        seed: Random seed for reproducibility
        
    Returns:
        Average fitness score across all opponents
    """
    if seed is not None:
        np.random.seed(seed)
    
    env = IPDEnv(num_rounds=num_rounds, seed=seed)
    strategy = MemoryOneStrategy(params)
    
    total_score = 0
    total_games = 0
    
    for opponent in opponent_strategies:
        opponent_scores = []
        
        for game in range(num_games_per_opponent):
            # Simulate match
            match_results = simulate_match(env, strategy, opponent, num_rounds)
            opponent_scores.append(match_results['player_score'])
        
        # Average score against this opponent
        avg_score = np.mean(opponent_scores)
        total_score += avg_score
        total_games += 1
    
    # Return average score across all opponents
    return total_score / total_games if total_games > 0 else 0.0


def run_cmaes_evolution(
    opponent_strategies: List[Strategy],
    generations: int = 100,
    population_size: int = 50,
    sigma: float = 0.3,
    seed: int = 42,
    num_games_per_opponent: int = 30,
    num_rounds: int = 100,
    save_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, List[float], Dict]:
    """
    Run CMA-ES evolution to find optimal memory-one strategy
    
    Args:
        opponent_strategies: List of strategies to evolve against
        generations: Number of generations to run
        population_size: Population size for CMA-ES
        sigma: Initial standard deviation for CMA-ES
        seed: Random seed
        num_games_per_opponent: Games per opponent for fitness evaluation
        num_rounds: Rounds per game
        save_dir: Directory to save models
        log_dir: Directory to save logs
        verbose: Whether to print progress
        
    Returns:
        best_params, fitness_history, results_dict
    """
    
    print(f"🧬 Running CMA-ES Evolution for {generations} generations")
    print(f"📊 Population size: {population_size}")
    print(f"🎯 Testing against {len(opponent_strategies)} opponents:")
    for strategy in opponent_strategies:
        print(f"   - {strategy.name}")
    
    # Set up directories
    repo_root = Path(__file__).resolve().parents[2]
    if save_dir is None:
        save_dir = repo_root / "models"
    if log_dir is None:
        log_dir = repo_root / "results" / "evolution"
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set random seed
    np.random.seed(seed)
    
    # Initialize CMA-ES
    # Parameters: [p_cc, p_cd, p_dc, p_dd, initial_action_prob]
    initial_guess = [0.5, 0.5, 0.5, 0.5, 0.5]  # Start with neutral strategy
    
    # CMA-ES options
    opts = {
        'seed': seed,
        'popsize': population_size,
        'maxiter': generations,
        'verb_disp': 1 if verbose else 0,
        'verb_log': 0,
        'bounds': [0, 1]  # All parameters should be probabilities
    }
    
    # Initialize CMA-ES optimizer
    es = cma.CMAEvolutionStrategy(initial_guess, sigma, opts)
    
    # Track evolution progress
    fitness_history = []
    best_fitness_history = []
    generation_times = []
    best_params = None
    best_fitness = float('-inf')
    
    print(f"\n🚀 Starting evolution...")
    start_time = time.time()
    
    generation = 0
    while not es.stop():
        generation += 1
        gen_start_time = time.time()
        
        # Generate candidate solutions
        solutions = es.ask()
        
        # Evaluate fitness for each solution
        fitness_values = []
        
        if verbose:
            solution_iter = tqdm(solutions, desc=f"Gen {generation:3d}")
        else:
            solution_iter = solutions
        
        for params in solution_iter:
            fitness = evaluate_fitness(
                params, 
                opponent_strategies,
                num_games_per_opponent,
                num_rounds,
                seed + generation  # Different seed per generation
            )
            fitness_values.append(fitness)
            
            # Track best solution
            if fitness > best_fitness:
                best_fitness = fitness
                best_params = params.copy()
        
        # Update CMA-ES with fitness values (CMA-ES minimizes, so negate fitness)
        es.tell(solutions, [-f for f in fitness_values])
        
        # Record statistics
        gen_fitness = np.mean(fitness_values)
        fitness_history.append(gen_fitness)
        best_fitness_history.append(best_fitness)
        
        gen_time = time.time() - gen_start_time
        generation_times.append(gen_time)
        
        if verbose:
            print(f"Generation {generation:3d}: avg_fitness={gen_fitness:.3f}, "
                  f"best_fitness={best_fitness:.3f}, time={gen_time:.1f}s")
    
    total_time = time.time() - start_time
    
    print(f"\n✅ Evolution completed!")
    print(f"🎯 Best fitness: {best_fitness:.3f}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"📊 Generations: {generation}")
    
    # Create best strategy
    best_strategy = MemoryOneStrategy(best_params, "Evolved-Best")
    
    # Detailed evaluation of best strategy
    print(f"\n🏆 Best evolved strategy parameters:")
    print(f"   p_cc (cooperate after CC): {best_params[0]:.3f}")
    print(f"   p_cd (cooperate after CD): {best_params[1]:.3f}")
    print(f"   p_dc (cooperate after DC): {best_params[2]:.3f}")
    print(f"   p_dd (cooperate after DD): {best_params[3]:.3f}")
    print(f"   initial_coop_prob:         {best_params[4]:.3f}")
    
    # Save results
    results_dict = {
        'best_params': best_params,
        'best_fitness': best_fitness,
        'fitness_history': fitness_history,
        'best_fitness_history': best_fitness_history,
        'generation_times': generation_times,
        'total_time': total_time,
        'generations': generation,
        'opponents': [s.name for s in opponent_strategies],
        'settings': {
            'population_size': population_size,
            'sigma': sigma,
            'num_games_per_opponent': num_games_per_opponent,
            'num_rounds': num_rounds,
            'seed': seed
        }
    }
    
    # Save evolution results
    save_evolution_results(results_dict, save_dir, log_dir)
    
    # Plot evolution progress
    plot_evolution_progress(fitness_history, best_fitness_history, log_dir)
    
    return best_params, fitness_history, results_dict


def save_evolution_results(results_dict: Dict, save_dir: str, log_dir: str):
    """Save evolution results to files"""
    
    # Save best strategy parameters
    with open(f"{save_dir}/evolved_strategy.pkl", "wb") as f:
        pickle.dump(results_dict, f)
    
    # Save fitness history as CSV
    history_df = pd.DataFrame({
        'generation': range(1, len(results_dict['fitness_history']) + 1),
        'avg_fitness': results_dict['fitness_history'],
        'best_fitness': results_dict['best_fitness_history']
    })
    history_df.to_csv(f"{log_dir}/evolution_history.csv", index=False)
    
    # Save best parameters as CSV
    params_df = pd.DataFrame({
        'parameter': ['p_cc', 'p_cd', 'p_dc', 'p_dd', 'initial_coop_prob'],
        'value': results_dict['best_params']
    })
    params_df.to_csv(f"{log_dir}/best_parameters.csv", index=False)
    
    print(f"💾 Results saved:")
    print(f"   Model: {save_dir}/evolved_strategy.pkl")
    print(f"   History: {log_dir}/evolution_history.csv")
    print(f"   Parameters: {log_dir}/best_parameters.csv")


def plot_evolution_progress(fitness_history: List[float], best_fitness_history: List[float], log_dir: str):
    """Plot evolution progress"""
    generations = list(range(1, len(fitness_history) + 1))
    
    plt.figure(figsize=(12, 5))
    
    # Fitness over generations
    plt.subplot(1, 2, 1)
    plt.plot(generations, fitness_history, 'b-', label='Average Fitness', alpha=0.7)
    plt.plot(generations, best_fitness_history, 'r-', label='Best Fitness', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Improvement over time
    plt.subplot(1, 2, 2)
    improvement = np.array(best_fitness_history) - best_fitness_history[0]
    plt.plot(generations, improvement, 'g-', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Fitness Improvement')
    plt.title('Cumulative Improvement')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{log_dir}/evolution_progress.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save individual plots
    save_plot_and_csv(generations, fitness_history, "evolution_avg_fitness", folder=log_dir)
    save_plot_and_csv(generations, best_fitness_history, "evolution_best_fitness", folder=log_dir)


def evaluate_strategy(
    strategy: MemoryOneStrategy,
    opponent_strategies: List[Strategy],
    num_games: int = 100,
    num_rounds: int = 100,
    seed: int = 42
) -> Dict:
    """Evaluate a strategy against multiple opponents"""
    
    print(f"🎯 Evaluating strategy: {strategy.name}")
    
    env = IPDEnv(num_rounds=num_rounds, seed=seed)
    results = {}
    
    for opponent in opponent_strategies:
        print(f"   vs {opponent.name}...")
        
        scores = []
        cooperation_rates = []
        
        for game in range(num_games):
            match_results = simulate_match(env, strategy, opponent, num_rounds)
            scores.append(match_results['player_score'])
            
            # Calculate cooperation rate
            player_actions = [step['player_action'] for step in match_results['history']]
            coop_rate = player_actions.count(0) / len(player_actions)
            cooperation_rates.append(coop_rate)
        
        results[opponent.name] = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'mean_cooperation_rate': np.mean(cooperation_rates),
            'std_cooperation_rate': np.std(cooperation_rates)
        }
        
        print(f"      Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
        print(f"      Cooperation: {np.mean(cooperation_rates):.2%} ± {np.std(cooperation_rates):.2%}")
    
    return results


def load_evolved_strategy(model_path: str) -> MemoryOneStrategy:
    """Load evolved strategy from file"""
    with open(model_path, 'rb') as f:
        results_dict = pickle.load(f)
    
    best_params = results_dict['best_params']
    strategy = MemoryOneStrategy(best_params, "Evolved-Loaded")
    
    print(f"📂 Loaded evolved strategy:")
    print(f"   Best fitness: {results_dict['best_fitness']:.3f}")
    print(f"   Parameters: {best_params}")
    
    return strategy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolve memory-one strategies for IPD")
    parser.add_argument("--generations", type=int, default=100,
                        help="Number of generations (default: 100)")
    parser.add_argument("--population_size", type=int, default=50,
                        help="Population size (default: 50)")
    parser.add_argument("--sigma", type=float, default=0.3,
                        help="Initial standard deviation for CMA-ES (default: 0.3)")
    parser.add_argument("--num_games", type=int, default=30,
                        help="Number of games per opponent for evaluation (default: 30)")
    parser.add_argument("--num_rounds", type=int, default=100,
                        help="Number of rounds per game (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--opponents", type=str, nargs='+',
                        choices=['all', 'tit_for_tat', 'always_cooperate', 'always_defect', 'random', 'pavlov', 'grudger', 'gtft'],
                        default=['all'],
                        help="Opponent strategies to evolve against")
    
    args = parser.parse_args()
    
    print("=== Evolutionary Strategy Training for IPD ===")
    
    # Set up opponent strategies
    if 'all' in args.opponents:
        opponent_strategies = [
            TitForTat(),
            AlwaysCooperate(), 
            AlwaysDefect(),
            RandomStrategy(seed=args.seed),
            PavlovStrategy(),
            GrudgerStrategy(),
            GTFTStrategy(seed=args.seed)
        ]
    else:
        opponent_map = {
            'tit_for_tat': TitForTat(),
            'always_cooperate': AlwaysCooperate(),
            'always_defect': AlwaysDefect(),
            'random': RandomStrategy(seed=args.seed),
            'pavlov': PavlovStrategy(),
            'grudger': GrudgerStrategy(),
            'gtft': GTFTStrategy(seed=args.seed)
        }
        opponent_strategies = [opponent_map[name] for name in args.opponents]
    
    # Run evolution
    best_params, fitness_history, results = run_cmaes_evolution(
        opponent_strategies=opponent_strategies,
        generations=args.generations,
        population_size=args.population_size,
        sigma=args.sigma,
        seed=args.seed,
        num_games_per_opponent=args.num_games,
        num_rounds=args.num_rounds
    )
    
    # Create and evaluate final strategy
    final_strategy = MemoryOneStrategy(best_params, "Evolved-Final")
    
    # Comprehensive evaluation
    print(f"\n📊 Comprehensive evaluation:")
    all_opponents = [
        TitForTat(), AlwaysCooperate(), AlwaysDefect(), 
        RandomStrategy(seed=args.seed+1000), PavlovStrategy(),
        GrudgerStrategy(), GTFTStrategy(seed=args.seed+1000)
    ]
    
    evaluation_results = evaluate_strategy(
        final_strategy, 
        all_opponents,
        num_games=100,
        num_rounds=args.num_rounds,
        seed=args.seed
    )
    
    print("\n✅ Evolution completed!")
    print(f"🎯 Best fitness achieved: {results['best_fitness']:.3f}")
    print(f"📊 Results saved in results/evolution/")
    print(f"💾 Model saved in models/evolved_strategy.pkl") 