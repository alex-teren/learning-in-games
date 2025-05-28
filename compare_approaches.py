#!/usr/bin/env python3
"""
Comprehensive comparison of PPO, Evolution, and Transformer approaches for IPD
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from env import IPDEnv, Strategy, TitForTat, AlwaysCooperate, AlwaysDefect, RandomStrategy, PavlovStrategy


def run_training_approach(
    approach: str, 
    args: argparse.Namespace,
    force_retrain: bool = False
) -> Tuple[float, Dict]:
    """
    Run training for a specific approach
    
    Args:
        approach: 'ppo', 'evolution', or 'transformer'
        args: Command line arguments
        force_retrain: Whether to force retraining even if model exists
        
    Returns:
        Training time and results dictionary
    """
    print(f"\n{'='*60}")
    print(f"🚀 Running {approach.upper()} Training")
    print(f"{'='*60}")
    
    repo_root = Path(__file__).resolve().parent
    
    # Check if model already exists
    model_paths = {
        'ppo': repo_root / "models" / "ppo_single_opponent.zip",
        'evolution': repo_root / "models" / "evolved_strategy.pkl", 
        'transformer': repo_root / "models" / "transformer_best.pth"
    }
    
    results_paths = {
        'ppo': repo_root / "results" / "ppo" / "evaluation_results.csv",
        'evolution': repo_root / "results" / "evolution" / "evaluation_results.csv",
        'transformer': repo_root / "results" / "transformer" / "evaluation_results.csv"
    }
    
    if not force_retrain and model_paths[approach].exists():
        print(f"✅ {approach.upper()} model already exists, skipping training...")
        
        # Try to load existing results
        if results_paths[approach].exists():
            results_df = pd.read_csv(results_paths[approach])
            results = results_df.to_dict('index')
            return 0.0, results
        else:
            print(f"⚠️  Results not found, will need to retrain or evaluate...")
    
    # Set up training command
    training_scripts = {
        'ppo': repo_root / "agents" / "ppo" / "train_ppo.py",
        'evolution': repo_root / "agents" / "evolution" / "train_evolution.py", 
        'transformer': repo_root / "agents" / "transformer" / "train_transformer.py"
    }
    
    cmd = [sys.executable, str(training_scripts[approach])]
    
    # Add approach-specific arguments
    if approach == 'ppo':
        cmd.extend([
            "--timesteps", str(args.timesteps),
            "--num_rounds", str(args.num_rounds),
            "--learning_rate", str(args.learning_rate),
            "--seed", str(args.seed)
        ])
        if args.fast:
            cmd.extend(["--timesteps", "50000"])
    
    elif approach == 'evolution':
        cmd.extend([
            "--generations", str(args.generations),
            "--population_size", str(args.population_size),
            "--num_games", str(args.num_games),
            "--num_rounds", str(args.num_rounds),
            "--seed", str(args.seed)
        ])
        if args.fast:
            cmd.extend(["--generations", "50", "--population_size", "30"])
    
    elif approach == 'transformer':
        cmd.extend([
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--num_games", str(args.num_games_transformer),
            "--num_rounds", str(args.num_rounds),
            "--learning_rate", str(args.learning_rate),
            "--seed", str(args.seed)
        ])
        if args.fast:
            cmd.append("--fast")
    
    # Run training
    start_time = time.time()
    
    try:
        print(f"🏃 Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Print output for debugging
        if result.stdout:
            print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
        if result.stderr:
            print("STDERR:", result.stderr[-1000:])
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for {approach}")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return -1, {}
    
    training_time = time.time() - start_time
    
    print(f"✅ {approach.upper()} training completed in {training_time:.1f} seconds")
    
    # Try to load results
    if results_paths[approach].exists():
        results_df = pd.read_csv(results_paths[approach])
        results = results_df.to_dict('index')
        return training_time, results
    else:
        print(f"⚠️  No results file found for {approach}")
        return training_time, {}


def load_results_from_files() -> Dict:
    """Load results from existing files if available"""
    repo_root = Path(__file__).resolve().parent
    
    results = {}
    
    for approach in ['ppo', 'evolution', 'transformer']:
        results_file = repo_root / "results" / approach / "evaluation_results.csv"
        
        if results_file.exists():
            try:
                df = pd.read_csv(results_file)
                if 'mean_reward' in df.columns or 'mean_score' in df.columns:
                    # PPO format
                    if 'mean_reward' in df.columns:
                        approach_results = {}
                        for _, row in df.iterrows():
                            strategy_name = row.iloc[0]  # First column is strategy name
                            approach_results[strategy_name] = {
                                'mean_score': row.get('mean_reward', 0),
                                'cooperation_rate': row.get('cooperation_rate', 0)
                            }
                    # Evolution/Transformer format  
                    else:
                        approach_results = {}
                        for _, row in df.iterrows():
                            strategy_name = row.iloc[0]  # First column is strategy name
                            approach_results[strategy_name] = {
                                'mean_score': row.get('mean_score', 0),
                                'cooperation_rate': row.get('mean_cooperation_rate', 0)
                            }
                    
                    results[approach] = approach_results
                    print(f"✅ Loaded {approach} results: {len(approach_results)} strategies")
                else:
                    print(f"⚠️  Unknown format in {approach} results file")
            except Exception as e:
                print(f"❌ Error loading {approach} results: {e}")
        else:
            print(f"📁 No results file found for {approach}")
    
    return results


def create_comparison_plots(results: Dict, output_dir: str) -> None:
    """Create comprehensive comparison plots"""
    
    print("\n📊 Creating comparison plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all strategies that appear in results
    all_strategies = set()
    for approach_results in results.values():
        all_strategies.update(approach_results.keys())
    
    strategies = sorted(list(all_strategies))
    approaches = list(results.keys())
    
    if not strategies or not approaches:
        print("⚠️  No data available for plotting")
        return
    
    # Prepare data for plotting
    scores_data = []
    coop_data = []
    
    for approach in approaches:
        approach_scores = []
        approach_coop = []
        
        for strategy in strategies:
            if strategy in results[approach]:
                approach_scores.append(results[approach][strategy]['mean_score'])
                approach_coop.append(results[approach][strategy]['cooperation_rate'])
            else:
                approach_scores.append(0)
                approach_coop.append(0)
        
        scores_data.append(approach_scores)
        coop_data.append(approach_coop)
    
    # Create comprehensive comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Performance comparison (bar plot)
    x = np.arange(len(strategies))
    width = 0.25
    
    colors = ['#2E8B57', '#4169E1', '#DC143C']  # Green, Blue, Red
    
    for i, (approach, scores) in enumerate(zip(approaches, scores_data)):
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, scores, width, label=approach.upper(), 
                      color=colors[i % len(colors)], alpha=0.8)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            if score > 0:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{score:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Opponent Strategy')
    ax1.set_ylabel('Mean Score')
    ax1.set_title('Performance Comparison Across Approaches')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cooperation rates (bar plot)
    for i, (approach, coop_rates) in enumerate(zip(approaches, coop_data)):
        offset = (i - 1) * width
        bars = ax2.bar(x + offset, coop_rates, width, label=approach.upper(),
                      color=colors[i % len(colors)], alpha=0.8)
        
        # Add percentage labels
        for bar, rate in zip(bars, coop_rates):
            if rate > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{rate:.0%}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_xlabel('Opponent Strategy')
    ax2.set_ylabel('Cooperation Rate')
    ax2.set_title('Cooperation Rate Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # 3. Overall performance (radar plot)
    if len(strategies) >= 3:
        angles = np.linspace(0, 2 * np.pi, len(strategies), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, (approach, scores) in enumerate(zip(approaches, scores_data)):
            values = scores + scores[:1]  # Complete the circle
            ax3.plot(angles, values, 'o-', linewidth=2, label=approach.upper(),
                    color=colors[i % len(colors)])
            ax3.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(strategies)
        ax3.set_title('Performance Radar Chart')
        ax3.legend()
        ax3.grid(True)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data\nfor radar chart', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Performance Radar Chart')
    
    # 4. Summary statistics
    summary_data = []
    for approach in approaches:
        approach_scores = [results[approach][s]['mean_score'] 
                          for s in strategies if s in results[approach]]
        if approach_scores:
            summary_data.append({
                'Approach': approach.upper(),
                'Mean Score': np.mean(approach_scores),
                'Max Score': np.max(approach_scores),
                'Min Score': np.min(approach_scores),
                'Std Score': np.std(approach_scores)
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Create table
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=summary_df.round(2).values,
                         colLabels=summary_df.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comprehensive_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual comparison plots
    create_individual_plots(results, output_dir)
    
    print(f"📊 Plots saved to {output_dir}/")


def create_individual_plots(results: Dict, output_dir: str) -> None:
    """Create individual comparison plots"""
    
    strategies = ['Tit-for-Tat', 'Always Cooperate', 'Always Defect', 'Random(p=0.5)', 'Pavlov']
    approaches = list(results.keys())
    
    # Performance heatmap
    plt.figure(figsize=(10, 6))
    
    heatmap_data = []
    for approach in approaches:
        row = []
        for strategy in strategies:
            if strategy in results[approach]:
                row.append(results[approach][strategy]['mean_score'])
            else:
                row.append(0)
        heatmap_data.append(row)
    
    heatmap_data = np.array(heatmap_data)
    
    im = plt.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, label='Mean Score')
    
    plt.xticks(range(len(strategies)), strategies, rotation=45, ha='right')
    plt.yticks(range(len(approaches)), [a.upper() for a in approaches])
    
    # Add text annotations
    for i in range(len(approaches)):
        for j in range(len(strategies)):
            text = plt.text(j, i, f'{heatmap_data[i, j]:.1f}',
                           ha="center", va="center", color="black", fontweight="bold")
    
    plt.title('Performance Heatmap: Mean Scores vs Opponents')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_comparison_results(results: Dict, training_times: Dict, output_dir: str) -> None:
    """Save comparison results to files"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comprehensive results DataFrame
    all_data = []
    
    for approach, approach_results in results.items():
        for strategy, metrics in approach_results.items():
            all_data.append({
                'Approach': approach.upper(),
                'Opponent': strategy,
                'Mean_Score': metrics['mean_score'],
                'Cooperation_Rate': metrics['cooperation_rate'],
                'Training_Time': training_times.get(approach, 0)
            })
    
    if all_data:
        results_df = pd.DataFrame(all_data)
        results_df.to_csv(f"{output_dir}/comprehensive_results.csv", index=False)
        
        # Create pivot tables
        score_pivot = results_df.pivot(index='Opponent', columns='Approach', values='Mean_Score')
        coop_pivot = results_df.pivot(index='Opponent', columns='Approach', values='Cooperation_Rate')
        
        score_pivot.to_csv(f"{output_dir}/scores_comparison.csv")
        coop_pivot.to_csv(f"{output_dir}/cooperation_comparison.csv")
        
        # Summary statistics
        summary_stats = results_df.groupby('Approach').agg({
            'Mean_Score': ['mean', 'std', 'min', 'max'],
            'Cooperation_Rate': ['mean', 'std'],
            'Training_Time': 'first'
        }).round(3)
        
        summary_stats.to_csv(f"{output_dir}/summary_statistics.csv")
        
        print(f"📊 Results saved to {output_dir}/")
        print(f"   - comprehensive_results.csv")
        print(f"   - scores_comparison.csv") 
        print(f"   - cooperation_comparison.csv")
        print(f"   - summary_statistics.csv")


def print_summary(results: Dict, training_times: Dict) -> None:
    """Print a comprehensive summary"""
    
    print(f"\n{'='*60}")
    print("🏆 FINAL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    if not results:
        print("❌ No results available for comparison")
        return
    
    # Training times
    print("\n⏱️  Training Times:")
    for approach, time_taken in training_times.items():
        if time_taken > 0:
            if time_taken < 60:
                print(f"   {approach.upper():12}: {time_taken:.1f} seconds")
            else:
                print(f"   {approach.upper():12}: {time_taken/60:.1f} minutes ({time_taken:.0f}s)")
        else:
            print(f"   {approach.upper():12}: (already trained)")
    
    # Performance summary
    print("\n📊 Performance Summary:")
    
    strategies = ['Tit-for-Tat', 'Always Cooperate', 'Always Defect', 'Random(p=0.5)', 'Pavlov']
    
    for strategy in strategies:
        print(f"\n   vs {strategy}:")
        
        strategy_results = []
        for approach in results.keys():
            if strategy in results[approach]:
                score = results[approach][strategy]['mean_score']
                coop = results[approach][strategy]['cooperation_rate']
                strategy_results.append((approach, score, coop))
        
        # Sort by score
        strategy_results.sort(key=lambda x: x[1], reverse=True)
        
        for i, (approach, score, coop) in enumerate(strategy_results):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"      {medal} {approach.upper():12}: Score {score:5.1f}, Coop {coop:5.1%}")
    
    # Overall winner
    print(f"\n🏆 Overall Performance Rankings:")
    
    overall_scores = {}
    for approach in results.keys():
        scores = [results[approach][s]['mean_score'] 
                 for s in strategies if s in results[approach]]
        if scores:
            overall_scores[approach] = np.mean(scores)
    
    ranked_approaches = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (approach, avg_score) in enumerate(ranked_approaches):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        print(f"   {medal} {approach.upper():12}: Average Score {avg_score:.2f}")
    
    # Cooperation analysis
    print(f"\n🤝 Cooperation Analysis:")
    
    overall_coop = {}
    for approach in results.keys():
        coop_rates = [results[approach][s]['cooperation_rate'] 
                     for s in strategies if s in results[approach]]
        if coop_rates:
            overall_coop[approach] = np.mean(coop_rates)
    
    ranked_coop = sorted(overall_coop.items(), key=lambda x: x[1], reverse=True)
    
    for approach, avg_coop in ranked_coop:
        print(f"   {approach.upper():12}: {avg_coop:.1%} average cooperation")


def main():
    parser = argparse.ArgumentParser(description="Compare PPO, Evolution, and Transformer approaches for IPD")
    
    # General parameters
    parser.add_argument("--approaches", nargs='+', choices=['ppo', 'evolution', 'transformer', 'all'],
                        default=['all'], help="Which approaches to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force_retrain", action="store_true", 
                        help="Force retraining even if models exist")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and only compare existing results")
    parser.add_argument("--fast", action="store_true",
                        help="Use fast training settings for all approaches")
    
    # Shared parameters
    parser.add_argument("--num_rounds", type=int, default=100, help="Rounds per game")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    
    # PPO specific
    parser.add_argument("--timesteps", type=int, default=100000, help="PPO timesteps")
    
    # Evolution specific  
    parser.add_argument("--generations", type=int, default=100, help="Evolution generations")
    parser.add_argument("--population_size", type=int, default=50, help="Population size")
    parser.add_argument("--num_games", type=int, default=30, help="Games per opponent (evolution)")
    
    # Transformer specific
    parser.add_argument("--epochs", type=int, default=25, help="Transformer epochs")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size")
    parser.add_argument("--num_games_transformer", type=int, default=2500, 
                        help="Games for transformer dataset")
    
    args = parser.parse_args()
    
    print("🎯 Comprehensive IPD Approaches Comparison")
    print(f"{'='*60}")
    
    # Determine which approaches to run
    if 'all' in args.approaches:
        approaches = ['ppo', 'evolution', 'transformer']
    else:
        approaches = args.approaches
    
    print(f"🚀 Running approaches: {', '.join(a.upper() for a in approaches)}")
    
    if args.fast:
        print("⚡ Fast mode enabled - using reduced training parameters")
    
    # Results storage
    results = {}
    training_times = {}
    
    if not args.skip_training:
        # Run each approach
        for approach in approaches:
            try:
                training_time, approach_results = run_training_approach(
                    approach, args, args.force_retrain
                )
                
                if approach_results:
                    results[approach] = approach_results
                    training_times[approach] = training_time
                    print(f"✅ {approach.upper()} completed successfully")
                else:
                    print(f"⚠️  {approach.upper()} completed but no results available")
                    
            except Exception as e:
                print(f"❌ Error running {approach}: {e}")
                continue
    
    # Load any existing results if we don't have them
    if not results or args.skip_training:
        print(f"\n📂 Loading existing results...")
        existing_results = load_results_from_files()
        
        # Merge with any new results
        for approach, approach_results in existing_results.items():
            if approach not in results:
                results[approach] = approach_results
                training_times[approach] = 0  # Mark as existing
    
    if not results:
        print("❌ No results available for comparison!")
        return
    
    # Create output directory
    output_dir = Path(__file__).resolve().parent / "comparison_results"
    
    # Generate comparison plots and analysis
    create_comparison_plots(results, str(output_dir))
    save_comparison_results(results, training_times, str(output_dir))
    print_summary(results, training_times)
    
    print(f"\n✅ Comparison completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Check comprehensive_comparison.png for visual summary")


if __name__ == "__main__":
    main() 