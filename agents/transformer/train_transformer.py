import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict, Any, Optional
import pickle
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
from datetime import timedelta

# Add project root to path to allow imports from other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from env import IPDEnv, Strategy, TitForTat, AlwaysCooperate, AlwaysDefect, RandomStrategy, simulate_match, PavlovStrategy, GrudgerStrategy, GTFTStrategy


def save_plot_and_csv(x, y, name: str, folder: str = "results"):
    """Save PNG plot and matching CSV"""
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


def generate_trajectory_dataset(
    num_games: int = 2500,
    num_rounds: int = 100,
    seed: int = 42,
    log_dir: Optional[str] = None
) -> List[Dict]:
    """
    Generate IPD trajectory dataset for Decision Transformer training
    
    Args:
        num_games: Number of IPD games to generate
        num_rounds: Number of rounds per game
        seed: Random seed
        log_dir: Directory to save dataset
        
    Returns:
        List of trajectory dictionaries
    """
    print(f"Generating trajectory dataset of {num_games} IPD games with {num_rounds} rounds each...")
    
    if log_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        log_dir = repo_root / "results"
    
    np.random.seed(seed)
    random.seed(seed)
    
    env = IPDEnv(num_rounds=num_rounds, seed=seed)
    
    # Diverse strategies for better learning
    strategies = [
        TitForTat(), AlwaysCooperate(), AlwaysDefect(), 
        RandomStrategy(seed=seed+100), PavlovStrategy(),
        GrudgerStrategy(), GTFTStrategy(seed=seed+200)
    ]
    
    trajectories = []
    
    for game_idx in tqdm(range(num_games), desc="Generating games"):
        # Sample two strategies
        player_strategy = random.choice(strategies)
        opponent_strategy = random.choice(strategies)
        
        # Simulate match
        match_results = simulate_match(env, player_strategy, opponent_strategy, num_rounds)
        
        # Extract data
        actions = [step['player_action'] for step in match_results['history']]
        rewards = [step['player_payoff'] for step in match_results['history']]
        opponent_actions = [step['opponent_action'] for step in match_results['history']]
        
        # Process into trajectory
        trajectory = {
            'player_strategy': player_strategy.name,
            'opponent_strategy': opponent_strategy.name,
            'actions': actions,
            'rewards': rewards,
            'opponent_actions': opponent_actions,
            'player_score': match_results['player_score'],
            'opponent_score': match_results['opponent_score']
        }
        
        # Calculate returns-to-go
        returns = []
        R = 0
        for r in reversed(trajectory['rewards']):
            R += r
            returns.insert(0, R)
        trajectory['returns_to_go'] = returns
        
        trajectories.append(trajectory)
    
    # Save dataset
    os.makedirs(f"{log_dir}/transformer", exist_ok=True)
    with open(f"{log_dir}/transformer/dataset.pkl", "wb") as f:
        pickle.dump(trajectories, f)
    
    print(f"Generated {len(trajectories)} game trajectories")
    return trajectories


class IPDDataset(Dataset):
    """Dataset for Decision Transformer training on IPD trajectories"""
    
    def __init__(self, trajectories: List[Dict], context_length: int = 7):
        self.trajectories = trajectories
        self.context_length = context_length
        self.samples = self._create_samples()
        print(f"Created {len(self.samples)} training samples")
    
    def _create_samples(self) -> List[Dict]:
        samples = []
        
        # Balanced sampling rate for quality vs speed
        sampling_rate = 2
        
        for traj in self.trajectories:
            traj_length = len(traj['actions'])
            
            if traj_length < self.context_length:
                continue
            
            # Create samples from trajectory
            for t in range(self.context_length, traj_length, sampling_rate):
                context_start = t - self.context_length
                
                # Context data
                player_actions = traj['actions'][context_start:t]
                opponent_actions = traj['opponent_actions'][context_start:t]
                returns_to_go = traj['returns_to_go'][context_start:t]
                
                target_action = traj['actions'][t]
                
                sample = {
                    'player_actions': player_actions,
                    'opponent_actions': opponent_actions,
                    'returns_to_go': returns_to_go,
                    'target_action': target_action,
                    'strategy_names': (traj['player_strategy'], traj['opponent_strategy'])
                }
                
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        return {
            'player_actions': torch.tensor(sample['player_actions'], dtype=torch.long),
            'opponent_actions': torch.tensor(sample['opponent_actions'], dtype=torch.long),
            'returns_to_go': torch.tensor(sample['returns_to_go'], dtype=torch.float),
            'target_action': torch.tensor(sample['target_action'], dtype=torch.long)
        }


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for learning IPD strategies
    Optimized for balance between quality and training speed
    """
    
    def __init__(
        self,
        action_dim: int = 2,
        hidden_size: int = 192,
        max_rtg: float = 100.0,
        n_heads: int = 6,
        n_layers: int = 4,
        dropout: float = 0.15,
        context_length: int = 7
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.max_rtg = max_rtg
        self.context_length = context_length
        
        # Improved embeddings
        self.player_action_embedding = nn.Embedding(action_dim, hidden_size)
        self.opponent_action_embedding = nn.Embedding(action_dim, hidden_size)
        self.rtg_embedding = nn.Linear(1, hidden_size)
        
        # Position embeddings
        self.position_embedding = nn.Embedding(context_length, hidden_size)
        
        # Layer normalization
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        # Transformer architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 3,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Enhanced output head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Improved weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, player_actions, opponent_actions, returns_to_go):
        batch_size, seq_length = player_actions.shape
        
        # Normalize returns to go
        rtg_normalized = (returns_to_go.unsqueeze(-1) / self.max_rtg).clamp(-1, 1)
        
        # Embed inputs
        player_embeds = self.player_action_embedding(player_actions)
        opponent_embeds = self.opponent_action_embedding(opponent_actions)
        rtg_embeds = self.rtg_embedding(rtg_normalized)
        
        # Weighted combination of embeddings
        combined_embeds = (
            0.4 * player_embeds + 
            0.4 * opponent_embeds + 
            0.2 * rtg_embeds
        )
        
        # Add positional embeddings
        positions = torch.arange(seq_length, device=player_actions.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        
        sequence = combined_embeds + pos_embeds
        sequence = self.embed_ln(sequence)
        
        # Pass through transformer
        transformer_output = self.transformer(sequence)
        
        # Get action predictions
        action_logits = self.action_head(transformer_output)
        
        return action_logits


def train_transformer(
    trajectories: List[Dict],
    epochs: int = 25,
    batch_size: int = 96,
    lr: float = 5e-4,
    weight_decay: float = 0.01,
    save_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    device: str = None
) -> Tuple[DecisionTransformer, Dict]:
    """
    Train Decision Transformer on IPD trajectories
    
    Args:
        trajectories: List of trajectory dictionaries
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        weight_decay: L2 regularization weight
        save_dir: Directory to save models
        log_dir: Directory to save logs
        device: Training device ('cuda' or 'cpu')
        
    Returns:
        Trained model and training history
    """
    print("🤖 Training Decision Transformer for IPD...")
    
    repo_root = Path(__file__).resolve().parents[2]
    if save_dir is None:
        save_dir = repo_root / "models"
    if log_dir is None:
        log_dir = repo_root / "results" / "transformer"
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = IPDDataset(trajectories, context_length=7)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Create model
    model = DecisionTransformer()
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            player_actions = batch['player_actions'].to(device)
            opponent_actions = batch['opponent_actions'].to(device)
            returns_to_go = batch['returns_to_go'].to(device)
            target_actions = batch['target_action'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            action_logits = model(player_actions, opponent_actions, returns_to_go)
            
            # Use last timestep prediction
            loss = criterion(action_logits[:, -1, :], target_actions)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(action_logits[:, -1, :], 1)
            train_total += target_actions.size(0)
            train_correct += (predicted == target_actions).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                player_actions = batch['player_actions'].to(device)
                opponent_actions = batch['opponent_actions'].to(device)
                returns_to_go = batch['returns_to_go'].to(device)
                target_actions = batch['target_action'].to(device)
                
                action_logits = model(player_actions, opponent_actions, returns_to_go)
                loss = criterion(action_logits[:, -1, :], target_actions)
                
                val_loss += loss.item()
                _, predicted = torch.max(action_logits[:, -1, :], 1)
                val_total += target_actions.size(0)
                val_correct += (predicted == target_actions).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Calculate ETA
        epoch_time = time.time() - epoch_start
        remaining_epochs = epochs - (epoch + 1)
        eta = timedelta(seconds=remaining_epochs * epoch_time)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.1e} | Epoch Time: {epoch_time:.1f}s | ETA: {eta}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/transformer_best.pth")
            print(f"  💾 New best model saved! Val Acc: {val_acc:.4f}")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    
    # Save final model and history
    torch.save(model.state_dict(), f"{save_dir}/transformer_final.pth")
    
    history_df = pd.DataFrame(history)
    history_df.to_csv(f"{log_dir}/training_history.csv", index=False)
    
    # Plot results
    plot_training_history(history, log_dir)
    
    return model, history


def plot_training_history(history: Dict, log_dir: str) -> None:
    """Plot detailed training history"""
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='red', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], label='Train Accuracy', color='blue', linewidth=2)
    ax2.plot(epochs, history['val_acc'], label='Val Accuracy', color='red', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate plot
    ax3.plot(epochs, history['lr'], label='Learning Rate', color='green', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Loss difference plot (generalization gap)
    gap = [v - t for v, t in zip(history['val_loss'], history['train_loss'])]
    ax4.plot(epochs, gap, label='Val - Train Loss', color='purple', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Difference')
    ax4.set_title('Generalization Gap')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{log_dir}/training_history.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save individual plots
    save_plot_and_csv(epochs, history['train_loss'], "transformer_train_loss", folder=log_dir)
    save_plot_and_csv(epochs, history['val_loss'], "transformer_val_loss", folder=log_dir)
    save_plot_and_csv(epochs, history['train_acc'], "transformer_train_acc", folder=log_dir)
    save_plot_and_csv(epochs, history['val_acc'], "transformer_val_acc", folder=log_dir)


def evaluate_transformer(
    model: DecisionTransformer,
    strategies: List[Strategy],
    num_rounds: int = 100,
    num_episodes: int = 50,
    device: str = 'cpu',
    log_dir: Optional[str] = None
) -> Dict:
    """Evaluate transformer against different strategies"""
    
    print("🎯 Evaluating transformer against opponents:")
    model.eval()
    results = {}
    
    for strategy in strategies:
        print(f"   vs {strategy.name}...")
        
        scores = []
        cooperation_rates = []
        
        for episode in range(num_episodes):
            env = IPDEnv(num_rounds=num_rounds, seed=episode)
            
            # Initialize history
            history = []
            episode_rewards = []
            actions_taken = []
            
            for round_num in range(num_rounds):
                if len(history) < 7:  # Context length
                    # Random action for initial rounds
                    action = np.random.choice([0, 1])
                else:
                    # Use transformer to predict action
                    context_start = len(history) - 7
                    player_actions = [h[0] for h in history[context_start:]]
                    opponent_actions = [h[1] for h in history[context_start:]]
                    
                    # Calculate returns-to-go (simplified)
                    recent_rewards = episode_rewards[context_start:]
                    rtg = [sum(recent_rewards[i:]) for i in range(len(recent_rewards))]
                    
                    # Prepare input tensors
                    player_tensor = torch.tensor([player_actions], dtype=torch.long).to(device)
                    opponent_tensor = torch.tensor([opponent_actions], dtype=torch.long).to(device)
                    rtg_tensor = torch.tensor([rtg], dtype=torch.float).to(device)
                    
                    with torch.no_grad():
                        logits = model(player_tensor, opponent_tensor, rtg_tensor)
                        probs = torch.softmax(logits[0, -1], dim=0)
                        action = torch.multinomial(probs, 1).item()
                
                # Get opponent action
                opponent_action = strategy.action(history, player_idx=1)
                
                # Calculate reward (IPD payoff matrix)
                if action == 0 and opponent_action == 0:  # Both cooperate
                    reward = 3
                elif action == 0 and opponent_action == 1:  # Player cooperates, opponent defects
                    reward = 0
                elif action == 1 and opponent_action == 0:  # Player defects, opponent cooperates
                    reward = 5
                else:  # Both defect
                    reward = 1
                
                # Update history
                history.append((action, opponent_action))
                episode_rewards.append(reward)
                actions_taken.append(action)
            
            # Calculate episode metrics
            total_score = sum(episode_rewards)
            cooperation_rate = actions_taken.count(0) / len(actions_taken)
            
            scores.append(total_score)
            cooperation_rates.append(cooperation_rate)
        
        # Store results
        results[strategy.name] = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'mean_cooperation_rate': np.mean(cooperation_rates),
            'std_cooperation_rate': np.std(cooperation_rates)
        }
        
        print(f"      Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
        print(f"      Cooperation: {np.mean(cooperation_rates):.2%} ± {np.std(cooperation_rates):.2%}")
    
    # Save results
    if log_dir:
        results_df = pd.DataFrame(results).T
        results_df.to_csv(f"{log_dir}/evaluation_results.csv")
        plot_evaluation_results(results, log_dir)
    
    return results


def plot_evaluation_results(results: Dict, log_dir: str):
    """Plot evaluation results"""
    strategies = list(results.keys())
    scores = [results[s]['mean_score'] for s in strategies]
    score_stds = [results[s]['std_score'] for s in strategies]
    coop_rates = [results[s]['mean_cooperation_rate'] for s in strategies]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scores plot
    bars1 = ax1.bar(strategies, scores, yerr=score_stds, capsize=5, alpha=0.8)
    ax1.set_ylabel('Mean Score')
    ax1.set_title('Transformer Performance vs Different Opponents')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    for bar, score in zip(bars1, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{score:.1f}', ha='center', va='bottom')
    
    # Cooperation rates plot
    bars2 = ax2.bar(strategies, coop_rates, alpha=0.8, color='green')
    ax2.set_ylabel('Cooperation Rate')
    ax2.set_title('Transformer Cooperation Rate vs Different Opponents')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    for bar, rate in zip(bars2, coop_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{log_dir}/transformer_evaluation.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Decision Transformer for IPD")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs (default: 25)")
    parser.add_argument("--batch_size", type=int, default=96,
                        help="Batch size (default: 96)")
    parser.add_argument("--num_games", type=int, default=2500,
                        help="Number of games for dataset (default: 2500)")
    parser.add_argument("--num_rounds", type=int, default=100,
                        help="Number of rounds per game (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="Learning rate (default: 5e-4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--fast", action="store_true",
                        help="Use fast training settings")
    
    args = parser.parse_args()
    
    # Adjust parameters for fast training
    if args.fast:
        args.num_games = 1000
        args.epochs = 15
        print("🚀 Fast training mode enabled")
    
    print("=== Decision Transformer Training for IPD ===")
    print(f"📊 Dataset: {args.num_games} games, {args.num_rounds} rounds each")
    print(f"🎯 Training: {args.epochs} epochs, batch size {args.batch_size}")
    
    # Generate dataset
    trajectories = generate_trajectory_dataset(
        num_games=args.num_games,
        num_rounds=args.num_rounds,
        seed=args.seed
    )
    
    # Train model
    model, history = train_transformer(
        trajectories=trajectories,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate
    )
    
    # Evaluate model
    print("\n📊 Comprehensive evaluation:")
    evaluation_strategies = [
        TitForTat(), AlwaysCooperate(), AlwaysDefect(), 
        RandomStrategy(seed=args.seed), PavlovStrategy(),
        GrudgerStrategy(), GTFTStrategy(seed=args.seed)
    ]
    
    evaluation_results = evaluate_transformer(
        model=model,
        strategies=evaluation_strategies,
        num_rounds=args.num_rounds,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n✅ Transformer training completed!")
    print(f"🎯 Best validation accuracy: {max(history['val_acc']):.4f}")
    print(f"📊 Results saved in results/transformer/")
    print(f"💾 Model saved in models/transformer_best.pth") 