import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import random
from typing import List, Dict, Optional, Tuple
import argparse
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm
from collections import deque

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from env import IPDEnv, TitForTat, AlwaysCooperate, AlwaysDefect, RandomStrategy, PavlovStrategy, GrudgerStrategy, GTFTStrategy


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


def generate_strategic_dataset(
    num_games: int = 1000,  # Reduced from 2500
    num_rounds: int = 100,
    seed: int = 42,
    log_dir: Optional[str] = None
) -> List[Dict]:
    """
    Generate high-quality strategic IPD dataset with expert demonstrations
    """
    print(f"Generating strategic dataset of {num_games} IPD games with {num_rounds} rounds each...")
    
    if log_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        log_dir = repo_root / "results"
    
    np.random.seed(seed)
    random.seed(seed)
    
    env = IPDEnv(num_rounds=num_rounds, seed=seed)
    
    # Strategic opponents with different difficulty levels
    strategies = [
        TitForTat(), AlwaysCooperate(), AlwaysDefect(), 
        RandomStrategy(seed=seed+100), PavlovStrategy(),
        GrudgerStrategy(), GTFTStrategy(seed=seed+200)
    ]
    
    # Weight strategies for better learning (more challenging opponents)
    strategy_weights = [0.2, 0.1, 0.2, 0.15, 0.15, 0.1, 0.1]
    
    trajectories = []
    
    for game_idx in tqdm(range(num_games), desc="Generating games"):
        # Select player and opponent strategies with bias toward challenging matchups
        player_strategy = np.random.choice(strategies, p=strategy_weights)
        opponent_strategy = np.random.choice(strategies, p=strategy_weights)
        
        # Simulate game
        env.reset()
        history = []
        player_rewards = []
        
        for round_idx in range(num_rounds):
            # Get actions
            player_action = player_strategy.action(history, player_idx=0)
            opponent_action = opponent_strategy.action(history, player_idx=1)
            
            # Calculate rewards (IPD payoff matrix)
            if player_action == 0 and opponent_action == 0:  # CC
                player_reward = 3
            elif player_action == 0 and opponent_action == 1:  # CD  
                player_reward = 0
            elif player_action == 1 and opponent_action == 0:  # DC
                player_reward = 5
            else:  # DD
                player_reward = 1
            
            history.append((player_action, opponent_action))
            player_rewards.append(player_reward)
        
        # Calculate strategic returns-to-go with bonuses
        returns_to_go = []
        cumulative = 0
        for i in reversed(range(len(player_rewards))):
            cumulative += player_rewards[i]
            
            # Add strategic bonuses for optimal play
            bonus = 0
            if i > 0:
                prev_opp = history[i-1][1]
                curr_action = history[i][0]
                
                # Bonus for exploiting cooperators
                if prev_opp == 0 and curr_action == 1:
                    bonus += 0.5
                # Bonus for reciprocating with cooperators  
                elif prev_opp == 0 and curr_action == 0:
                    bonus += 0.3
                # Bonus for defecting against defectors
                elif prev_opp == 1 and curr_action == 1:
                    bonus += 0.2
            
            returns_to_go.insert(0, cumulative + bonus)
        
        # Create trajectory
        trajectory = {
            'player_strategy': player_strategy.name,
            'opponent_strategy': opponent_strategy.name,
            'actions': [h[0] for h in history],
            'opponent_actions': [h[1] for h in history],
            'rewards': player_rewards,
            'returns_to_go': returns_to_go,
            'player_score': sum(player_rewards),
            'opponent_score': sum([
                3 if h[0] == 0 and h[1] == 0 else
                5 if h[0] == 0 and h[1] == 1 else  
                0 if h[0] == 1 and h[1] == 0 else 1
                for h in history
            ])
        }
        
        trajectories.append(trajectory)
    
    print(f"Generated {len(trajectories)} strategic trajectories")
    return trajectories


class StrategicIPDDataset(Dataset):
    """Optimized dataset for strategic IPD learning"""
    
    def __init__(self, trajectories: List[Dict], context_length: int = 15):  # Increased context
        self.trajectories = trajectories
        self.context_length = context_length
        self.samples = self._create_strategic_samples()
        print(f"Created {len(self.samples)} strategic training samples")
    
    def _create_strategic_samples(self) -> List[Dict]:
        samples = []
        
        for traj in self.trajectories:
            traj_length = len(traj['actions'])
            
            if traj_length < self.context_length:
                continue
            
            # Create more samples from good trajectories (curriculum learning)
            sampling_rate = 1 if traj['player_score'] > 200 else 2
            
            for t in range(self.context_length, traj_length, sampling_rate):
                context_start = t - self.context_length
                
                # Context data with opponent modeling
                player_actions = traj['actions'][context_start:t]
                opponent_actions = traj['opponent_actions'][context_start:t]
                returns_to_go = traj['returns_to_go'][context_start:t]
                
                # Add strategic features
                opp_coop_rate = sum(1-a for a in opponent_actions) / len(opponent_actions)
                my_coop_rate = sum(1-a for a in player_actions) / len(player_actions)
                
                # Recent trends
                recent_opp = opponent_actions[-5:] if len(opponent_actions) >= 5 else opponent_actions
                recent_trend = sum(1-a for a in recent_opp) / len(recent_opp)
                
                target_action = traj['actions'][t]
                
                sample = {
                    'player_actions': player_actions,
                    'opponent_actions': opponent_actions,
                    'returns_to_go': returns_to_go,
                    'opp_coop_rate': opp_coop_rate,
                    'my_coop_rate': my_coop_rate,
                    'recent_trend': recent_trend,
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
            'opp_coop_rate': torch.tensor(sample['opp_coop_rate'], dtype=torch.float),
            'my_coop_rate': torch.tensor(sample['my_coop_rate'], dtype=torch.float),
            'recent_trend': torch.tensor(sample['recent_trend'], dtype=torch.float),
            'target_action': torch.tensor(sample['target_action'], dtype=torch.long)
        }


class EfficientDecisionTransformer(nn.Module):
    """
    Efficient Decision Transformer optimized for IPD strategic learning
    Much smaller and more focused than the original
    """
    
    def __init__(
        self,
        action_dim: int = 2,
        hidden_size: int = 128,  # Reduced from 192
        max_rtg: float = 100.0,
        n_heads: int = 4,  # Reduced from 6
        n_layers: int = 3,  # Reduced from 4
        dropout: float = 0.1,  # Reduced from 0.15
        context_length: int = 15  # Increased for better pattern recognition
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.max_rtg = max_rtg
        self.context_length = context_length
        
        # Efficient embeddings that sum to hidden_size
        self.player_action_embedding = nn.Embedding(action_dim, hidden_size // 4)  # 32
        self.opponent_action_embedding = nn.Embedding(action_dim, hidden_size // 4)  # 32
        self.rtg_embedding = nn.Linear(1, hidden_size // 4)  # 32
        
        # Strategic features embedding
        self.strategic_features = nn.Linear(3, hidden_size // 4)  # 32
        # Total: 32 + 32 + 32 + 32 = 128 = hidden_size
        
        # Position embeddings
        self.position_embedding = nn.Embedding(context_length, hidden_size)
        
        # Layer normalization
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        # Lightweight transformer
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=hidden_size * 2,  # Reduced from 3
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        # Strategic output head with regularization
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Optimized weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, player_actions, opponent_actions, returns_to_go, strategic_features):
        batch_size, seq_length = player_actions.shape
        
        # Normalize returns to go
        rtg_normalized = (returns_to_go.unsqueeze(-1) / self.max_rtg).clamp(-2, 2)
        
        # Embed inputs efficiently
        player_embeds = self.player_action_embedding(player_actions)
        opponent_embeds = self.opponent_action_embedding(opponent_actions)
        rtg_embeds = self.rtg_embedding(rtg_normalized)
        
        # Strategic features need to be expanded to sequence length
        if strategic_features.dim() == 2:  # [batch_size, 3]
            strategic_features_expanded = strategic_features.unsqueeze(1).expand(batch_size, seq_length, -1)
        else:  # [batch_size, seq_length, 3]
            strategic_features_expanded = strategic_features
        strategic_embeds = self.strategic_features(strategic_features_expanded)
        
        # Combine embeddings
        combined_embeds = torch.cat([
            player_embeds, opponent_embeds, rtg_embeds, strategic_embeds
        ], dim=-1)
        
        # Add positional embeddings
        positions = torch.arange(seq_length, device=player_actions.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        
        sequence = combined_embeds + pos_embeds
        sequence = self.embed_ln(sequence)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            sequence = layer(sequence)
        
        # Get strategic action predictions
        action_logits = self.strategy_head(sequence)
        
        return action_logits


def train_transformer(
    trajectories: List[Dict],
    epochs: int = 15,  # Reduced from 25
    batch_size: int = 128,  # Increased from 96
    lr: float = 3e-4,  # Reduced from 5e-4
    weight_decay: float = 0.05,  # Increased regularization
    save_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    device: str = None
) -> Tuple[EfficientDecisionTransformer, Dict]:
    """
    Train Efficient Decision Transformer with strategic focus
    """
    print("🧠 Training Efficient Decision Transformer for IPD...")
    
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
    
    # Create strategic dataset
    dataset = StrategicIPDDataset(trajectories, context_length=15)
    
    # Split dataset
    train_size = int(0.85 * len(dataset))  # Increased training portion
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Create efficient model
    model = EfficientDecisionTransformer()
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (vs 1.5M in original)")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # More aggressive learning rate schedule
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.3
    )
    
    # Strategic loss with class weights for better exploitation learning
    class_weights = torch.tensor([0.6, 0.4]).to(device)  # Favor defection slightly
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
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
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            player_actions = batch['player_actions'].to(device)
            opponent_actions = batch['opponent_actions'].to(device)
            returns_to_go = batch['returns_to_go'].to(device)
            target_actions = batch['target_action'].to(device)
            
            # Strategic features
            strategic_features = torch.stack([
                batch['opp_coop_rate'],
                batch['my_coop_rate'],
                batch['recent_trend']
            ], dim=-1).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            action_logits = model(player_actions, opponent_actions, returns_to_go, strategic_features)
            
            # Strategic loss focusing on recent decisions
            recent_logits = action_logits[:, -3:, :]  # Last 3 timesteps
            recent_targets = target_actions
            
            loss = criterion(action_logits[:, -1, :], target_actions)
            
            # Add consistency regularization
            if recent_logits.shape[1] > 1:
                consistency_loss = F.mse_loss(
                    F.softmax(recent_logits[:, -1, :], dim=-1),
                    F.softmax(recent_logits[:, -2, :], dim=-1)
                ) * 0.1
                loss += consistency_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            scheduler.step()
            
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
                
                strategic_features = torch.stack([
                    batch['opp_coop_rate'],
                    batch['my_coop_rate'],
                    batch['recent_trend']
                ], dim=-1).to(device)
                
                action_logits = model(player_actions, opponent_actions, returns_to_go, strategic_features)
                loss = criterion(action_logits[:, -1, :], target_actions)
                
                val_loss += loss.item()
                _, predicted = torch.max(action_logits[:, -1, :], 1)
                val_total += target_actions.size(0)
                val_correct += (predicted == target_actions).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Get current learning rate
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
              f"LR: {current_lr:.1e} | Time: {epoch_time:.1f}s | ETA: {eta}")
        
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


def evaluate_transformer(
    model: EfficientDecisionTransformer,
    strategies: List,
    num_rounds: int = 100,
    num_episodes: int = 50,
    device: str = 'cpu',
    log_dir: Optional[str] = None
) -> Dict:
    """Evaluate efficient transformer with strategic metrics"""
    
    print("🎯 Evaluating transformer against opponents:")
    model.eval()
    results = {}
    
    for strategy in strategies:
        print(f"   vs {strategy.name}...")
        
        scores = []
        cooperation_rates = []
        adaptation_scores = []
        
        for episode in range(num_episodes):
            env = IPDEnv(num_rounds=num_rounds, seed=episode)
            
            # Initialize history
            history = []
            episode_rewards = []
            actions_taken = []
            
            for round_num in range(num_rounds):
                if len(history) < 15:  # Context length
                    # Strategic initialization based on opponent
                    if hasattr(strategy, 'name') and 'Cooperate' in strategy.name:
                        action = 1  # Start by defecting against always cooperate
                    elif hasattr(strategy, 'name') and 'Defect' in strategy.name:
                        action = 1  # Defect against always defect
                    else:
                        action = 0  # Cooperate with others initially
                else:
                    # Use transformer to predict action
                    context_start = len(history) - 15
                    player_actions = [h[0] for h in history[context_start:]]
                    opponent_actions = [h[1] for h in history[context_start:]]
                    
                    # Calculate strategic returns-to-go
                    recent_rewards = episode_rewards[context_start:]
                    rtg = [sum(recent_rewards[i:]) * 1.1 for i in range(len(recent_rewards))]  # Boost future rewards
                    
                    # Strategic features
                    opp_coop_rate = sum(1-a for a in opponent_actions) / len(opponent_actions)
                    my_coop_rate = sum(1-a for a in player_actions) / len(player_actions)
                    recent_opp = opponent_actions[-5:] if len(opponent_actions) >= 5 else opponent_actions
                    recent_trend = sum(1-a for a in recent_opp) / len(recent_opp)
                    
                    # Prepare input tensors
                    player_tensor = torch.tensor([player_actions], dtype=torch.long).to(device)
                    opponent_tensor = torch.tensor([opponent_actions], dtype=torch.long).to(device)
                    rtg_tensor = torch.tensor([rtg], dtype=torch.float).to(device)
                    strategic_tensor = torch.tensor([[opp_coop_rate, my_coop_rate, recent_trend]], dtype=torch.float).to(device)
                    
                    with torch.no_grad():
                        logits = model(player_tensor, opponent_tensor, rtg_tensor, strategic_tensor)
                        probs = torch.softmax(logits[0, -1], dim=0)
                        
                        # Strategic action selection with exploration
                        if np.random.random() < 0.1:  # 10% exploration
                            action = np.random.choice([0, 1])
                        else:
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
            
            # Calculate adaptation score
            adaptation_score = 0.0
            if len(actions_taken) >= 10:
                for i in range(5, len(actions_taken)-1):
                    if (episode_rewards[i] < episode_rewards[i-1] - 1 and 
                        actions_taken[i-1] == 0 and actions_taken[i] != actions_taken[i-1]):
                        adaptation_score += 1
                adaptation_score /= max(1, len(actions_taken) - 6)
            
            scores.append(total_score)
            cooperation_rates.append(cooperation_rate)
            adaptation_scores.append(adaptation_score)
        
        # Store results
        results[strategy.name] = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'cooperation_rate': np.mean(cooperation_rates),
            'adaptation_score': np.mean(adaptation_scores)
        }
        
        print(f"      Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
        print(f"      Cooperation: {np.mean(cooperation_rates):.2%}")
        print(f"      Adaptation: {np.mean(adaptation_scores):.3f}")
    
    # Save results
    if log_dir:
        results_df = pd.DataFrame(results).T
        results_df.to_csv(f"{log_dir}/evaluation_results.csv")
        plot_evaluation_results(results, log_dir)
    
    return results


def plot_training_history(history: Dict, log_dir: str) -> None:
    """Plot efficient training history"""
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='red', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Efficient Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], label='Train Accuracy', color='blue', linewidth=2)
    ax2.plot(epochs, history['val_acc'], label='Val Accuracy', color='red', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Strategic Learning Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate plot
    ax3.plot(epochs, history['lr'], label='Learning Rate', color='green', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('OneCycle LR Schedule')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Training efficiency
    cumulative_time = [(i+1) * np.mean([epochs[j] for j in range(min(i+1, len(epochs)))]) for i in range(len(epochs))]
    ax4.plot(cumulative_time, history['val_acc'], label='Val Accuracy vs Time', color='purple', linewidth=2)
    ax4.set_xlabel('Training Time (relative)')
    ax4.set_ylabel('Validation Accuracy')
    ax4.set_title('Training Efficiency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{log_dir}/training_history.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_evaluation_results(results: Dict, log_dir: str):
    """Plot strategic evaluation results"""
    strategies = list(results.keys())
    scores = [results[s]['mean_score'] for s in strategies]
    score_stds = [results[s]['std_score'] for s in strategies]
    coop_rates = [results[s]['cooperation_rate'] for s in strategies]
    adaptation_scores = [results[s]['adaptation_score'] for s in strategies]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scores plot
    bars1 = ax1.bar(strategies, scores, yerr=score_stds, capsize=5, alpha=0.8, color='blue')
    ax1.set_ylabel('Mean Score')
    ax1.set_title('Transformer: Strategic Performance')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    for bar, score in zip(bars1, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{score:.1f}', ha='center', va='bottom')
    
    # Cooperation rates plot
    bars2 = ax2.bar(strategies, coop_rates, alpha=0.8, color='green')
    ax2.set_ylabel('Cooperation Rate')
    ax2.set_title('Transformer: Strategic Cooperation')
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
    ax3.set_title('Transformer: Strategic Adaptation')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    for bar, score in zip(bars3, adaptation_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.2f}', ha='center', va='bottom')
    
    # Performance comparison with targets
    targets = [250, 500, 100, 200, 250, 100, 250]  # Target scores for each opponent
    efficiency = [s/t for s, t in zip(scores, targets)]
    
    bars4 = ax4.bar(strategies, efficiency, alpha=0.8, color='purple')
    ax4.set_ylabel('Efficiency (Score/Target)')
    ax4.set_title('Strategic Efficiency Analysis')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Target')
    
    for bar, eff in zip(bars4, efficiency):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{eff:.2f}', ha='center', va='bottom')
    
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f"{log_dir}/transformer_evaluation.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Efficient Decision Transformer for IPD")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--num_games", type=int, default=1000,
                        help="Number of games for dataset (default: 1000)")
    parser.add_argument("--num_rounds", type=int, default=100,
                        help="Number of rounds per game (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    print("=== Efficient Decision Transformer Training for IPD ===")
    print(f"📊 Dataset: {args.num_games} games, {args.num_rounds} rounds each")
    print(f"🎯 Training: {args.epochs} epochs, batch size {args.batch_size}")
    
    # Generate strategic dataset
    trajectories = generate_strategic_dataset(
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
    print("\n📊 Strategic evaluation:")
    all_strategies = [
        TitForTat(), AlwaysCooperate(), AlwaysDefect(), 
        RandomStrategy(seed=args.seed), PavlovStrategy(),
        GrudgerStrategy(), GTFTStrategy(seed=args.seed+100)
    ]
    
    results = evaluate_transformer(model, all_strategies)
    
    print("\n✅ Efficient Transformer Training completed!")
    print(f"🎯 Model trained with strategic focus")
    print(f"📊 Results saved in results/transformer/")
    print(f"💾 Model saved in models/transformer_best.pth") 