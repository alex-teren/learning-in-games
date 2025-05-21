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

# Add project root to path to allow imports from other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from env import IPDEnv, Strategy, TitForTat, AlwaysCooperate, AlwaysDefect, RandomStrategy, simulate_match, PavlovStrategy


def save_plot_and_csv(x, y, name: str, folder: str = "results"):
    """Save PNG plot **and** matching CSV so LLM can analyse the numbers."""
    import os, pandas as pd, matplotlib.pyplot as plt
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame({"x": x, "y": y}).to_csv(f"{folder}/{name}_data.csv", index=False)
    plt.figure(); plt.plot(x, y); plt.title(name.replace("_", " ").title())
    plt.savefig(f"{folder}/{name}.png", dpi=120, bbox_inches="tight"); plt.close()


def generate_trajectory_dataset(
    num_games: int = 1000,
    num_rounds: int = 10,
    strategy_pairs: Optional[List[Tuple[Strategy, Strategy]]] = None,
    include_random_games: bool = True,
    seed: int = 42,
    log_dir: Optional[str] = None
) -> List[Dict]:
    """
    Generate a dataset of game trajectories by simulating IPD games
    
    Args:
        num_games: Number of games to simulate
        num_rounds: Number of rounds per game
        strategy_pairs: List of (player, opponent) strategy pairs to use
        include_random_games: Whether to include games with random actions
        seed: Random seed for reproducibility
        log_dir: Directory to save dataset
        
    Returns:
        List of game trajectory dictionaries
    """
    print(f"Generating dataset of {num_games} IPD games...")
    
    # Get repo root and set default path if not provided
    if log_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        log_dir = repo_root / "results"
    
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)
    
    # Create environment
    env = IPDEnv(num_rounds=num_rounds, seed=seed)
    
    # Define strategies if not provided
    if strategy_pairs is None:
        # Create classic strategies
        strategies = [
            TitForTat(),
            AlwaysCooperate(),
            AlwaysDefect(),
            RandomStrategy(seed=seed),
            PavlovStrategy()
        ]
        
        # Create all possible pairs of strategies
        strategy_pairs = []
        for i, s1 in enumerate(strategies):
            for s2 in strategies:
                strategy_pairs.append((s1, s2))
    
    # Generate trajectories
    trajectories = []
    
    for game_idx in tqdm(range(num_games), desc="Generating games"):
        # Decide whether to use strategy pair or random actions
        if include_random_games and random.random() < 0.3:  # 30% random games
            # Random action sequences
            trajectory = simulate_random_game(env, num_rounds)
        else:
            # Select a random strategy pair
            player_strategy, opponent_strategy = random.choice(strategy_pairs)
            
            # Simulate game with these strategies
            match_results = simulate_match(env, player_strategy, opponent_strategy, num_rounds)
            
            # Process results into trajectory
            trajectory = {
                'player_strategy': player_strategy.name,
                'opponent_strategy': opponent_strategy.name,
                'states': [],
                'actions': [],
                'rewards': [],
                'returns_to_go': [],
                'player_score': match_results['player_score'],
                'opponent_score': match_results['opponent_score']
            }
            
            # Extract timestep data
            for step in match_results['history']:
                # State is the observation at this step
                # For simplicity, we'll use the opponent's action from the previous round as state
                if len(trajectory['actions']) == 0:
                    prev_opponent_action = -1  # No previous action for first round
                else:
                    prev_opponent_action = step['opponent_action']
                
                trajectory['states'].append(prev_opponent_action)
                trajectory['actions'].append(step['player_action'])
                trajectory['rewards'].append(step['player_payoff'])
            
            # Calculate returns-to-go (sum of future rewards from each state)
            returns = []
            R = 0
            for r in reversed(trajectory['rewards']):
                R += r
                returns.insert(0, R)
            trajectory['returns_to_go'] = returns
        
        trajectories.append(trajectory)
    
    # Save the raw dataset
    os.makedirs(f"{log_dir}/transformer", exist_ok=True)
    with open(f"{log_dir}/transformer/trajectory_dataset.pkl", "wb") as f:
        pickle.dump(trajectories, f)
    
    print(f"Generated {len(trajectories)} game trajectories")
    return trajectories


def simulate_random_game(env: IPDEnv, num_rounds: int) -> Dict:
    """
    Simulate a game with random actions
    
    Args:
        env: IPD environment
        num_rounds: Number of rounds
        
    Returns:
        Game trajectory dictionary
    """
    trajectory = {
        'player_strategy': 'Random',
        'opponent_strategy': 'Random',
        'states': [],
        'actions': [],
        'rewards': [],
        'returns_to_go': [],
        'player_score': 0,
        'opponent_score': 0
    }
    
    # Reset environment
    obs, _ = env.reset()
    
    # Play random actions
    for _ in range(num_rounds):
        # Choose random actions
        player_action = random.randint(0, 1)
        
        # Take step in environment
        _, reward, terminated, truncated, info = env.step(player_action)
        
        # Record data
        if len(trajectory['actions']) == 0:
            prev_opponent_action = -1
        else:
            prev_opponent_action = info['opponent_action']
        
        trajectory['states'].append(prev_opponent_action)
        trajectory['actions'].append(player_action)
        trajectory['rewards'].append(reward)
    
    # Update final scores
    trajectory['player_score'] = env.player_score
    trajectory['opponent_score'] = env.opponent_score
    
    # Calculate returns-to-go
    returns = []
    R = 0
    for r in reversed(trajectory['rewards']):
        R += r
        returns.insert(0, R)
    trajectory['returns_to_go'] = returns
    
    return trajectory


class IPDDataset(Dataset):
    """PyTorch Dataset for IPD trajectories"""
    
    def __init__(self, trajectories: List[Dict], context_length: int = 5):
        """
        Initialize dataset from trajectories
        
        Args:
            trajectories: List of trajectory dictionaries
            context_length: Number of timesteps to use as context
        """
        self.trajectories = trajectories
        self.context_length = context_length
        
        # Create flattened samples
        self.samples = self._create_samples()
    
    def _create_samples(self) -> List[Dict]:
        """
        Create individual samples from trajectories
        
        Each sample consists of:
        - state_context: Previous opponent actions
        - action_context: Previous player actions
        - rtg_context: Returns-to-go for each timestep
        - target_action: Action to predict
        
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        for traj in self.trajectories:
            traj_length = len(traj['actions'])
            
            # Skip trajectories that are too short
            if traj_length < 2:  # Need at least one previous action and one to predict
                continue
            
            # Create samples with increasing context length
            for t in range(1, traj_length):
                # Determine context start (up to context_length previous steps)
                context_start = max(0, t - self.context_length)
                context_size = t - context_start
                
                # Extract context
                state_context = traj['states'][context_start:t]
                action_context = traj['actions'][context_start:t]
                rtg_context = traj['returns_to_go'][context_start:t]
                
                # Get target
                target_action = traj['actions'][t]
                
                # Pad if needed
                if context_size < self.context_length:
                    pad_length = self.context_length - context_size
                    state_context = [-1] * pad_length + state_context
                    action_context = [-1] * pad_length + action_context
                    rtg_context = [0.0] * pad_length + rtg_context
                
                # Create sample
                sample = {
                    'state_context': state_context,
                    'action_context': action_context,
                    'rtg_context': rtg_context,
                    'target_action': target_action
                }
                
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Convert to torch tensors
        return {
            'state_context': torch.tensor(sample['state_context'], dtype=torch.long),
            'action_context': torch.tensor(sample['action_context'], dtype=torch.long),
            'rtg_context': torch.tensor(sample['rtg_context'], dtype=torch.float),
            'target_action': torch.tensor(sample['target_action'], dtype=torch.long)
        }


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for the Iterated Prisoner's Dilemma
    
    A simplified transformer model that takes a sequence of (return-to-go, state, action)
    tokens and predicts the next action.
    """
    
    def __init__(
        self,
        state_dim: int = 3,  # -1, 0, 1 for no action, cooperate, defect
        action_dim: int = 2,  # 0, 1 for cooperate, defect
        hidden_size: int = 128,
        max_rtg: float = 50.0,  # Maximum return-to-go value for normalization
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        context_length: int = 5
    ):
        """
        Initialize the Decision Transformer
        
        Args:
            state_dim: Number of possible states (-1, 0, 1)
            action_dim: Number of possible actions (0, 1)
            hidden_size: Size of hidden layers
            max_rtg: Maximum return-to-go value
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout probability
            context_length: Maximum sequence length
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.max_rtg = max_rtg
        self.context_length = context_length
        
        # Embeddings
        self.state_embedding = nn.Embedding(state_dim, hidden_size)  # -1, 0, 1 -> embedding
        self.action_embedding = nn.Embedding(action_dim + 1, hidden_size)  # -1, 0, 1 -> embedding
        self.rtg_embedding = nn.Linear(1, hidden_size)
        
        # Position embeddings
        self.position_embedding = nn.Embedding(context_length, hidden_size)
        
        # Token type embeddings (to distinguish rtg, state, action)
        self.token_type_embedding = nn.Embedding(3, hidden_size)  # 0=rtg, 1=state, 2=action
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=n_heads,
            dim_feedforward=hidden_size*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    
    def forward(self, states, actions, rtgs, attention_mask=None):
        """
        Forward pass
        
        Args:
            states: Batch of state contexts [B, T]
            actions: Batch of action contexts [B, T]
            rtgs: Batch of return-to-go contexts [B, T]
            attention_mask: Optional attention mask
            
        Returns:
            Action logits
        """
        batch_size, seq_length = states.shape
        
        # Create position indices
        position_ids = torch.arange(seq_length, device=states.device).unsqueeze(0).repeat(batch_size, 1)
        
        # Embed states, actions, and rtgs
        # Convert rtgs to proper shape
        rtgs = rtgs.unsqueeze(-1) / self.max_rtg  # Normalize
        
        # Convert states from -1, 0, 1 to 0, 1, 2 for embedding lookup
        state_indices = states + 1
        state_indices = torch.clamp(state_indices, 0, self.state_dim - 1)
        
        # Convert actions from -1, 0, 1 to 0, 1, 2 for embedding lookup
        action_indices = actions + 1
        action_indices = torch.clamp(action_indices, 0, self.action_dim)
        
        # Embed all tokens
        rtg_embeddings = self.rtg_embedding(rtgs)
        state_embeddings = self.state_embedding(state_indices)
        action_embeddings = self.action_embedding(action_indices)
        
        # Add position embeddings
        pos_embeddings = self.position_embedding(position_ids)
        
        # Add token type embeddings
        token_type_ids = torch.zeros_like(states)  # rtg=0, state=1, action=2
        rtg_embeddings = rtg_embeddings + pos_embeddings + self.token_type_embedding(token_type_ids)
        
        token_type_ids = torch.ones_like(states)
        state_embeddings = state_embeddings + pos_embeddings + self.token_type_embedding(token_type_ids)
        
        token_type_ids = 2 * torch.ones_like(states)
        action_embeddings = action_embeddings + pos_embeddings + self.token_type_embedding(token_type_ids)
        
        # Combine embeddings
        sequence = torch.cat([rtg_embeddings, state_embeddings, action_embeddings], dim=1)
        
        # Pass through transformer
        if attention_mask is not None:
            # Repeat mask 3 times (once for each token type)
            attention_mask = torch.cat([attention_mask, attention_mask, attention_mask], dim=1)
        
        transformer_outputs = self.transformer(sequence, src_key_padding_mask=attention_mask)
        
        # Extract action embeddings (every 3rd token starting from index 2)
        action_outputs = transformer_outputs[:, 2::3, :]
        
        # Predict actions
        action_logits = self.action_head(action_outputs)
        
        return action_logits


def train_transformer(
    trajectories: List[Dict],
    model_params: Dict = None,
    training_params: Dict = None,
    save_dir: Optional[str] = None,
    log_dir: Optional[str] = None
) -> Tuple[DecisionTransformer, Dict]:
    """
    Train a Decision Transformer on IPD trajectory data
    
    Args:
        trajectories: List of trajectory dictionaries
        model_params: Parameters for model architecture
        training_params: Parameters for training
        save_dir: Directory to save models
        log_dir: Directory to save logs and plots
        
    Returns:
        Trained model and training history
    """
    print("Training Decision Transformer for IPD...")
    
    # Get repo root and set default paths if not provided
    repo_root = Path(__file__).resolve().parents[2]
    if save_dir is None:
        save_dir = repo_root / "models"
    if log_dir is None:
        log_dir = repo_root / "results" / "transformer"
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Default model parameters
    if model_params is None:
        model_params = {
            'hidden_size': 128,
            'n_heads': 4,
            'n_layers': 3,
            'dropout': 0.1,
            'context_length': 5
        }
    
    # Default training parameters
    if training_params is None:
        training_params = {
            'batch_size': 64,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'epochs': 50,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'seed': 42
        }
    
    # Set random seeds
    torch.manual_seed(training_params['seed'])
    np.random.seed(training_params['seed'])
    random.seed(training_params['seed'])
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_params['seed'])
    
    # Create dataset and data loader
    dataset = IPDDataset(trajectories, context_length=model_params['context_length'])
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_params['batch_size'],
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_params['batch_size'],
        shuffle=False
    )
    
    print(f"Dataset: {len(dataset)} samples, {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
    # Create model
    model = DecisionTransformer(
        hidden_size=model_params['hidden_size'],
        n_heads=model_params['n_heads'],
        n_layers=model_params['n_layers'],
        dropout=model_params['dropout'],
        context_length=model_params['context_length']
    )
    
    model.to(training_params['device'])
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_params['lr'],
        weight_decay=training_params['weight_decay']
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(training_params['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_params['epochs']} (Training)"):
            # Move batch to device
            state_context = batch['state_context'].to(training_params['device'])
            action_context = batch['action_context'].to(training_params['device'])
            rtg_context = batch['rtg_context'].to(training_params['device'])
            target_action = batch['target_action'].to(training_params['device'])
            
            # Create attention mask (1 for padding, 0 for actual tokens)
            # In this case, we don't need masking as all sequences are the same length
            attention_mask = None
            
            # Forward pass
            optimizer.zero_grad()
            action_logits = model(state_context, action_context, rtg_context, attention_mask)
            
            # Compute loss
            # We only care about the prediction for the last timestep in each sequence
            loss = criterion(action_logits[:, -1, :], target_action)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(action_logits[:, -1, :], 1)
            train_total += target_action.size(0)
            train_correct += (predicted == target_action).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{training_params['epochs']} (Validation)"):
                # Move batch to device
                state_context = batch['state_context'].to(training_params['device'])
                action_context = batch['action_context'].to(training_params['device'])
                rtg_context = batch['rtg_context'].to(training_params['device'])
                target_action = batch['target_action'].to(training_params['device'])
                
                # Forward pass
                action_logits = model(state_context, action_context, rtg_context)
                
                # Compute loss
                loss = criterion(action_logits[:, -1, :], target_action)
                
                # Track statistics
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(action_logits[:, -1, :], 1)
                val_total += target_action.size(0)
                val_correct += (predicted == target_action).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{training_params['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/transformer_best.pth")
            print(f"Saved new best model with validation accuracy: {val_acc:.4f}")
    
    # Training completed
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    # Save final model
    torch.save(model.state_dict(), f"{save_dir}/transformer_final.pth")
    
    # Save history
    history_df = pd.DataFrame({
        'epoch': list(range(1, training_params['epochs'] + 1)),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc']
    })
    
    history_df.to_csv(f"{log_dir}/training_history.csv", index=False)
    
    # Plot history
    plot_training_history(history, log_dir)
    
    # Evaluate trained model
    print("Evaluating trained model against classic strategies...")
    evaluate_transformer_agent(model, model_params, training_params, log_dir)
    
    return model, history


def plot_training_history(history: Dict, log_dir: str) -> None:
    """
    Plot training history curves
    
    Args:
        history: Training history dictionary
        log_dir: Directory to save plots
    """
    # Create epochs list for plotting
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # Plot and save loss
    save_plot_and_csv(
        epochs,
        history['train_loss'],
        "transformer_train_loss",
        folder=log_dir
    )
    
    save_plot_and_csv(
        epochs,
        history['val_loss'],
        "transformer_val_loss",
        folder=log_dir
    )
    
    # Plot and save accuracy
    save_plot_and_csv(
        epochs,
        history['train_acc'],
        "transformer_train_accuracy",
        folder=log_dir
    )
    
    save_plot_and_csv(
        epochs,
        history['val_acc'],
        "transformer_val_accuracy",
        folder=log_dir
    )
    
    print(f"Training curves saved to {log_dir}")


class TransformerStrategy(Strategy):
    """
    Strategy that uses a trained Decision Transformer to make decisions
    """
    
    def __init__(self, model: DecisionTransformer, model_params: Dict, device: str = 'cpu'):
        """
        Initialize transformer strategy
        
        Args:
            model: Trained Decision Transformer model
            model_params: Model parameters
            device: Device to run model on
        """
        super().__init__("TransformerAgent")
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.context_length = model_params['context_length']
        
        # For determining returns-to-go
        self.target_return = 30.0  # Target total return for a game
    
    def action(self, history: List[Tuple[int, int]], player_idx: int = 0) -> int:
        """
        Determine next action based on game history
        
        Args:
            history: List of tuples (player_action, opponent_action) for each past round
            player_idx: Index of the player using this strategy (0 or 1)
            
        Returns:
            int: 0 for Cooperate, 1 for Defect
        """
        # If no history, default to cooperate
        if not history:
            return 0
        
        # Extract past states (opponent actions) and own actions
        states = []
        actions = []
        rewards = []
        
        for h in history:
            player_action = h[player_idx]
            opponent_action = h[1 - player_idx]
            
            # Determine reward for this state-action pair
            if player_action == 0 and opponent_action == 0:  # Both cooperate
                reward = 3
            elif player_action == 0 and opponent_action == 1:  # Player cooperates, opponent defects
                reward = 0
            elif player_action == 1 and opponent_action == 0:  # Player defects, opponent cooperates
                reward = 5
            else:  # Both defect
                reward = 1
            
            states.append(opponent_action)
            actions.append(player_action)
            rewards.append(reward)
        
        # Calculate return so far
        return_so_far = sum(rewards)
        
        # Create context for transformer
        context_states = []
        context_actions = []
        context_rtgs = []
        
        # Use up to context_length previous steps
        context_start = max(0, len(history) - self.context_length)
        
        # Calculate returns-to-go for each timestep in context
        rtgs = []
        remaining_return = self.target_return - return_so_far
        for t in range(context_start, len(history)):
            # Simple heuristic: divide remaining return by remaining timesteps
            remaining_timesteps = len(history) - t
            if remaining_timesteps > 0:
                rtg = remaining_return / remaining_timesteps
            else:
                rtg = 0
            rtgs.append(rtg)
        
        # Extract context
        context_states = states[context_start:]
        context_actions = actions[context_start:]
        context_rtgs = rtgs
        
        # Pad if needed
        if len(context_states) < self.context_length:
            pad_length = self.context_length - len(context_states)
            context_states = [-1] * pad_length + context_states
            context_actions = [-1] * pad_length + context_actions
            context_rtgs = [0.0] * pad_length + context_rtgs
        
        # Convert to tensors
        states_tensor = torch.tensor(context_states, dtype=torch.long).unsqueeze(0).to(self.device)
        actions_tensor = torch.tensor(context_actions, dtype=torch.long).unsqueeze(0).to(self.device)
        rtgs_tensor = torch.tensor(context_rtgs, dtype=torch.float).unsqueeze(0).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            action_logits = self.model(states_tensor, actions_tensor, rtgs_tensor)
            action_probs = torch.softmax(action_logits[:, -1, :], dim=-1)
            
            # Sample from distribution (or take argmax for deterministic policy)
            # action = torch.multinomial(action_probs, 1).item()  # Stochastic
            _, action = torch.max(action_probs, dim=-1)  # Deterministic
            action = action.item()
        
        return action


def evaluate_transformer_agent(
    model: DecisionTransformer,
    model_params: Dict,
    training_params: Dict,
    log_dir: Optional[str] = None,
    num_rounds: int = 100,
    num_matches: int = 20,
    seed: int = 42
) -> pd.DataFrame:
    """
    Evaluate the transformer agent against classic strategies
    
    Args:
        model: Trained transformer model
        model_params: Model parameters
        training_params: Training parameters
        log_dir: Directory to save results
        num_rounds: Number of rounds per match
        num_matches: Number of matches per opponent
        seed: Random seed
        
    Returns:
        DataFrame with evaluation results
    """
    # Get repo root and set default path if not provided
    if log_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        log_dir = repo_root / "results" / "transformer"
    
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)
    
    # Create IPD environment
    env = IPDEnv(num_rounds=num_rounds, seed=seed)
    
    # Create transformer strategy
    transformer_strategy = TransformerStrategy(
        model,
        model_params,
        device=training_params['device']
    )
    
    # Define opponent strategies
    opponent_strategies = {
        "tit_for_tat": TitForTat(),
        "always_cooperate": AlwaysCooperate(),
        "always_defect": AlwaysDefect(),
        "random": RandomStrategy(seed=seed+100),
        "pavlov": PavlovStrategy()
    }
    
    # Results container
    results_list = []
    
    # Evaluate against each opponent
    for opponent_name, opponent in opponent_strategies.items():
        matches_results = []
        
        for match in range(num_matches):
            # Simulate match
            match_results = simulate_match(env, transformer_strategy, opponent, num_rounds)
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
    
    os.makedirs(log_dir, exist_ok=True)
    results_df.to_csv(f"{log_dir}/evaluation_results.csv", index=False)
    
    print("Evaluation results:")
    print(results_df)
    
    # Create visualization of results
    plt.figure(figsize=(12, 6))
    
    # Bar chart of scores
    x = np.arange(len(results_df))
    width = 0.35
    
    plt.bar(x - width/2, results_df['avg_player_score'], width, label='Transformer')
    plt.bar(x + width/2, results_df['avg_opponent_score'], width, label='Opponent')
    
    plt.xlabel('Opponent Strategy')
    plt.ylabel('Average Score')
    plt.title('Transformer Agent vs. Classic Strategies')
    plt.xticks(x, results_df['opponent'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{log_dir}/evaluation_results.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    return results_df


def load_transformer_model(
    model_path: Optional[str] = None,
    model_params: Dict = None,
    device: str = None
) -> DecisionTransformer:
    """
    Load a saved transformer model
    
    Args:
        model_path: Path to saved model weights
        model_params: Model parameters
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    # Get repo root and set default path if not provided
    if model_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        model_path = repo_root / "models" / "transformer_best.pth"
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_params is None:
        model_params = {
            'hidden_size': 128,
            'n_heads': 4,
            'n_layers': 3,
            'dropout': 0.1,
            'context_length': 5
        }
    
    # Create model
    model = DecisionTransformer(
        hidden_size=model_params['hidden_size'],
        n_heads=model_params['n_heads'],
        n_layers=model_params['n_layers'],
        dropout=model_params['dropout'],
        context_length=model_params['context_length']
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Set parameters
    num_games = 2000  # Number of games for dataset generation
    num_rounds = 10   # Number of rounds per game
    
    # Model parameters
    model_params = {
        'hidden_size': 128,
        'n_heads': 4,
        'n_layers': 3,
        'dropout': 0.1,
        'context_length': 5
    }
    
    # Training parameters
    training_params = {
        'batch_size': 64,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42
    }
    
    # Step 1: Generate dataset
    trajectories = generate_trajectory_dataset(
        num_games=num_games,
        num_rounds=num_rounds,
        include_random_games=True,
        seed=training_params['seed']
    )
    
    # Step 2: Train transformer model
    model, history = train_transformer(
        trajectories=trajectories,
        model_params=model_params,
        training_params=training_params
    )
    
    print("Training completed!") 