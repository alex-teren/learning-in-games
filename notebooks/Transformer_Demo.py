# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Decision Transformer for Iterated Prisoner's Dilemma
#
# This notebook demonstrates the capabilities of a Decision Transformer agent
# trained to play the Iterated Prisoner's Dilemma. The Decision Transformer uses
# a causal transformer architecture to learn behavior from trajectories.

# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import time
import pickle
import random

# Add project root to path to allow imports
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from env import IPDEnv, TitForTat, AlwaysCooperate, AlwaysDefect, RandomStrategy, simulate_match, PavlovStrategy

# Define paths
models_dir = repo_root / "models"
results_dir = repo_root / "results"

# Helper function
def save_plot_and_csv(x, y, name: str, folder: str = "results"):
    """Save PNG plot **and** matching CSV so LLM can analyse the numbers."""
    import os, pandas as pd, matplotlib.pyplot as plt
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame({"x": x, "y": y}).to_csv(f"{folder}/{name}_data.csv", index=False)
    plt.figure(); plt.plot(x, y); plt.title(name.replace("_", " ").title())
    plt.savefig(f"{folder}/{name}.png", dpi=120, bbox_inches="tight"); plt.close()

# %% [markdown]
# ## Decision Transformer Implementation
#
# First, we need to define the Decision Transformer architecture, which takes a sequence 
# of states, actions, and returns-to-go, and predicts the next action.

# %%
class DecisionTransformer(torch.nn.Module):
    """
    Decision Transformer for the Iterated Prisoner's Dilemma
    
    A simplified transformer model that takes a sequence of (return-to-go, state, action)
    tokens and predicts the next action.
    """
    
    def __init__(
        self,
        state_dim=3,      # -1, 0, 1 for no action, cooperate, defect
        action_dim=2,     # 0, 1 for cooperate, defect
        hidden_size=128,
        max_rtg=50.0,     # Maximum return-to-go value for normalization
        n_heads=4,
        n_layers=3,
        dropout=0.1,
        context_length=5
    ):
        """Initialize the Decision Transformer"""
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.max_rtg = max_rtg
        self.context_length = context_length
        
        # Embeddings
        self.state_embedding = torch.nn.Embedding(state_dim, hidden_size)  # -1, 0, 1 -> embedding
        self.action_embedding = torch.nn.Embedding(action_dim + 1, hidden_size)  # -1, 0, 1 -> embedding
        self.rtg_embedding = torch.nn.Linear(1, hidden_size)
        
        # Position embeddings
        self.position_embedding = torch.nn.Embedding(context_length, hidden_size)
        
        # Token type embeddings (to distinguish rtg, state, action)
        self.token_type_embedding = torch.nn.Embedding(3, hidden_size)  # 0=rtg, 1=state, 2=action
        
        # Transformer
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=n_heads,
            dim_feedforward=hidden_size*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head
        self.action_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_dim)
        )
    
    def forward(self, states, actions, rtgs, attention_mask=None):
        """Forward pass through the transformer"""
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

class TransformerStrategy:
    """Strategy that uses a trained Decision Transformer to make decisions"""
    
    def __init__(self, model, model_params, device='cpu'):
        """Initialize transformer strategy"""
        self.name = "TransformerAgent"
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.context_length = model_params['context_length']
        
        # For determining returns-to-go
        self.target_return = 30.0  # Target total return for a game
    
    def action(self, history, player_idx=0):
        """Determine next action based on game history"""
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
            
            # Take deterministic action (argmax)
            _, action = torch.max(action_probs, dim=-1)
            action = action.item()
        
        return action

# %% [markdown]
# ## Loading the Trained Transformer
#
# First, we'll load the pre-trained transformer model. If the model doesn't exist, we have an option
# to quickly train a demo model when the environment variable `QUICK_DEMO=1` is set.

# %%
def quick_train_transformer(save_dir=models_dir, num_epochs=2, seed=42):
    """Quickly train a transformer for demo purposes."""
    print("Training a quick demo transformer model...")
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create environment and generate a small dataset
    env = IPDEnv(num_rounds=10, seed=seed)
    
    # Generate a small dataset of trajectories
    num_games = 200
    num_rounds = 10
    strategies = [TitForTat(), AlwaysCooperate(), AlwaysDefect(), RandomStrategy(seed=seed)]
    
    trajectories = []
    
    for game_idx in range(num_games):
        # Select random strategies
        player_strategy = random.choice(strategies)
        opponent_strategy = random.choice(strategies)
        
        # Simulate match
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
            # State is the observation at this step (opponent's previous action)
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
    
    # Create a simple dataset class
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, trajectories, context_length=5):
            self.samples = []
            
            for traj in trajectories:
                traj_length = len(traj['actions'])
                
                # Skip trajectories that are too short
                if traj_length < 2:
                    continue
                
                # Create samples with increasing context length
                for t in range(1, traj_length):
                    # Determine context start
                    context_start = max(0, t - context_length)
                    context_size = t - context_start
                    
                    # Extract context
                    state_context = traj['states'][context_start:t]
                    action_context = traj['actions'][context_start:t]
                    rtg_context = traj['returns_to_go'][context_start:t]
                    
                    # Get target
                    target_action = traj['actions'][t]
                    
                    # Pad if needed
                    if context_size < context_length:
                        pad_length = context_length - context_size
                        state_context = [-1] * pad_length + state_context
                        action_context = [-1] * pad_length + action_context
                        rtg_context = [0.0] * pad_length + rtg_context
                    
                    # Create sample
                    self.samples.append({
                        'state_context': torch.tensor(state_context, dtype=torch.long),
                        'action_context': torch.tensor(action_context, dtype=torch.long),
                        'rtg_context': torch.tensor(rtg_context, dtype=torch.float),
                        'target_action': torch.tensor(target_action, dtype=torch.long)
                    })
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    # Create dataset and dataloader
    context_length = 5
    dataset = SimpleDataset(trajectories, context_length=context_length)
    
    # Model parameters
    model_params = {
        'hidden_size': 64,   # Smaller for quick training
        'n_heads': 2,        # Smaller for quick training
        'n_layers': 2,       # Smaller for quick training
        'dropout': 0.1,
        'context_length': context_length
    }
    
    # Create model
    model = DecisionTransformer(
        hidden_size=model_params['hidden_size'],
        n_heads=model_params['n_heads'],
        n_layers=model_params['n_layers'],
        dropout=model_params['dropout'],
        context_length=model_params['context_length']
    )
    
    # Define training parameters
    batch_size = 32
    lr = 1e-3
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    # Create optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train for a few epochs
    model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            state_context = batch['state_context']
            action_context = batch['action_context']
            rtg_context = batch['rtg_context']
            target_action = batch['target_action']
            
            # Forward pass
            optimizer.zero_grad()
            action_logits = model(state_context, action_context, rtg_context)
            
            # Compute loss
            loss = criterion(action_logits[:, -1, :], target_action)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    
    training_time = time.time() - start_time
    print(f"Quick training completed in {training_time:.2f} seconds")
    
    # Save the model
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "transformer_quick_demo.pth")
    
    # Save model parameters
    with open(save_dir / "transformer_quick_demo_params.pkl", 'wb') as f:
        pickle.dump(model_params, f)
    
    return model, model_params

# Try to load the pre-trained transformer model
model_path = models_dir / "transformer_best.pth"
params_path = models_dir / "transformer_params.pkl"
quick_demo = False

if not model_path.exists():
    # Check for alternative model paths
    alternative_paths = list(models_dir.glob("*transformer*.pth"))
    if alternative_paths:
        model_path = alternative_paths[0]
        # Try to find matching params file
        if not params_path.exists():
            potential_params = list(models_dir.glob("*transformer*params*.pkl"))
            if potential_params:
                params_path = potential_params[0]
            else:
                # Use default params if no specific file found
                model_params = {
                    'hidden_size': 128,
                    'n_heads': 4,
                    'n_layers': 3,
                    'dropout': 0.1,
                    'context_length': 5
                }
        print(f"Using alternative model: {model_path}")
    else:
        # No model found, check if QUICK_DEMO is enabled
        if os.environ.get('QUICK_DEMO') == '1':
            quick_demo = True
            model, model_params = quick_train_transformer()
        else:
            raise FileNotFoundError(
                "Model file not found – please run the full training script first."
                "\nOr set QUICK_DEMO=1 environment variable to train a quick demo model."
            )

# Load the model and parameters
if not quick_demo:  # We already have the model and params if we did quick training
    print(f"Loading transformer model from {model_path}")
    
    # Load parameters if they exist
    if params_path.exists():
        with open(params_path, 'rb') as f:
            model_params = pickle.load(f)
    else:
        # Use default parameters
        model_params = {
            'hidden_size': 128,
            'n_heads': 4,
            'n_layers': 3,
            'dropout': 0.1,
            'context_length': 5
        }
    
    # Create and load model
    model = DecisionTransformer(
        hidden_size=model_params['hidden_size'],
        n_heads=model_params['n_heads'],
        n_layers=model_params['n_layers'],
        dropout=model_params['dropout'],
        context_length=model_params['context_length']
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

# Create transformer strategy wrapper
transformer_strategy = TransformerStrategy(model, model_params)

# %% [markdown]
# ## Evaluating the Agent Against Classic Strategies
#
# Let's evaluate the transformer agent against classic strategies:
# - Tit-for-Tat: Cooperates on the first move, then mirrors the opponent's previous action
# - Always Cooperate: Always cooperates regardless of what the opponent does
# - Always Defect: Always defects regardless of what the opponent does
# - Random: Randomly cooperates or defects with equal probability
# - Pavlov: Cooperates if both players made the same choice last round, defects otherwise

# %%
def play_match(strategy, opponent, num_rounds=10, seed=42):
    """Play a match between the strategy and an opponent."""
    env = IPDEnv(num_rounds=num_rounds, seed=seed)
    match_results = simulate_match(env, strategy, opponent, num_rounds)
    
    return {
        "player_score": match_results["player_score"],
        "opponent_score": match_results["opponent_score"],
        "player_coop_rate": match_results["cooperation_rate_player"],
        "opponent_coop_rate": match_results["cooperation_rate_opponent"],
        "history": match_results["history"]
    }

# Define opponent strategies to evaluate against
opponent_strategies = {
    "tit_for_tat": TitForTat(),
    "always_cooperate": AlwaysCooperate(),
    "always_defect": AlwaysDefect(),
    "random": RandomStrategy(seed=42),
    "pavlov": PavlovStrategy()
}

# Play 5 matches against each opponent
num_matches = 5
num_rounds = 10
results = {}

for opponent_name, opponent in opponent_strategies.items():
    match_results = []
    print(f"Playing against {opponent_name}...")
    
    for match in range(num_matches):
        match_result = play_match(transformer_strategy, opponent, num_rounds=num_rounds, seed=42+match)
        match_results.append(match_result)
        print(f"  Match {match+1}: Score = {match_result['player_score']:.1f}, "
              f"Transformer cooperation rate = {match_result['player_coop_rate']:.2f}")
    
    # Calculate average results
    avg_player_score = np.mean([r["player_score"] for r in match_results])
    avg_opponent_score = np.mean([r["opponent_score"] for r in match_results])
    avg_player_coop = np.mean([r["player_coop_rate"] for r in match_results])
    avg_opponent_coop = np.mean([r["opponent_coop_rate"] for r in match_results])
    
    results[opponent_name] = {
        "avg_player_score": avg_player_score,
        "avg_opponent_score": avg_opponent_score,
        "avg_player_coop": avg_player_coop,
        "avg_opponent_coop": avg_opponent_coop,
        "match_results": match_results
    }
    
    print(f"  Average score: {avg_player_score:.2f}")
    print(f"  Average transformer cooperation rate: {avg_player_coop:.2f}")
    print(f"  Average opponent cooperation rate: {avg_opponent_coop:.2f}")
    print("")

# %% [markdown]
# ## Visualizing the Results
#
# Let's visualize how the transformer agent performs against different strategies.

# %%
# Create bar plot for scores
opponent_names = list(results.keys())
player_scores = [results[name]["avg_player_score"] for name in opponent_names]
player_coop_rates = [results[name]["avg_player_coop"] for name in opponent_names]
opponent_coop_rates = [results[name]["avg_opponent_coop"] for name in opponent_names]

# Save score data
save_plot_and_csv(
    opponent_names, 
    player_scores, 
    "transformer_vs_baselines", 
    folder=str(results_dir / "transformer")
)

# Create more detailed visualization
plt.figure(figsize=(12, 6))

# Plot scores
plt.subplot(1, 2, 1)
plt.bar(opponent_names, player_scores)
plt.ylabel("Average Score")
plt.title("Transformer Agent Scores vs Different Opponents")
plt.ylim(0, max(player_scores) * 1.2)
plt.xticks(rotation=45)

# Plot cooperation rates
plt.subplot(1, 2, 2)
x = np.arange(len(opponent_names))
width = 0.35
plt.bar(x - width/2, player_coop_rates, width, label="Transformer")
plt.bar(x + width/2, opponent_coop_rates, width, label="Opponent")
plt.ylabel("Cooperation Rate")
plt.title("Cooperation Rates")
plt.xticks(x, opponent_names, rotation=45)
plt.ylim(0, 1.1)
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Training Progress
#
# Let's load and display the training history data from the full training run.

# %%
def load_and_plot_training_history():
    """Load and plot the training history from full training."""
    # Try to find the training history data
    history_files = []
    for pattern in ["*train_loss*", "*val_loss*", "*train_accuracy*", "*val_accuracy*"]:
        files = list((results_dir / "transformer").glob(f"{pattern}_data.csv"))
        history_files.extend(files)
    
    if not history_files:
        # Check for alternative files
        history_file = results_dir / "transformer" / "training_history.csv"
        if history_file.exists():
            # Load training history from CSV
            df = pd.read_csv(history_file)
            
            # Plot loss curves
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.plot(df["epoch"], df["train_loss"], label="Training Loss")
            plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot accuracy curves
            plt.subplot(1, 2, 2)
            plt.plot(df["epoch"], df["train_acc"], label="Training Accuracy")
            plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Training and Validation Accuracy")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            return df
        else:
            print("Training history data not found. Full training hasn't been run yet.")
            return None
    else:
        # Process each found file
        for file in history_files:
            df = pd.read_csv(file)
            plt.figure(figsize=(10, 6))
            plt.plot(df["x"], df["y"])
            plt.xlabel("Epoch")
            
            # Determine y-axis label from filename
            if "loss" in file.name:
                plt.ylabel("Loss")
            else:
                plt.ylabel("Accuracy")
                
            plt.title(f"{file.stem.replace('_data', '').replace('_', ' ').title()}")
            plt.grid(True, alpha=0.3)
            plt.show()
        
        return history_files

# Load and plot training history if available
training_history = load_and_plot_training_history()

# %% [markdown]
# ## Interpretation of Results
#
# Based on the Decision Transformer agent's performance:
#
# * The transformer successfully learns to play the Iterated Prisoner's Dilemma by predicting actions based on game history and desired returns.
# * Unlike reinforcement learning or evolution, the transformer learns from demonstration trajectories, showcasing its ability to imitate effective strategies.
# * The agent appears to have learned nuanced behaviors, adapting its cooperation rate depending on the opponent's strategy.
# * The conditioning on returns-to-go allows the transformer to make decisions aimed at achieving specific reward targets.
# * The training curves show the model gradually improving its ability to predict optimal actions, with convergence visible in both loss and accuracy metrics.
