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
# This notebook demonstrates a Decision-Transformer agent trained to play
# the Iterated Prisoner's Dilemma (IPD). The model imitates behaviour from
# trajectories rather than learning online.

# %%
import os
import sys
import time
import random
import pickle
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# %%
# Parse command line arguments
parser = argparse.ArgumentParser(description="Demo of Decision Transformer for Iterated Prisoner's Dilemma")
parser.add_argument("--num_rounds", type=int, default=100,
                    help="Number of rounds per episode (default: 100)")
parser.add_argument("--num_matches", type=int, default=20,
                    help="Number of matches per opponent for evaluation (default: 20)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")

# Handle running in Jupyter vs as script
try:
    # When running as a script
    args = parser.parse_args()
except SystemExit:
    # When running in Jupyter (no command line args)
    args = parser.parse_args([])

# %%
# Locate repo root in both .py (jupytext) and .ipynb contexts
try:                         # executed as .py file
    repo_root = Path(__file__).resolve().parents[1]
except NameError:            # executed in a Jupyter kernel
    repo_root = Path.cwd().resolve()
    if repo_root.name == "notebooks":
        repo_root = repo_root.parent

sys.path.append(str(repo_root))

from env import (
    IPDEnv,
    TitForTat,
    AlwaysCooperate,
    AlwaysDefect,
    RandomStrategy,
    PavlovStrategy,
    GrudgerStrategy,
    GTFTStrategy,
    simulate_match,
)

# Paths
models_dir = repo_root / "models"
results_dir = repo_root / "results"

# Helper: PNG + CSV saver
def save_plot_and_csv(x, y, name: str, folder: str = "results"):
    """Save PNG and matching CSV so numbers stay inspectable."""
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(folder, exist_ok=True)
    pd.DataFrame({"x": x, "y": y}).to_csv(f"{folder}/{name}_data.csv", index=False)
    plt.figure()
    plt.plot(x, y)
    plt.title(name.replace("_", " ").title())
    plt.savefig(f"{folder}/{name}.png", dpi=120, bbox_inches="tight")
    plt.close()

# %% [markdown]
# ## Decision-Transformer architecture

# %%
class DecisionTransformer(torch.nn.Module):
    """Causal transformer that predicts the next action given (RTG, state, action) context."""

    def __init__(
        self,
        state_dim=3,         # -1 / 0 / 1   (no-action, C, D)
        action_dim=2,        # 0 / 1        (C, D)
        hidden_size=128,
        max_rtg=50.0,
        n_heads=4,
        n_layers=3,
        dropout=0.1,
        context_length=5,
    ):
        super().__init__()
        self.context_length = context_length
        self.max_rtg = max_rtg

        # Token embeddings
        self.state_embedding = torch.nn.Embedding(state_dim, hidden_size)
        self.action_embedding = torch.nn.Embedding(action_dim + 1, hidden_size)  # +1 for padding/-1
        self.rtg_embedding = torch.nn.Linear(1, hidden_size) 

        # Positional & token-type embeddings
        self.position_embedding = torch.nn.Embedding(context_length, hidden_size)
        self.token_type_embedding = torch.nn.Embedding(3, hidden_size)  # 0=rtg 1=state 2=action

        # Transformer encoder
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Action head
        self.action_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_dim),
        )

    def forward(self, states, actions, rtgs, attn_mask=None):
        B, L = states.shape

        pos_ids = torch.arange(L, device=states.device).unsqueeze(0).repeat(B, 1)

        # normalise RTG
        rtgs = (rtgs.unsqueeze(-1) / self.max_rtg).clamp(0, 1)

        s_idx = (states + 1).clamp(0)        # -1→0, 0→1, 1→2
        a_idx = (actions + 1).clamp(0)

        rtg_tok    = self.rtg_embedding(rtgs)
        state_tok  = self.state_embedding(s_idx)
        action_tok = self.action_embedding(a_idx)

        # + positional / token-type
        rtg_tok    += self.position_embedding(pos_ids) + self.token_type_embedding(torch.zeros_like(states))
        state_tok  += self.position_embedding(pos_ids) + self.token_type_embedding(torch.ones_like(states))
        action_tok += self.position_embedding(pos_ids) + self.token_type_embedding(2 * torch.ones_like(states))

        seq = torch.cat([rtg_tok, state_tok, action_tok], dim=1)  # shape B × 3L × H

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(1, 3)  # expand to sequence length
        h = self.transformer(seq, src_key_padding_mask=attn_mask)

        action_h = h[:, 2::3, :]  # every third token (action positions)
        logits = self.action_head(action_h)
        return logits


class TransformerStrategy:
    """Wrapper that exposes `.action(history)` for simulate_match."""

    def __init__(self, model, model_params, device="cpu"):
        self.name = "TransformerAgent"
        self.model = model.to(device).eval()
        self.device = device
        self.context_len = model_params["context_length"]
        self.target_return = 30.0  # heuristic RTG

    def action(self, history, player_idx=0):
        if not history:
            return 0  # cooperate first

        states, actions, rewards = [], [], []

        for p_act, o_act in [(h[player_idx], h[1 - player_idx]) for h in history]:
            # PD payoff
            rewards.append((3, 0, 5, 1)[p_act * 2 + o_act])  # quick map
            states.append(o_act)
            actions.append(p_act)

        rtg = self.target_return - sum(rewards)
        rtgs = [rtg / max(1, len(history) - i) for i in range(len(history))]

        # take last `context_len`
        states = ([-1] * self.context_len + states)[-self.context_len :]
        actions = ([-1] * self.context_len + actions)[-self.context_len :]
        rtgs = ([0.0] * self.context_len + rtgs)[-self.context_len :]

        s = torch.tensor(states, dtype=torch.long, device=self.device).unsqueeze(0)
        a = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(0)
        r = torch.tensor(rtgs, dtype=torch.float, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(s, a, r)[:, -1, :]
            action = torch.argmax(logits, dim=-1).item()
        return action

# %% [markdown]
# ## Loading the trained model (quick-demo fallback)

# %%
def quick_train_transformer(save_dir=models_dir, num_epochs=2, seed=args.seed, num_rounds=args.num_rounds):
    """Quickly train a transformer for demo purposes."""
    print("Training a quick demo transformer model...")
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create environment and generate a small dataset
    env = IPDEnv(num_rounds=num_rounds, seed=seed)
    
    # Generate a small dataset of trajectories
    num_games = 200
    strategies = [TitForTat(), AlwaysCooperate(), AlwaysDefect(), RandomStrategy(seed=seed), PavlovStrategy(), GrudgerStrategy(), GTFTStrategy(seed=seed+100)]
    
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

model_path   = models_dir / "transformer_best.pth"
params_path  = models_dir / "transformer_params.pkl"
quick_demo   = False

if not model_path.exists():
    alt = list(models_dir.glob("*transformer*.pth"))
    if alt:
        model_path = alt[0]
        if not params_path.exists():
            alt_par = list(models_dir.glob("*transformer*params*.pkl"))
            params_path = alt_par[0] if alt_par else params_path
        print(f"Using alternative model: {model_path}")
    elif os.getenv("QUICK_DEMO") == "1":
        quick_demo = True
        model, model_params = quick_train_transformer()
    else:
        raise FileNotFoundError(
            "Transformer model not found. Run full training or set QUICK_DEMO=1."
        )

if not quick_demo:
    if params_path.exists():
        with open(params_path, "rb") as f:
            model_params = pickle.load(f)
    else:
        model_params = dict(hidden_size=128, n_heads=4, n_layers=3, dropout=0.1, context_length=5)

    model = DecisionTransformer(
        state_dim=3,
        action_dim=2,
        hidden_size=model_params["hidden_size"],
        n_heads=model_params["n_heads"],
        n_layers=model_params["n_layers"],
        dropout=model_params["dropout"],
        context_length=model_params["context_length"],
        max_rtg=50.0
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

transformer_strategy = TransformerStrategy(model, model_params)

# %% [markdown]
# ## Evaluation against classic strategies

# %%
def play_match(strategy, opponent, num_rounds=args.num_rounds, seed=args.seed):
    env = IPDEnv(num_rounds=num_rounds, seed=seed)
    res = simulate_match(env, strategy, opponent, num_rounds)
    return dict(
        player_score=res["player_score"],
        opponent_score=res["opponent_score"],
        player_coop_rate=res["cooperation_rate_player"],
        opponent_coop_rate=res["cooperation_rate_opponent"],
    )


opponents = {
    "tit_for_tat": TitForTat(),
    "always_cooperate": AlwaysCooperate(),
    "always_defect": AlwaysDefect(),
    "random": RandomStrategy(seed=args.seed),
    "pavlov": PavlovStrategy(),
    "grudger": GrudgerStrategy(),
    "gtft": GTFTStrategy(seed=args.seed+100),
}

results = {}

for name, opp in opponents.items():
    rows = [play_match(transformer_strategy, opp, seed=args.seed + i) for i in range(args.num_matches)]
    results[name] = {
        "avg_player_score": np.mean([r["player_score"] for r in rows]),
        "avg_opponent_score": np.mean([r["opponent_score"] for r in rows]),
        "avg_player_coop": np.mean([r["player_coop_rate"] for r in rows]),
        "avg_opponent_coop": np.mean([r["opponent_coop_rate"] for r in rows]),
    }
    print(f"{name}: score {results[name]['avg_player_score']:.1f}, coop {results[name]['avg_player_coop']:.2f}")

# %% [markdown]
# ## Visualisation

# %%
names = list(results.keys())
scores = [results[n]["avg_player_score"] for n in names]
p_coop = [results[n]["avg_player_coop"] for n in names]
o_coop = [results[n]["avg_opponent_coop"] for n in names]

save_plot_and_csv(names, scores, "transformer_vs_baselines", folder=str(results_dir / "transformer"))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(names, scores)
plt.ylabel("Average Score")
plt.title("Transformer Scores vs Opponents")
plt.ylim(0, max(scores) * 1.2)
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
x = np.arange(len(names))
w = 0.35
plt.bar(x - w / 2, p_coop, w, label="Transformer")
plt.bar(x + w / 2, o_coop, w, label="Opponent")
plt.ylabel("Cooperation Rate")
plt.title("Cooperation Rates")
plt.xticks(x, names, rotation=45)
plt.ylim(0, 1.05)
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
# * **Consistent cooperation strategy:** The transformer learned to always cooperate (100% cooperation rate against TFT, Always Cooperate, Pavlov, and nearly 100% against AlwaysDefect).
# * **Good performance with cooperative opponents:** The model achieves high scores (~300) against TFT, Always Cooperate, and Pavlov by maintaining mutual cooperation.
# * **Vulnerable to Always Defect:** The model receives zero reward against Always Defect because it continues to cooperate despite being exploited.
# * **Moderate success with Random:** The model earns ~160 points against Random opponents (which cooperate ~50% of the time), showing limited adaptation to mixed strategies.
# * **Training approach impact:** The transformer was trained through behavioral cloning on cooperative trajectories, which explains why it learned a fixed cooperative pattern rather than an adaptive strategy.
# * **Convergence in learning:** The loss curves show quick initial improvement that stabilizes around epoch 10, while accuracy rises to ~77%, suggesting the model learns a simple policy that fits the training data well.
# * **Reason for behavior:** Unlike reinforcement learning approaches, the transformer doesn't explore different strategies - it simply imitates the most common patterns in its training data, which likely contained many examples of mutual cooperation.

# %% [markdown]
# ## Інтерпретація результатів
#
# * **Стратегія стабільної кооперації:** Трансформер навчився завжди кооперувати (100% рівень кооперації проти TFT, Always Cooperate, Pavlov, і майже 100% проти AlwaysDefect).
# * **Хороші результати з кооперативними опонентами:** Модель досягає високих балів (~300) проти TFT, Always Cooperate і Pavlov завдяки підтримці взаємної кооперації.
# * **Вразливість до Always Defect:** Модель отримує нульову винагороду проти Always Defect, оскільки продовжує кооперувати, незважаючи на експлуатацію.
# * **Помірний успіх з Random:** Модель заробляє ~160 балів проти випадкових опонентів (які кооперують ~50% часу), демонструючи обмежену адаптацію до змішаних стратегій.
# * **Вплив підходу до навчання:** Трансформер був навчений шляхом поведінкового клонування на кооперативних траєкторіях, що пояснює, чому він вивчив фіксований кооперативний патерн, а не адаптивну стратегію.
# * **Збіжність у навчанні:** Криві втрат показують швидке початкове покращення, яке стабілізується близько 10-ї епохи, а точність зростає до ~77%, що свідчить про те, що модель вивчає просту політику, яка добре відповідає тренувальним даним.
# * **Причина поведінки:** На відміну від підходів з підкріпленням, трансформер не досліджує різні стратегії - він просто імітує найбільш поширені патерни у своїх навчальних даних, які, ймовірно, містили багато прикладів взаємної кооперації.
