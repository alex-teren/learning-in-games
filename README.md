# Learning in Games: IPD Strategy Optimization

A comprehensive comparison of three computational approaches for learning optimal strategies in the Iterated Prisoner's Dilemma (IPD):

1. **🎯 PPO** - Proximal Policy Optimization (reinforcement learning)
2. **🧬 Evolution** - Memory-One strategy optimization with CMA-ES  
3. **🤖 Transformer** - Decision Transformer approach (supervised learning)

## 📋 Overview

This project implements and compares three distinct machine learning paradigms for strategy learning in game theory. Each approach learns to play against a comprehensive set of classic IPD strategies including TitForTat, AlwaysCooperate, AlwaysDefect, Random, Pavlov, Grudger, and GTFT.

### 🔬 Unified Experimental Setup
- **Rounds per game**: 100
- **Evaluation matches**: 20 per opponent
- **Opponent strategies**: 7 classic strategies
- **Evaluation seed**: 42 (reproducible results)

## 🏗️ Project Structure

```
learning-in-games/
├── 🤖 agents/                        # Core learning algorithms
│   ├── ppo/
│   │   ├── train_ppo.py             # PPO training with enhanced environment
│   │   └── __init__.py
│   ├── evolution/
│   │   ├── train_evolution.py       # CMA-ES evolution of memory-one strategies
│   │   └── __init__.py
│   └── transformer/
│       ├── train_transformer.py     # Decision Transformer training
│       └── __init__.py
├── 📊 notebooks/                     # Interactive demonstrations
│   ├── PPO_Demo.py                  # PPO strategy analysis & visualization
│   ├── Evolution_Demo.py            # Evolution strategy analysis & visualization
│   └── Transformer_Demo.py          # Transformer strategy analysis & visualization
├── 🎮 env/                          # Game environment
│   ├── ipd_env.py                   # IPD environment implementation
│   ├── strategies.py                # Classic strategy implementations
│   └── __init__.py
├── 📈 results/                      # Training results & evaluations
│   ├── ppo/                         # PPO results & visualizations
│   ├── evolution/                   # Evolution results & visualizations
│   └── transformer/                 # Transformer results & visualizations
├── 🔬 comparison_results/            # Cross-approach comparisons
│   ├── comprehensive_results.csv    # Complete performance comparison
│   ├── scores_comparison.csv        # Score analysis by opponent
│   ├── cooperation_comparison.csv   # Cooperation behavior analysis
│   └── summary_statistics.csv       # Overall performance summary
├── 💾 models/                       # Trained models
│   ├── ppo_ipd.zip                 # Trained PPO agent
│   └── evolved_strategy.pkl         # Evolved memory-one strategy
├── 📋 analysis/                     # Comprehensive analysis
│   ├── FINAL_ANALYSIS.md           # English comparative analysis
│   └── FINAL_ANALYSIS_UA.md        # Ukrainian comparative analysis
├── ⚙️ compare_approaches.py         # Unified comparison framework
├── 🔧 check_parameters.py           # Parameter validation utility
├── 📦 requirements.txt              # Python dependencies
└── 📖 README.md                     # This file
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd learning-in-games

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Interactive Demos
All models are pre-trained and ready for demonstration:

```bash
# PPO Demo - Reinforcement Learning Analysis
python notebooks/PPO_Demo.py

# Evolution Demo - Memory-One Strategy Analysis  
python notebooks/Evolution_Demo.py

# Transformer Demo - Decision Transformer Analysis
python notebooks/Transformer_Demo.py
```

### 3. Convert to Jupyter Notebooks (Optional)
```bash
pip install jupytext

# Convert .py demos to .ipynb format
jupytext --to notebook notebooks/PPO_Demo.py
jupytext --to notebook notebooks/Evolution_Demo.py  
jupytext --to notebook notebooks/Transformer_Demo.py

# Launch Jupyter
jupyter lab
```

## 🎯 Approach Details

### 🎯 PPO (Proximal Policy Optimization)
- **Type**: Reinforcement Learning
- **Architecture**: Enhanced IPD environment with 25-dimensional observation space
- **Training**: Multi-opponent curriculum against all 7 strategies
- **Strategy**: Learns adaptive policy through reward optimization
- **Strengths**: Excellent against reciprocal strategies, strong overall performance

### 🧬 Evolution (CMA-ES Memory-One)
- **Type**: Evolutionary Optimization
- **Strategy**: Memory-One with 5 parameters (p_cc, p_cd, p_dc, p_dd, initial_action)
- **Training**: CMA-ES optimization over 50 generations with population size 20
- **Behavior**: Mathematically interpretable conditional cooperation probabilities
- **Strengths**: Consistent performance, interpretable strategy parameters

### 🤖 Transformer (Decision Transformer)
- **Type**: Supervised Learning
- **Architecture**: 32-dim hidden size, 15-step context length, 4 layers
- **Training**: Learns from expert trajectories via imitation learning
- **Strategy**: Sequence-to-sequence decision making
- **Strengths**: Stable training, pattern recognition in opponent behavior

## 📊 Results Summary

Based on comprehensive evaluation across all opponent strategies:

| Approach | Average Score | Rank | Best Against | Strategy Type |
|----------|---------------|------|--------------|---------------|
| **PPO** | 258.8 | 🥇 #1 | Reciprocal opponents | Adaptive RL policy |
| **Evolution** | 233.7 | 🥈 #2 | Tit-for-Tat, Always Defect | Memory-One rules |
| **Transformer** | 217.0 | 🥉 #3 | Defensive strategies | Sequence learning |

### 🎯 Key Findings
- **PPO** excels with adaptive learning and handles complex opponent behaviors best
- **Evolution** provides interpretable strategy with consistent reciprocal behavior  
- **Transformer** shows stable learning but requires more sophisticated architectures
- All approaches successfully learn cooperative strategies adapted to opponent types

## 📈 Performance Visualizations

Each demo notebook generates comprehensive visualizations:
- **Performance comparison** across all opponents
- **Cooperation rate analysis** showing adaptive behavior
- **Cross-approach comparisons** highlighting strengths/weaknesses
- **Strategic insights** with detailed behavioral analysis

Results are automatically saved to CSV files for further analysis.

## 🔬 Analysis & Documentation

Comprehensive analysis available in both English and Ukrainian:

### English Analysis:
- [`analysis/FINAL_ANALYSIS.md`](analysis/FINAL_ANALYSIS.md) - Complete comparative study

### Ukrainian Analysis (Українською):
- [`analysis/FINAL_ANALYSIS_UA.md`](analysis/FINAL_ANALYSIS_UA.md) - Повний порівняльний аналіз

Each analysis includes:
- **Training methodology** and implementation details
- **Performance metrics** and statistical analysis  
- **Strategic behavior** interpretation and insights
- **Technical assessment** with recommendations
- **Future research directions**

## 🛠️ Advanced Usage

### Re-train Models
```bash
# Train PPO agent
python agents/ppo/train_ppo.py

# Evolve memory-one strategy
python agents/evolution/train_evolution.py

# Train Decision Transformer
python agents/transformer/train_transformer.py
```

### Run Complete Comparison
```bash
# Execute unified comparison of all approaches
python compare_approaches.py
```

### Parameter Validation
```bash
# Verify unified parameters across all approaches
python check_parameters.py
```

## 🎮 Game Environment

The IPD environment supports:
- **Configurable rounds** (default: 100)
- **Multiple opponent strategies** (7 classic strategies implemented)
- **Reproducible results** with fixed seeds
- **Detailed logging** of cooperation rates and scores
- **Flexible observation spaces** for different learning approaches

## 📚 Dependencies

Key dependencies (see `requirements.txt` for complete list):
- **stable-baselines3** - PPO implementation
- **cma** - CMA-ES evolutionary optimization  
- **transformers** - Transformer architecture
- **matplotlib, pandas** - Analysis and visualization
- **numpy** - Numerical computations

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional learning algorithms (A3C, SAC, etc.)
- Extended opponent strategy sets
- Multi-objective optimization
- Tournament-style evaluation
- Advanced transformer architectures

## 📄 License

This project is part of academic research on learning in games and computational game theory.

---

🎯 **Explore the interactive demos to see each approach in action!**