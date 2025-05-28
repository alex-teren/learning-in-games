# Learning in Games: IPD Strategy Optimization

A comprehensive comparison of three computational approaches for learning optimal strategies in the Iterated Prisoner's Dilemma:

1. **PPO** - Proximal Policy Optimization (reinforcement learning)
2. **Evolution** - Evolutionary strategy optimization with CMA-ES
3. **Transformer** - Decision Transformer approach (deep learning)

## 🏗️ Project Structure

```
├── agents/                           # Core learning algorithms
│   ├── ppo/train_ppo.py             # PPO training and evaluation
│   ├── evolution/train_evolution.py  # Evolutionary optimization
│   └── transformer/train_transformer.py # Transformer training
├── analysis/                        # Comprehensive analysis files
│   ├── PPO_ANALYSIS.md              # English PPO analysis
│   ├── PPO_ANALYSIS_UA.md           # Ukrainian PPO analysis  
│   ├── EVOLUTION_ANALYSIS.md        # English evolution analysis
│   ├── EVOLUTION_ANALYSIS_UA.md     # Ukrainian evolution analysis
│   ├── TRANSFORMER_ANALYSIS.md      # English transformer analysis
│   ├── TRANSFORMER_ANALYSIS_UA.md   # Ukrainian transformer analysis
│   ├── FINAL_ANALYSIS.md            # English comparative analysis
│   └── FINAL_ANALYSIS_UA.md         # Ukrainian comparative analysis
├── env/                             # IPD environment
│   ├── ipd_env.py                   # Core game environment
│   └── strategies.py                # Classic strategies (TitForTat, etc.)
├── compare_approaches.py            # Unified comparison of all approaches
├── notebooks/                       # Jupyter demonstrations
│   ├── PPO_Demo.ipynb
│   ├── Evolution_Demo.ipynb
│   └── Transformer_Demo.ipynb
├── results/                         # Training results
├── models/                          # Saved models
└── docs/                           # Analysis documentation
```

## 📊 Results Analysis

The `analysis/` directory contains comprehensive analysis files for all approaches:

### English Analysis Files:
- `analysis/PPO_ANALYSIS.md` - Analysis of reinforcement learning approach
- `analysis/EVOLUTION_ANALYSIS.md` - Analysis of evolutionary approach  
- `analysis/TRANSFORMER_ANALYSIS.md` - Analysis of transformer approach
- `analysis/FINAL_ANALYSIS.md` - Comprehensive comparative analysis

### Ukrainian Analysis Files (Українською):
- `analysis/PPO_ANALYSIS_UA.md` - Аналіз підходу навчання з підкріпленням
- `analysis/EVOLUTION_ANALYSIS_UA.md` - Аналіз еволюційного підходу
- `analysis/TRANSFORMER_ANALYSIS_UA.md` - Аналіз підходу з трансформером
- `analysis/FINAL_ANALYSIS_UA.md` - Комплексний порівняльний аналіз

Each analysis includes:
- Training configuration and performance metrics
- Strategic behavior interpretation
- Comparative evaluation against different opponent types
- Technical assessment and recommendations