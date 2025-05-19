# Learning in Games: Iterated Prisoner's Dilemma

This repository contains implementations of various learning agents for the Iterated Prisoner's Dilemma (IPD), developed as part of a diploma thesis on "Training Agents in the Prisoner's Dilemma using Reinforcement Learning, Evolutionary Algorithms, and Transformers."

## Project Overview

The Iterated Prisoner's Dilemma is a classic game theory problem that models cooperation and competition between rational agents. In this project, we implement three different approaches to train agents to play the IPD:

1. **Reinforcement Learning (PPO)**: Using Proximal Policy Optimization to learn a policy through interaction with the environment.
2. **Evolutionary Strategy (CMA-ES)**: Using Covariance Matrix Adaptation Evolution Strategy to evolve a memory-one strategy.
3. **Transformer-based Agent**: Using a Decision Transformer to learn from pre-collected game trajectories.

Each approach has its unique advantages and limitations, and this project aims to compare their performance in the IPD context.

## Repository Structure`

- **`env/`** – The environment and game logic:
  - `env/ipd_env.py` – Implementation of the Iterated Prisoner's Dilemma environment.
  - `env/strategies.py` – Definitions of classic strategies (TFT, AllC, AllD, Random, etc.).
- **`agents/`** – Agent implementations:
  - `agents/ppo/` – PPO reinforcement learning agent.
  - `agents/evolution/` – Evolutionary strategy agent.
  - `agents/transformer/` – Transformer-based agent.
- **`notebooks/`** – Jupyter notebooks demonstrating each agent:
  - `notebooks/PPO_Demo.ipynb` – Demo of the PPO agent.
  - `notebooks/Evolution_Demo.ipynb` – Demo of the evolutionary agent.
  - `notebooks/Transformer_Demo.ipynb` – Demo of the transformer agent.
- **`results/`** – Directory for saving experiment results, graphs, etc.
- **`models/`** – Directory for saving trained models.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/alex-teren/learning-in-games.git
   cd learning-in-games
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Demos

The easiest way to get started is to run the demo notebooks:

1. Start Jupyter:
   ```
   jupyter lab
   ```
   or
   ```
   jupyter notebook
   ```

2. Navigate to the `notebooks/` directory and open any of the demo notebooks.

3. Follow the step-by-step instructions in the notebooks to run the experiments.

### Training Agents from Scratch

To train the agents from scratch, you can run the training scripts directly:

#### PPO Agent
```
python agents/ppo/train_ppo.py
```

#### Evolutionary Agent
```
python agents/evolution/train_evolution.py
```

#### Transformer Agent
```
python agents/transformer/train_transformer.py
```

### Google Colab

The notebooks are also compatible with Google Colab. To run them:

1. Upload the notebooks to Google Drive
2. Open with Google Colab
3. Run the installation cell to set up the environment
4. Follow the instructions in the notebook

## Iterated Prisoner's Dilemma Environment

The IPD environment implements the following payoff matrix:
- Both cooperate (CC): Reward (R=3) for both
- Both defect (DD): Punishment (P=1) for both
- One cooperates, one defects (CD): Sucker (S=0) for cooperator, Temptation (T=5) for defector

The environment supports various opponent strategies and can be configured with different parameters like the number of rounds and memory size.

## Classic Strategies

The project includes several classic IPD strategies:

- **Tit-for-Tat (TFT)**: Cooperates on the first move, then copies the opponent's last move.
- **Always Cooperate (AllC)**: Always plays Cooperate.
- **Always Defect (AllD)**: Always plays Defect.
- **Random**: Chooses Cooperate or Defect randomly with equal probability.
- **Pavlov (Win-Stay, Lose-Shift)**: Cooperates on the first move, then repeats the previous action if it received a good payoff (R or T), otherwise switches.

## Reproducing Experiments

To reproduce the experiments from the thesis:

1. Train each agent type using their respective training scripts or notebooks.
2. The trained models will be saved in the `models/` directory.
3. Evaluation results and plots will be saved in the `results/` directory.

## Acknowledgments

This project was developed as part of a diploma thesis at National University of Kyiv-Mohyla Academy. It builds upon various research papers in reinforcement learning, evolutionary computation, and sequence modeling.

## References

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.
- Hansen, N., & Ostermeier, A. (2001). Completely Derandomized Self-Adaptation in Evolution Strategies. Evolutionary Computation, 9(2), 159-195.
- Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., ... & Mordatch, I. (2021). Decision Transformer: Reinforcement Learning via Sequence Modeling. Advances in Neural Information Processing Systems, 34. 