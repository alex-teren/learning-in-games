# Comprehensive Comparative Analysis: IPD Strategy Learning

## Introduction

The Iterated Prisoner's Dilemma (IPD) is a classical model for studying strategic interaction in multi-agent systems. The study of optimal strategies in this environment is important for the development of game theory, machine learning, and artificial intelligence. This work presents a comparative analysis of three approaches to strategy learning in IPD: reinforcement learning (PPO), evolutionary optimization (CMA-ES), and transformer-based architecture (Decision Transformer). Each approach implements different learning principles and has its own advantages and limitations.

## Summary

An experimental study was conducted with unified parameters and a single evaluation protocol for all approaches. According to the results of average effectiveness against 7 classic opponent strategies, PPO achieved the highest score (258.78 points), followed by the evolutionary approach (233.66 points) and the transformer (217.03 points). Each approach is characterized by distinct cooperation, adaptability, and interpretability metrics.

## Experimental Setup

**Hardware Platform:**
- Processor: Apple M1 Max
- RAM: 32GB

**Unified Parameters:**
- Game length: 100 rounds per game
- Opponent set: TitForTat, AlwaysCooperate, AlwaysDefect, Random(p=0.5), Pavlov, Grudger, GTFT(p=0.1) (7 strategies total)
- Evaluation: 20 matches per opponent for Evolution/Transformer, 20 episodes for PPO
- Random seed: 42 (identical for all approaches)
- Payoff matrix: Standard IPD (3,1,5,0)

## Training Duration

| Approach | Training Time |
|----------|---------------|
| PPO | 354.6 seconds (5.9 min) |
| Evolutionary Approach | 300.6 seconds (5.0 min) |
| Transformer | 1115.51 seconds (18.6 min) |

## Strategic Effectiveness Analysis

The results show that different learning approaches form distinct styles of strategic behavior. Each algorithm demonstrates its own balance between effectiveness, cooperation, and adaptability.

### Overall Effectiveness Metrics

| Approach | Average Score | Score Range | Average Cooperation | Cooperation Variability |
|----------|---------------|-------------|---------------------|------------------------|
| PPO | 258.78 | 99.0 - 496.0 | 33.5% | High (σ=0.323) |
| Evolutionary Approach | 233.66 | 99.62 - 400.0 | 40.1% | High (σ=0.275) |
| Transformer | 217.03 | 29.44 - 421.52 | 63.0% | Moderate (σ=0.191) |

### Results Against Specific Opponents

| Opponent | PPO Score | Evolutionary Score | Transformer Score | Leading Approach |
|----------|-----------|-------------------|------------------|------------------|
| TitForTat | 250.0 | 263.08 | 262.44 | Evolutionary Approach |
| AlwaysCooperate | 496.0 | 400.0 | 421.52 | PPO |
| AlwaysDefect | 99.0 | 99.62 | 68.5 | Evolutionary Approach |
| Random | 249.11 | 234.93 | 194.58 | PPO |
| Pavlov | 250.0 | 263.04 | 265.38 | Transformer |
| Grudger | 152.0 | 101.67 | 29.44 | PPO |
| GTFT | 315.36 | 273.31 | 277.32 | PPO |

## Detailed Analysis by Approach

### PPO (Reinforcement Learning)

The PPO approach provides the highest average effectiveness among the considered algorithms. Its main advantage is the ability to adapt to different types of opponents, which allows it to effectively exploit the weaknesses of cooperative strategies (e.g., AlwaysCooperate) and demonstrate competitiveness against defensive strategies (Grudger, GTFT). At the same time, the cooperation rate remains relatively low, indicating a tendency to exploit opponents in appropriate scenarios. Training time is moderate.

### Evolutionary Approach (CMA-ES)

The evolutionary approach forms a Memory-One strategy characterized by a balance between cooperation and defense. The algorithm demonstrates high effectiveness against reciprocal strategies (TitForTat, Pavlov) and provides the best defensive results against AlwaysDefect. The advantage is the transparency and interpretability of the strategy parameters. Training time is the shortest among all approaches, making this method attractive for resource-constrained tasks.

### Transformer (Decision Transformer)

The transformer architecture demonstrates the highest cooperation rate and the lowest behavioral variability. The algorithm achieves stable results in cooperative scenarios but is inferior to competitive approaches when facing aggressive opponents (AlwaysDefect, Grudger). The main advantage is the ability to maintain consistent cooperative behavior, but training time is the highest among all approaches.

## Strategic Adaptation Comparison

### Opponent Recognition Capabilities

- PPO: High adaptability, effective exploitation of weak opponents, competitiveness against complex strategies.
- Evolutionary Approach: Consistent strategic behavior, good reciprocity, best defensive results.
- Transformer: High cooperation, stability, limited effectiveness against aggressive strategies.

### Game-Theoretic Characteristics

| Aspect | PPO | Evolutionary Approach | Transformer |
|--------|-----|----------------------|-------------|
| Reciprocity | Good | Very good | Good |
| Exploiting opponents | Very good | Good | Moderate |
| Defense | Good | Very good | Weak |
| Adaptability | Very good | Good | Limited |
| Consistency | Moderate | Good | Very good |

## Computational Requirements

| Metric | PPO | Evolutionary Approach | Transformer |
|--------|-----|----------------------|-------------|
| Training time | 354.6s (5.9 min) | 300.6s (5.0 min) | 1115.5s (18.6 min) |
| Memory usage | Moderate | Low | High |
| Computational complexity | Moderate | Low | Very high |
| Scalability | Decent | Very good | Limited |

## Interpretability

- Evolutionary Approach: High interpretability due to explicit Memory-One strategy parameters.
- PPO: Limited interpretability due to the use of a neural network.
- Transformer: Lowest interpretability due to architectural complexity.

## Practical Recommendations

- PPO is suitable for competitive environments where maximizing effectiveness is important.
- The evolutionary approach is optimal for tasks requiring transparency, fast training, and a balance between cooperation and defense.
- The transformer is recommended for cooperative or educational scenarios where stable cooperation is valued.

## Conclusions

The study shows that none of the approaches is universally optimal for all scenarios. The choice of method should be based on requirements for effectiveness, interpretability, training speed, and the nature of the environment. The developed unified evaluation methodology enables objective comparison of different approaches and the formation of recommendations for practical use. Further research may focus on the development of hybrid systems that combine the advantages of the considered methods. 