import gym
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from gym import spaces

from env.strategies import Strategy, TitForTat, AlwaysCooperate, AlwaysDefect, RandomStrategy


class IPDEnv(gym.Env):
    """
    Iterated Prisoner's Dilemma Environment
    
    This environment implements the Iterated Prisoner's Dilemma game as a Gym environment.
    Two agents repeatedly play the Prisoner's Dilemma game for a fixed number of rounds.
    
    Payoff Matrix:
    - Both cooperate (CC): Reward (R=3) for both
    - Both defect (DD): Punishment (P=1) for both
    - One cooperates, one defects (CD): Sucker (S=0) for cooperator, Temptation (T=5) for defector
    
    Actions:
    - 0: Cooperate
    - 1: Defect
    
    Observation:
    - By default, the last `memory_size` rounds of actions from both players
    """
    
    metadata = {'render.modes': ['human']}
    
    # Standard Prisoner's Dilemma payoff values
    REWARD = 3      # CC: Both cooperate
    PUNISHMENT = 1  # DD: Both defect
    TEMPTATION = 5  # DC: Player defects, opponent cooperates
    SUCKER = 0      # CD: Player cooperates, opponent defects
    
    # Action space constants
    COOPERATE = 0
    DEFECT = 1
    
    def __init__(
        self,
        num_rounds: int = 10,
        memory_size: int = 3,
        opponent_strategy: Optional[Union[Strategy, str]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the IPD environment
        
        Args:
            num_rounds: Number of rounds in a match
            memory_size: Number of previous rounds to include in the observation
            opponent_strategy: Strategy for the opponent (Strategy object or string name)
            seed: Random seed
        """
        super(IPDEnv, self).__init__()
        
        self.num_rounds = num_rounds
        self.memory_size = memory_size
        self.rng = np.random.RandomState(seed)
        
        # Set opponent strategy
        if opponent_strategy is None or opponent_strategy == "random":
            self.opponent_strategy = RandomStrategy(seed=self.rng.randint(10000))
        elif opponent_strategy == "tit_for_tat":
            self.opponent_strategy = TitForTat()
        elif opponent_strategy == "always_cooperate":
            self.opponent_strategy = AlwaysCooperate()
        elif opponent_strategy == "always_defect":
            self.opponent_strategy = AlwaysDefect()
        elif isinstance(opponent_strategy, Strategy):
            self.opponent_strategy = opponent_strategy
        else:
            raise ValueError(f"Unsupported opponent strategy: {opponent_strategy}")
            
        # Action and observation spaces
        self.action_space = spaces.Discrete(2)  # Cooperate (0) or Defect (1)
        
        # Observation space: memory_size * 2 values (for player and opponent actions)
        # Each value is either COOPERATE (0), DEFECT (1), or -1 (no action yet)
        low = np.ones(memory_size * 2, dtype=np.int8) * -1
        high = np.ones(memory_size * 2, dtype=np.int8)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int8)
        
        # Initialize state variables
        self.reset()

    def payoff(self, action_player: int, action_opponent: int) -> Tuple[float, float]:
        """
        Calculate payoffs for a single round based on actions
        
        Args:
            action_player: Action of the player (0=cooperate, 1=defect)
            action_opponent: Action of the opponent (0=cooperate, 1=defect)
            
        Returns:
            Tuple of (player_payoff, opponent_payoff)
        """
        if action_player == self.COOPERATE and action_opponent == self.COOPERATE:
            return self.REWARD, self.REWARD
        elif action_player == self.COOPERATE and action_opponent == self.DEFECT:
            return self.SUCKER, self.TEMPTATION
        elif action_player == self.DEFECT and action_opponent == self.COOPERATE:
            return self.TEMPTATION, self.SUCKER
        else:  # Both defect
            return self.PUNISHMENT, self.PUNISHMENT

    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new episode
        
        Returns:
            Initial observation
        """
        # Reset game state
        self.current_round = 0
        self.history = []  # [(player_action, opponent_action), ...]
        self.player_score = 0
        self.opponent_score = 0
        
        # Initial observation is all -1 (no actions yet)
        observation = np.ones(self.memory_size * 2, dtype=np.int8) * -1
        
        return observation

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one round of the game
        
        Args:
            action: Player's action (0=cooperate, 1=defect)
            
        Returns:
            (observation, reward, done, info)
        """
        # Ensure action is valid
        if action not in [self.COOPERATE, self.DEFECT]:
            raise ValueError(f"Invalid action: {action}, must be {self.COOPERATE} or {self.DEFECT}")
        
        # Get opponent's action
        opponent_action = self.opponent_strategy.action(self.history, player_idx=1)
        
        # Calculate payoffs
        player_payoff, opponent_payoff = self.payoff(action, opponent_action)
        
        # Update scores
        self.player_score += player_payoff
        self.opponent_score += opponent_payoff
        
        # Update history
        self.history.append((action, opponent_action))
        
        # Update round counter
        self.current_round += 1
        
        # Check if episode is done
        done = self.current_round >= self.num_rounds
        
        # Prepare next observation
        observation = self._get_observation()
        
        # Prepare info dict
        info = {
            'round': self.current_round,
            'player_action': action,
            'opponent_action': opponent_action,
            'player_payoff': player_payoff,
            'opponent_payoff': opponent_payoff,
            'player_score': self.player_score,
            'opponent_score': self.opponent_score,
            'opponent_strategy': self.opponent_strategy.name
        }
        
        return observation, player_payoff, done, info

    def _get_observation(self) -> np.ndarray:
        """
        Generate the observation vector from history
        
        Returns:
            Observation vector with player and opponent actions
        """
        # Initialize with -1 (no action)
        observation = np.ones(self.memory_size * 2, dtype=np.int8) * -1
        
        # Fill in the available history, most recent first
        history_len = len(self.history)
        for i in range(min(self.memory_size, history_len)):
            idx = history_len - 1 - i  # Start from most recent
            observation[i*2] = self.history[idx][0]      # Player action
            observation[i*2+1] = self.history[idx][1]    # Opponent action
        
        return observation

    def render(self, mode='human'):
        """
        Render the current state of the environment
        
        Args:
            mode: Rendering mode
        """
        if mode != 'human':
            raise NotImplementedError(f"Render mode {mode} not implemented")
        
        if len(self.history) == 0:
            print("Game not started yet!")
            return
        
        last_round = len(self.history) - 1
        player_action, opponent_action = self.history[-1]
        
        action_names = {self.COOPERATE: "Cooperate", self.DEFECT: "Defect"}
        
        player_payoff, opponent_payoff = self.payoff(player_action, opponent_action)
        
        print(f"Round {last_round + 1}/{self.num_rounds}:")
        print(f"  Player: {action_names[player_action]} → {player_payoff}")
        print(f"  Opponent ({self.opponent_strategy.name}): {action_names[opponent_action]} → {opponent_payoff}")
        print(f"  Scores: Player {self.player_score} - {self.opponent_score} Opponent")

    def get_payoff_matrix(self) -> np.ndarray:
        """
        Return the payoff matrix for the current game
        
        Returns:
            2x2x2 numpy array where [i,j,0] is player payoff and [i,j,1] is opponent payoff
            for player action i and opponent action j
        """
        payoff_matrix = np.zeros((2, 2, 2))
        for i in range(2):
            for j in range(2):
                player_payoff, opponent_payoff = self.payoff(i, j)
                payoff_matrix[i, j, 0] = player_payoff
                payoff_matrix[i, j, 1] = opponent_payoff
        return payoff_matrix


def simulate_match(env: IPDEnv, player_strategy: Strategy, opponent_strategy: Optional[Strategy] = None, 
                   num_rounds: Optional[int] = None) -> Dict[str, Any]:
    """
    Simulate a match between two strategies
    
    Args:
        env: The IPD environment
        player_strategy: Strategy for the player
        opponent_strategy: Strategy for the opponent (if None, use env's opponent)
        num_rounds: Number of rounds (if None, use env's num_rounds)
        
    Returns:
        Dictionary with match results
    """
    # Save original opponent strategy if we're replacing it
    original_opponent = None
    if opponent_strategy is not None:
        original_opponent = env.opponent_strategy
        env.opponent_strategy = opponent_strategy
    
    # Save original num_rounds if we're replacing it
    original_num_rounds = None
    if num_rounds is not None:
        original_num_rounds = env.num_rounds
        env.num_rounds = num_rounds
    
    # Reset the environment
    obs = env.reset()
    
    # Run the match
    done = False
    history = []
    
    while not done:
        # Get player action from strategy
        player_action = player_strategy.action(env.history, player_idx=0)
        
        # Take a step in the environment
        obs, reward, done, info = env.step(player_action)
        
        # Store round information
        history.append(info)
    
    # Prepare results
    match_results = {
        'player_strategy': player_strategy.name,
        'opponent_strategy': env.opponent_strategy.name,
        'player_score': env.player_score,
        'opponent_score': env.opponent_score,
        'history': history,
        'cooperation_rate_player': sum(h[0] == env.COOPERATE for h in env.history) / len(env.history),
        'cooperation_rate_opponent': sum(h[1] == env.COOPERATE for h in env.history) / len(env.history)
    }
    
    # Restore original settings if changed
    if original_opponent is not None:
        env.opponent_strategy = original_opponent
    
    if original_num_rounds is not None:
        env.num_rounds = original_num_rounds
    
    return match_results 