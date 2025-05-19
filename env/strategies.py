import numpy as np
from typing import List, Tuple, Optional


class Strategy:
    """Base class for Iterated Prisoner's Dilemma strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def action(self, history: List[Tuple[int, int]], player_idx: int = 0) -> int:
        """
        Determine next action based on game history
        
        Args:
            history: List of tuples (player_action, opponent_action) for each past round
            player_idx: Index of the player using this strategy (0 or 1)
            
        Returns:
            int: 0 for Cooperate, 1 for Defect
        """
        raise NotImplementedError("Strategy subclasses must implement action()")


class TitForTat(Strategy):
    """
    Tit-for-Tat strategy:
    - Cooperate on first move
    - Then copy opponent's previous move
    """
    
    def __init__(self):
        super().__init__("Tit-for-Tat")
    
    def action(self, history: List[Tuple[int, int]], player_idx: int = 0) -> int:
        if not history:  # First move
            return 0  # Cooperate
        
        # Get opponent's index (0 or 1)
        opponent_idx = 1 - player_idx
        
        # Copy opponent's last move
        return history[-1][opponent_idx]


class AlwaysCooperate(Strategy):
    """Always Cooperate strategy"""
    
    def __init__(self):
        super().__init__("Always Cooperate")
    
    def action(self, history: List[Tuple[int, int]], player_idx: int = 0) -> int:
        return 0  # Always cooperate


class AlwaysDefect(Strategy):
    """Always Defect strategy"""
    
    def __init__(self):
        super().__init__("Always Defect")
    
    def action(self, history: List[Tuple[int, int]], player_idx: int = 0) -> int:
        return 1  # Always defect


class RandomStrategy(Strategy):
    """Random strategy with configurable cooperation probability"""
    
    def __init__(self, coop_prob: float = 0.5, seed: Optional[int] = None):
        super().__init__(f"Random(p={coop_prob})")
        self.coop_prob = coop_prob
        self.rng = np.random.RandomState(seed)
    
    def action(self, history: List[Tuple[int, int]], player_idx: int = 0) -> int:
        return 0 if self.rng.random() < self.coop_prob else 1


class PavlovStrategy(Strategy):
    """
    Pavlov (Win-Stay, Lose-Shift) strategy:
    - Cooperate on first move
    - If got reward or temptation payoff (CC or DC), repeat last action
    - If got punishment or sucker payoff (DD or CD), change action
    """
    
    def __init__(self):
        super().__init__("Pavlov")
    
    def action(self, history: List[Tuple[int, int]], player_idx: int = 0) -> int:
        if not history:  # First move
            return 0  # Cooperate
        
        # Get player and opponent's last actions
        player_last_action = history[-1][player_idx]
        opponent_last_action = history[-1][1 - player_idx]
        
        # Win-Stay, Lose-Shift logic
        if player_last_action == opponent_last_action:  # CC or DD
            return player_last_action  # Keep same action
        else:  # CD or DC
            return 1 - player_last_action  # Switch action


# Add more strategies as needed, e.g.:
# - Generous Tit-for-Tat
# - Grim Trigger
# - Memory-N strategies 