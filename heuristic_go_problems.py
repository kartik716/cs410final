from typing import Optional, Any, Union
import numpy as np
from go_search_problem import GoState, HeuristicGoProblem
BLACK = 0
WHITE = 1

class GoProblemSimpleHeuristic(HeuristicGoProblem):
    def __init__(self, size: int = 5, state=None, player_to_move: int = 0):
        super().__init__(size=size, state=state, player_to_move=player_to_move)

    def heuristic(self, state, player_index):
        """
        Very simple heuristic that just compares the number of pieces for each player
        
        Having more pieces than the opponent means that some were captured, capturing is generally good.
        Returns value from BLACK's perspective: positive = good for BLACK, negative = good for WHITE.
        """
        black_stones = len(state.get_pieces_coordinates(BLACK))
        white_stones = len(state.get_pieces_coordinates(WHITE))

        return black_stones - white_stones

    def __str__(self) -> str:
        return "Simple Heuristic"


class GoProblemLearnedHeuristic(HeuristicGoProblem):
    def __init__(self, model=None, size: int = 5, state=None, player_to_move: int = 0):
        super().__init__(size=size, state=state, player_to_move=player_to_move)
        self.model = model

    def encoding(self, state):
        pass

    def heuristic(self, state, player_index):
        pass

    def __str__(self) -> str:
        return "Learned Heuristic"
