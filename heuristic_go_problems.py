from typing import Optional, Any, Union
import numpy as np
import torch
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

        # Normalize by board size
        return (black_stones - white_stones) / (state.size * state.size)

    def __str__(self) -> str:
        return "Simple Heuristic"


class GoProblemLearnedHeuristic(HeuristicGoProblem):
    def __init__(self, model=None, size: int = 5, state=None, player_to_move: int = 0):
        super().__init__(size=size, state=state, player_to_move=player_to_move)
        self.model = model

    def encoding(self, state):
        """
        Get encoding of state (convert state to features)
        This uses the same feature extraction as in supervised learning
        """
        board_size = state.size
        board = state.get_board()
        
        # Expressive encoding
        black_pieces = board[0].flatten()
        white_pieces = board[1].flatten()
        empty_spaces = board[2].flatten()
        
        # Player to move
        player_to_move = [board[3][0][0]]
        
        # Stone count difference
        num_black = np.sum(black_pieces)
        num_white = np.sum(white_pieces)
        stone_diff = (num_black - num_white) / (board_size * board_size)
        
        # Liberty counts approximation
        black_liberties = 0
        white_liberties = 0
        for y in range(board_size):
            for x in range(board_size):
                if black_pieces[y * board_size + x] == 1:
                    if y > 0 and empty_spaces[(y-1) * board_size + x] == 1:
                        black_liberties += 1
                    if y < board_size - 1 and empty_spaces[(y+1) * board_size + x] == 1:
                        black_liberties += 1
                    if x > 0 and empty_spaces[y * board_size + (x-1)] == 1:
                        black_liberties += 1
                    if x < board_size - 1 and empty_spaces[y * board_size + (x+1)] == 1:
                        black_liberties += 1
                elif white_pieces[y * board_size + x] == 1:
                    if y > 0 and empty_spaces[(y-1) * board_size + x] == 1:
                        white_liberties += 1
                    if y < board_size - 1 and empty_spaces[(y+1) * board_size + x] == 1:
                        white_liberties += 1
                    if x > 0 and empty_spaces[y * board_size + (x-1)] == 1:
                        white_liberties += 1
                    if x < board_size - 1 and empty_spaces[y * board_size + (x+1)] == 1:
                        white_liberties += 1
        
        black_liberties_feat = black_liberties / (board_size * board_size * 4)
        white_liberties_feat = white_liberties / (board_size * board_size * 4)
        liberty_diff = black_liberties_feat - white_liberties_feat
        
        # Center control
        center = board_size // 2
        center_control_black = 0.0
        center_control_white = 0.0
        for y in range(board_size):
            for x in range(board_size):
                dist_to_center = abs(y - center) + abs(x - center)
                weight = 1.0 / (dist_to_center + 1)
                if black_pieces[y * board_size + x] == 1:
                    center_control_black += weight
                elif white_pieces[y * board_size + x] == 1:
                    center_control_white += weight
        
        center_control_black = center_control_black / (board_size * board_size)
        center_control_white = center_control_white / (board_size * board_size)
        
        # Combine all features
        features = []
        features.extend([float(x) for x in black_pieces])
        features.extend([float(x) for x in white_pieces])
        features.extend([float(x) for x in empty_spaces])
        features.extend(player_to_move)
        features.append(float(stone_diff))
        features.append(float(liberty_diff))
        features.append(float(center_control_black))
        features.append(float(center_control_white))
        
        return features

    def heuristic(self, state, player_index):
        """
        Return heuristic (value) of current state using learned model
        """
        if self.model is None:
            return 0
        
        features = self.encoding(state)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        with torch.no_grad():
            value = self.model(features_tensor).item()
        
        # Flip sign for WHITE player
        if player_index == 1:
            value = -value
        
        return value

    def __str__(self) -> str:
        return "Learned Heuristic"