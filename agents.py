import random
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, List

import numpy as np
import torch
from torch import nn

from go_search_problem import GoProblem, GoState, Action, HeuristicGoProblem
from heuristic_go_problems import GoProblemSimpleHeuristic
from models import ValueNetwork, load_model

MAXIMIZER = 0
MIMIZER = 1

class GameAgent(ABC):
    """Abstract base class for all Go game agents."""
    
    @abstractmethod
    def get_move(self, state: GoState, time_limit: float) -> Action:
        """Get the best move for the given state within the time limit.
        
        Args:
            state: Current game state
            time_limit: Maximum time in seconds to spend on this move
            
        Returns:
            Action index representing the chosen move
        """
        pass

    def reset(self):
        """Reset any internal state of the agent if necessary.
            Called after a new game is started.
        """
        pass


class RandomAgent(GameAgent):
    # An Agent that makes random moves

    def __init__(self):
        self.search_problem = GoProblem()

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        get random move for a given state
        """
        actions = self.search_problem.get_available_actions(game_state)
        return random.choice(actions)

    def __str__(self):
        return "RandomAgent"


class GreedyAgent(GameAgent):
    def __init__(self, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        get move of agent for given game state.
        Greedy agent looks one step ahead with the provided heuristic and chooses the best available action
        (Greedy agent does not consider remaining time)

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        """
        # Create new GoSearchProblem with provided heuristic
        search_problem = self.search_problem

        # Player 0 is maximizing
        if game_state.player_to_move() == MAXIMIZER:
            best_value = -float('inf')
        else:
            best_value = float('inf')
        best_action = None

        # Get Available actions
        actions = search_problem.get_available_actions(game_state)

        # Compare heuristic of every reachable next state
        for action in actions:
            new_state = search_problem.transition(game_state, action)
            value = search_problem.heuristic(new_state, new_state.player_to_move())
            if game_state.player_to_move() == MAXIMIZER:
                if value > best_value:
                    best_value = value
                    best_action = action
            else:
                if value < best_value:
                    best_value = value
                    best_action = action

        # Return best available action
        return best_action

    def __str__(self):
        """
        Description of agent (Greedy + heuristic/search problem used)
        """
        return "GreedyAgent + " + str(self.search_problem)

#############################################
# 
#
# Part 1: Basic Adversarial Search Algorithms
#
#
#############################################

class MinimaxAgent(GameAgent):
    def __init__(self, depth_cutoff=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth_cutoff
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using minimax algorithm

        MinimaxAgents should not consider time limit, they simply search to their specified depth_cutoff
        If your agent is running out of time, you should use a shorter cutoff depth
        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO Part 1: implement get_move method of MinimaxAgent
        pass

    def __str__(self):
        return f"MinimaxAgent w/ depth {self.depth} + " + str(self.search_problem)


class AlphaBetaAgent(GameAgent):
    def __init__(self, depth_cutoff=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth_cutoff
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using alpha-beta algorithm

        AlphaBetaAgents should not consider time limit, they simply search to their specified depth_cutoff
        If your agent is running out of time, you should use a shorter cutoff depth

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO Part 1: implement get_move algorithm of AlphaBeta Agent
        pass

    def __str__(self):
        return f"AlphaBeta w/ depth {self.depth} + " + str(self.search_problem)



def create_value_agent_from_model():
    """
    Create agent object from saved model. 
    This (or other methods like this) will be how your agents will be created in gradescope and in the final tournament.

    In the game_runner file, there is a factory function that will call this function to create an agent.
    You can run games with your agent against other agents by running game_runner.py with the appropriate command line arguments.
    """
    # TODO: Update model path to your saved model
    model_path = "value_model.pt"

    # TODO: Update number of features for your own encoding size
    feature_size = 0
    model = load_model(model_path, ValueNetwork(feature_size))
    heuristic_search_problem = GoProblemLearnedHeuristic(model)

    # TODO: Try with other heuristic agents (IDS/AB/Minimax)
    learned_agent = GreedyAgent(heuristic_search_problem)

    return learned_agent


################################################
#
# Part 2: Advanced Adversarial Search Algorithms
#
################################################

class IterativeDeepeningAgent(GameAgent):
    def __init__(self, cutoff_time=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.cutoff_time = cutoff_time
        self.search_problem = search_problem

    def get_move(self, game_state, time_limit):
        """
        Get move of agent for given game state using iterative deepening algorithm (+ alpha-beta).
        Iterative deepening is a search algorithm that repeatedly searches for a solution to a problem,
        increasing the depth of the search with each iteration.

        The advantage of iterative deepening is that you can stop the search based on the time limit, rather than depth.
        The recommended approach is to modify your implementation of Alpha-beta to stop when the time limit is reached
        and run IDS on that modified version.

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO Part 2: implement get_move algorithm of IterativeDeepeningAgent
        pass

    def __str__(self):
        return f"IterativeDeepneing + " + str(self.search_problem)
    

class MCTSNode:
    def __init__(self, state, parent=None, children=None, action=None):
        # GameState for Node
        self.state = state

        # Parent (MCTSNode)
        self.parent = parent
        
        # Children List of MCTSNodes
        if children is None:
            children = []
        self.children = children
        
        # Number of times this node has been visited in tree search
        self.visits = 0
        
        # Value of node (number of times simulations from children results in black win)
        self.value = 0
        
        # Action that led to this node
        self.action = action

    def __hash__(self):
        """
        Hash function for MCTSNode is hash of state
        """
        return hash(self.state)
    
class MCTSAgent(GameAgent):
    def __init__(self, c=np.sqrt(2)):
        """
        Args: 
            c (float): exploration constant of UCT algorithm
        """
        super().__init__()
        self.c = c

        # Initialize Search problem
        self.search_problem = GoProblem()

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using MCTS algorithm
        
        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO Part 2: Implement MCTS
        pass

    def __str__(self):
        return "MCTS"
    

###################################################
#
# Part 3: Final Agent
#
###################################################

def get_final_agent_5x5():
    """Called to construct agent for final submission for 5x5 board"""
    return MCTSAgent()

def get_final_agent_9x9():
    """Called to construct agent for final submission for 9x9 board"""
    return None
