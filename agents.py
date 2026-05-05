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
        # Shuffle for randomness (not deterministic)
        random.shuffle(actions)
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
        search_problem = self.search_problem

        if game_state.player_to_move() == MAXIMIZER:
            best_value = -float('inf')
        else:
            best_value = float('inf')
        best_action = None

        actions = search_problem.get_available_actions(game_state)
        # Shuffle actions to break ties randomly
        random.shuffle(actions)

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

        return best_action

    def __str__(self):
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

    def minimax(self, state, depth, is_maximizing):
        """Recursive minimax algorithm with depth cutoff."""
        if depth == 0 or self.search_problem.is_terminal_state(state):
            # Return heuristic value from current player's perspective
            return self.search_problem.heuristic(state, state.player_to_move())
        
        if is_maximizing:
            max_eval = -float('inf')
            actions = self.search_problem.get_available_actions(state)
            for action in actions:
                new_state = self.search_problem.transition(state, action)
                eval = self.minimax(new_state, depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            actions = self.search_problem.get_available_actions(state)
            for action in actions:
                new_state = self.search_problem.transition(state, action)
                eval = self.minimax(new_state, depth - 1, True)
                min_eval = min(min_eval, eval)
            return min_eval

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using minimax algorithm
        """
        is_maximizing = (game_state.player_to_move() == MAXIMIZER)
        best_action = None
        
        if is_maximizing:
            best_value = -float('inf')
        else:
            best_value = float('inf')
        
        actions = self.search_problem.get_available_actions(game_state)
        random.shuffle(actions)
        
        for action in actions:
            new_state = self.search_problem.transition(game_state, action)
            # Depth-1 recursion because we already made one move
            value = self.minimax(new_state, self.depth - 1, not is_maximizing)
            
            if is_maximizing:
                if value > best_value:
                    best_value = value
                    best_action = action
            else:
                if value < best_value:
                    best_value = value
                    best_action = action
        
        return best_action

    def __str__(self):
        return f"MinimaxAgent w/ depth {self.depth} + " + str(self.search_problem)


class AlphaBetaAgent(GameAgent):
    def __init__(self, depth_cutoff=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth_cutoff
        self.search_problem = search_problem

    def alphabeta(self, state, depth, alpha, beta, is_maximizing):
        """Recursive alpha-beta pruning algorithm."""
        if depth == 0 or self.search_problem.is_terminal_state(state):
            return self.search_problem.heuristic(state, state.player_to_move())
        
        if is_maximizing:
            max_eval = -float('inf')
            actions = self.search_problem.get_available_actions(state)
            for action in actions:
                new_state = self.search_problem.transition(state, action)
                eval = self.alphabeta(new_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            actions = self.search_problem.get_available_actions(state)
            for action in actions:
                new_state = self.search_problem.transition(state, action)
                eval = self.alphabeta(new_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using alpha-beta algorithm
        """
        is_maximizing = (game_state.player_to_move() == MAXIMIZER)
        best_action = None
        
        if is_maximizing:
            best_value = -float('inf')
        else:
            best_value = float('inf')
        
        actions = self.search_problem.get_available_actions(game_state)
        random.shuffle(actions)
        
        alpha = -float('inf')
        beta = float('inf')
        
        for action in actions:
            new_state = self.search_problem.transition(game_state, action)
            value = self.alphabeta(new_state, self.depth - 1, alpha, beta, not is_maximizing)
            
            if is_maximizing:
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, value)
            else:
                if value < best_value:
                    best_value = value
                    best_action = action
                beta = min(beta, value)
        
        return best_action

    def __str__(self):
        return f"AlphaBeta w/ depth {self.depth} + " + str(self.search_problem)


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
    
    def alphabeta_with_timeout(self, state, depth, alpha, beta, is_maximizing, start_time, time_limit):
        """Alpha-beta with time checking."""
        if depth == 0 or self.search_problem.is_terminal_state(state):
            return self.search_problem.heuristic(state, state.player_to_move())
        
        if is_maximizing:
            max_eval = -float('inf')
            actions = self.search_problem.get_available_actions(state)
            for action in actions:
                if time.time() - start_time > time_limit:
                    return max_eval  # Timeout, return current best
                new_state = self.search_problem.transition(state, action)
                eval = self.alphabeta_with_timeout(new_state, depth - 1, alpha, beta, False, start_time, time_limit)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            actions = self.search_problem.get_available_actions(state)
            for action in actions:
                if time.time() - start_time > time_limit:
                    return min_eval
                new_state = self.search_problem.transition(state, action)
                eval = self.alphabeta_with_timeout(new_state, depth - 1, alpha, beta, True, start_time, time_limit)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using iterative deepening algorithm (+ alpha-beta).
        """
        start_time = time.time()
        is_maximizing = (game_state.player_to_move() == MAXIMIZER)
        best_action = None
        depth = 1
        
        # Allocate ~90% of time for search, reserve 10% for overhead
        search_time_limit = time_limit * 0.9
        
        while True:
            # Check if we have time for another iteration
            elapsed = time.time() - start_time
            if elapsed > search_time_limit:
                break
            
            # Run alpha-beta at current depth
            best_value = -float('inf') if is_maximizing else float('inf')
            current_best = None
            alpha = -float('inf')
            beta = float('inf')
            
            actions = self.search_problem.get_available_actions(game_state)
            random.shuffle(actions)
            
            for action in actions:
                # Check timeout before evaluating each action
                if time.time() - start_time > search_time_limit:
                    break
                    
                new_state = self.search_problem.transition(game_state, action)
                value = self.alphabeta_with_timeout(
                    new_state, depth - 1, alpha, beta, not is_maximizing, 
                    start_time, search_time_limit - (time.time() - start_time)
                )
                
                if is_maximizing:
                    if value > best_value:
                        best_value = value
                        current_best = action
                    alpha = max(alpha, value)
                else:
                    if value < best_value:
                        best_value = value
                        current_best = action
                    beta = min(beta, value)
            
            if current_best is not None:
                best_action = current_best
            
            depth += 1
            # Safety: don't go too deep
            if depth > 10:
                break
        
        return best_action

    def __str__(self):
        return f"IterativeDeepening + " + str(self.search_problem)


class MCTSNode:
    def __init__(self, state, parent=None, children=None, action=None):
        self.state = state
        self.parent = parent
        if children is None:
            children = []
        self.children = children
        self.visits = 0
        self.value = 0
        self.action = action
        self.untried_actions = None

    def __hash__(self):
        return hash(str(self.state))  # Simple hashing for state


class MCTSAgent(GameAgent):
    def __init__(self, c=np.sqrt(2)):
        """
        Args: 
            c (float): exploration constant of UCT algorithm
        """
        super().__init__()
        self.c = c
        self.search_problem = GoProblem()
    
    def select(self, node):
        """Select a child node using UCT formula."""
        total_visits = node.visits
        best_child = None
        best_uct = -float('inf')
        
        for child in node.children:
            if child.visits == 0:
                return child
            exploitation = child.value / child.visits
            exploration = self.c * np.sqrt(np.log(total_visits) / child.visits)
            uct_value = exploitation + exploration
            
            if uct_value > best_uct:
                best_uct = uct_value
                best_child = child
        
        return best_child
    
    def expand(self, node):
        """Expand a node by adding a new child."""
        if node.untried_actions is None:
            node.untried_actions = self.search_problem.get_available_actions(node.state)
            random.shuffle(node.untried_actions)
        
        if node.untried_actions:
            action = node.untried_actions.pop()
            new_state = self.search_problem.transition(node.state, action)
            child_node = MCTSNode(new_state, parent=node, action=action)
            node.children.append(child_node)
            return child_node
        return None
    
    def simulate(self, state, max_depth=20):
        """Simulate a random rollout from the given state."""
        current_state = state.clone()
        depth = 0
        
        while not self.search_problem.is_terminal_state(current_state) and depth < max_depth:
            actions = self.search_problem.get_available_actions(current_state)
            if not actions:
                break
            action = random.choice(actions)
            current_state = self.search_problem.transition(current_state, action)
            depth += 1
        
        # Get result from BLACK's perspective
        if self.search_problem.is_terminal_state(current_state):
            return self.search_problem.get_result(current_state)
        else:
            # Approximate value based on simple heuristic
            simple_heuristic = GoProblemSimpleHeuristic()
            return simple_heuristic.heuristic(current_state, 0)
    
    def backpropagate(self, node, result):
        """Backpropagate the simulation result up the tree."""
        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent
            # Flip result for opponent's perspective
            result = -result
    
    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using MCTS algorithm
        """
        root = MCTSNode(game_state.clone())
        root.untried_actions = self.search_problem.get_available_actions(game_state)
        start_time = time.time()
        
        num_iterations = 0
        while time.time() - start_time < time_limit * 0.95:  # Leave a small buffer
            node = root
            
            # Selection
            while node.untried_actions is None and node.children:
                node = self.select(node)
            
            # Expansion
            if node.untried_actions is not None and node.untried_actions:
                node = self.expand(node)
            
            # Simulation
            if node is not None:
                result = self.simulate(node.state)
                
                # Backpropagation
                self.backpropagate(node, result)
            
            num_iterations += 1
            
            # Early stopping if we've done enough iterations
            if num_iterations > 1000 and root.visits > 0:
                # Check if we have a clear best move
                best_visits = max(child.visits for child in root.children) if root.children else 0
                if best_visits > root.visits * 0.8:
                    break
        
        # Select best action based on highest visit count
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            return best_child.action
        
        # Fallback to random move
        actions = self.search_problem.get_available_actions(game_state)
        return random.choice(actions)

    def __str__(self):
        return "MCTS"


###################################################
#
# Part 3: Final Agent
#
###################################################

def get_final_agent_5x5():
    """Called to construct agent for final submission for 5x5 board"""
    # Use MCTS with learned heuristic for best performance
    return MCTSAgent(c=1.0)

def get_final_agent_9x9():
    """Called to construct agent for final submission for 9x9 board"""
    return MCTSAgent(c=1.0)