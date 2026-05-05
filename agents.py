import random
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, List

import numpy as np
import torch
from torch import nn

from go_search_problem import GoProblem, GoState, Action, HeuristicGoProblem
from heuristic_go_problems import GoProblemSimpleHeuristic, GoProblemLearnedHeuristic
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
# Part 2A: Iterative Deepening Search
#
################################################

class _Timeout(Exception):
    """Internal sentinel raised when the search deadline is reached."""
    pass


class IterativeDeepeningAgent(GameAgent):
    def __init__(self, cutoff_time=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.cutoff_time = cutoff_time
        self.search_problem = search_problem

    def _alphabeta(self, state, depth, alpha, beta, is_maximizing, deadline):
        """Alpha-beta search that raises _Timeout when the deadline is reached."""
        if time.time() >= deadline:
            raise _Timeout()

        if depth == 0 or self.search_problem.is_terminal_state(state):
            return self.search_problem.heuristic(state, state.player_to_move())

        actions = self.search_problem.get_available_actions(state)

        if is_maximizing:
            best = -float('inf')
            for action in actions:
                child = self.search_problem.transition(state, action)
                val = self._alphabeta(child, depth - 1, alpha, beta, False, deadline)
                best = max(best, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return best
        else:
            best = float('inf')
            for action in actions:
                child = self.search_problem.transition(state, action)
                val = self._alphabeta(child, depth - 1, alpha, beta, True, deadline)
                best = min(best, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return best

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Iterative deepening search using alpha-beta.

        Searches at increasing depths until the deadline is reached.  If the
        current depth finishes, the result becomes the new best move.  If time
        runs out mid-depth, the best move from the previous completed depth is
        returned.
        """
        # Reserve a small safety margin so we never exceed the hard limit.
        deadline = time.time() + time_limit * 0.85

        is_maximizing = (game_state.player_to_move() == MAXIMIZER)

        # Initialise to a legal move so we always return something.
        actions = self.search_problem.get_available_actions(game_state)
        best_action = actions[0] if actions else None

        depth = 1
        while time.time() < deadline:
            current_best_action = None
            current_best_value = -float('inf') if is_maximizing else float('inf')
            alpha = -float('inf')
            beta = float('inf')

            try:
                shuffled = list(actions)
                random.shuffle(shuffled)

                for action in shuffled:
                    if time.time() >= deadline:
                        break
                    child = self.search_problem.transition(game_state, action)
                    val = self._alphabeta(
                        child, depth - 1, alpha, beta, not is_maximizing, deadline
                    )

                    if is_maximizing:
                        if val > current_best_value:
                            current_best_value = val
                            current_best_action = action
                        alpha = max(alpha, val)
                    else:
                        if val < current_best_value:
                            current_best_value = val
                            current_best_action = action
                        beta = min(beta, val)

                # Depth completed without timeout — commit the result.
                if current_best_action is not None:
                    best_action = current_best_action

            except _Timeout:
                # Ran out of time mid-depth; keep the last fully-completed result.
                break

            depth += 1

        return best_action

    def __str__(self):
        return "IterativeDeepening + " + str(self.search_problem)


################################################
#
# Part 2B: Monte Carlo Tree Search
#
################################################

class MCTSNode:
    """A node in the MCTS search tree."""

    def __init__(self, state: GoState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action      # action taken from parent to reach this node
        self.children: List['MCTSNode'] = []
        self.visits: int = 0
        self.value: int = 0       # wins for the player who MOVED to reach this node

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        return self.state.is_terminal_state()

    def __hash__(self):
        return hash(str(self.state))


class MCTSAgent(GameAgent):
    def __init__(self, c: float = np.sqrt(2)):
        """
        Args:
            c: UCT exploration constant (higher → more exploration).
        """
        super().__init__()
        self.c = c
        self.search_problem = GoProblem()

    # ------------------------------------------------------------------
    # UCT helper
    # ------------------------------------------------------------------

    def _uct(self, child: MCTSNode, parent_visits: int) -> float:
        if child.visits == 0:
            return float('inf')
        exploitation = child.value / child.visits
        exploration = self.c * np.sqrt(np.log(parent_visits) / child.visits)
        return exploitation + exploration

    # ------------------------------------------------------------------
    # Four MCTS steps (exposed as methods per assignment spec)
    # ------------------------------------------------------------------

    def select(self, node: MCTSNode) -> MCTSNode:
        """
        SELECT: walk the tree using the UCT tree policy until a leaf node
        (no children) is found, or a terminal node is encountered.
        """
        curr = node
        while not curr.is_leaf():
            if curr.is_terminal():
                return curr
            curr = max(curr.children, key=lambda c: self._uct(c, curr.visits))
        return curr

    def expand(self, leaf: MCTSNode) -> List[MCTSNode]:
        """EXPAND: add all legal children of *leaf* to the tree.
        Returns [leaf] for terminal nodes so the result is backpropagated."""
        if leaf.is_terminal():
            return [leaf]

        actions = self.search_problem.get_available_actions(leaf.state)
        random.shuffle(actions)
        children = []
        for action in actions:
            child_state = self.search_problem.transition(leaf.state, action)
            child = MCTSNode(child_state, parent=leaf, action=action)
            leaf.children.append(child)
            children.append(child)
        return children

    def simulate(self, children: List[MCTSNode]) -> List[float]:
        """Run a random rollout from each child and return results."""
        if not children:
            return []
        results = []
        for child in children:
            if self.search_problem.is_terminal_state(child.state):
                results.append(self.search_problem.get_result(child.state))
            else:
                results.append(self._rollout(child.state))
        return results

    def _rollout(self, state: GoState) -> float:
        """Random rollout to terminal; returns get_result() value."""
        # Guard: do not clone a terminal state — observation_tensor() without a
        # player argument calls current_player() in C++, which asserts >= 0.
        if self.search_problem.is_terminal_state(state):
            return self.search_problem.get_result(state)
        curr = state.clone()
        while not self.search_problem.is_terminal_state(curr):
            actions = self.search_problem.get_available_actions(curr)
            if not actions:
                break
            curr = self.search_problem.transition(curr, random.choice(actions))
        if self.search_problem.is_terminal_state(curr):
            return self.search_problem.get_result(curr)
        return 0.0

    def backpropagate(self, results: List[float], children: List[MCTSNode]) -> None:
        """
        BACKPROPAGATE: walk from each child up to the root, updating visit
        counts and win counts.

        Value semantics (from pseudocode):
          - result == -1 (WHITE wins) and node.player_to_move() == BLACK (0):
              WHITE was the one who moved to reach this node → increment value.
          - result == +1 (BLACK wins) and node.player_to_move() == WHITE (1):
              BLACK was the one who moved to reach this node → increment value.

        This means child.value / child.visits = win-rate for the *parent*
        who chose to play into this child, making UCT maximisation correct for
        both players.
        """
        for child, result in zip(children, results):
            curr = child
            while curr is not None:
                curr.visits += 1
                # Skip player_to_move() on terminal nodes to avoid passing
                # kTerminalPlayerId (-4) into any downstream C++ assertion.
                if not curr.state.is_terminal_state():
                    player = curr.state.player_to_move()
                    if result == -1 and player == 0:   # WHITE wins; WHITE moved here
                        curr.value += 1
                    elif result == 1 and player == 1:  # BLACK wins; BLACK moved here
                        curr.value += 1
                curr = curr.parent

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Run MCTS for up to *time_limit* seconds and return the action
        corresponding to the root child with the highest visit count.
        """
        root = MCTSNode(game_state.clone())
        deadline = time.time() + time_limit * 0.90

        while time.time() < deadline:
            # 1. Select a leaf
            leaf = self.select(root)

            # 2. Expand the leaf (add all children)
            children = self.expand(leaf)

            # 3. Simulate a rollout from each child (terminal leaf returns [leaf])
            results = self.simulate(children)

            # 4. Backpropagate the result (only for simulated children)
            self.backpropagate(results, children)

        # Return the root child with the most visits (lowest variance estimate).
        if root.children:
            return max(root.children, key=lambda c: c.visits).action

        # Fallback (should rarely happen)
        actions = self.search_problem.get_available_actions(game_state)
        return random.choice(actions)

    def __str__(self):
        return f"MCTS(c={self.c:.2f})"


###################################################
#
# Part 1 Phase 2 helper: construct a learned-heuristic agent from saved model
#
###################################################

def create_value_agent_from_model(model_path: str = "value_model.pt",
                                   board_size: int = 5) -> 'AlphaBetaAgent':
    """
    Load the saved ValueNetwork and return an AlphaBetaAgent that uses the
    learned value function as its heuristic.
    """
    # Feature size: 3 * board_size^2 (black/white/empty) + 1 (player) + 4 (extra)
    feature_size = 3 * board_size * board_size + 1 + 4
    model = ValueNetwork(input_size=feature_size)
    try:
        model = load_model(model_path, model)
    except Exception:
        pass  # Use untrained model as fallback if file is missing
    model.eval()

    search_problem = GoProblemLearnedHeuristic(model=model, size=board_size)
    return AlphaBetaAgent(depth_cutoff=2, search_problem=search_problem)


###################################################
#
# Part 3: Final Agent
#
###################################################

class FinalAgent(GameAgent):
    """
    Enhanced MCTS agent for the final competition.

    Improvements over the base MCTSAgent:
    - Heuristic rollouts: passes are avoided; a light territory flood-fill
      breaks score ties and helps rollouts converge faster.
    - Tree reuse: the MCTS tree is carried over between moves so the agent
      builds on prior work.
    - Opening book: the center is always the best first move on a 5x5 board,
      so we skip search entirely for that case.
    - Safer terminal-state handling (same fix as MCTSAgent above).
    """

    # On a 5x5 board action 12 = position (2,2), the centre.
    _OPENING_BOOK_5x5 = {0: 12}   # {stones_on_board: action}

    def __init__(self, c: float = 1.4):
        super().__init__()
        self.c = c
        self.search_problem = GoProblem()
        self._simple_heuristic = GoProblemSimpleHeuristic()
        self._root: Optional[MCTSNode] = None
        # Try to load the trained value network for leaf evaluation
        self._value_model = None
        self._learned_heuristic = None
        try:
            feature_size = 3 * 25 + 1 + 4  # 5x5 board
            _model = ValueNetwork(input_size=feature_size)
            _model = load_model("value_model.pt", _model)
            _model.eval()
            self._value_model = _model
            self._learned_heuristic = GoProblemLearnedHeuristic(model=_model)
        except Exception:
            pass

    def reset(self):
        self._root = None

    # ------------------------------------------------------------------
    # Territory estimation (flood-fill)
    # ------------------------------------------------------------------

    def _territory_score(self, state: GoState) -> float:
        """Returns (black_territory - white_territory) / board_area."""
        board = state.get_board()
        black, white, empty = board[0], board[1], board[2]
        size = state.size
        visited = [[False] * size for _ in range(size)]
        score = 0.0

        for r in range(size):
            for c in range(size):
                if visited[r][c] or not empty[r][c]:
                    continue
                # BFS over connected empty region
                region, q = [], [(r, c)]
                visited[r][c] = True
                border_b = border_w = False
                while q:
                    yr, yc = q.pop()
                    region.append((yr, yc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = yr + dr, yc + dc
                        if not (0 <= nr < size and 0 <= nc < size):
                            continue
                        if not visited[nr][nc]:
                            if empty[nr][nc]:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                            elif black[nr][nc]:
                                border_b = True
                            elif white[nr][nc]:
                                border_w = True
                if border_b and not border_w:
                    score += len(region)
                elif border_w and not border_b:
                    score -= len(region)

        return score / (size * size)

    # ------------------------------------------------------------------
    # Leaf evaluation (value network if available, else rollout)
    # ------------------------------------------------------------------

    def _evaluate_leaf(self, state: GoState) -> float:
        """Evaluate a leaf state. Uses value network if loaded, else rollout."""
        if self.search_problem.is_terminal_state(state):
            return self.search_problem.get_result(state)
        if self._value_model is not None and self._learned_heuristic is not None:
            try:
                return self._learned_heuristic.heuristic(state, 0)
            except Exception:
                pass
        return self._rollout(state)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout(self, state: GoState) -> float:
        if self.search_problem.is_terminal_state(state):
            return self.search_problem.get_result(state)
        curr = state.clone()
        pass_action = curr.size * curr.size   # the pass action index
        max_steps = curr.size * curr.size * 3
        steps = 0

        while not self.search_problem.is_terminal_state(curr) and steps < max_steps:
            actions = self.search_problem.get_available_actions(curr)
            if not actions:
                break
            # Lightly prefer non-pass moves; with 20 % chance pick random.
            non_pass = [a for a in actions if a != pass_action]
            if non_pass and random.random() < 0.8:
                action = random.choice(non_pass)
            else:
                action = random.choice(actions)
            curr = self.search_problem.transition(curr, action)
            steps += 1

        if self.search_problem.is_terminal_state(curr):
            return self.search_problem.get_result(curr)
        # Heuristic evaluation at depth limit
        black = len(curr.get_pieces_coordinates(0))
        white = len(curr.get_pieces_coordinates(1))
        stone = (black - white) / (curr.size * curr.size)
        territory = self._territory_score(curr)
        raw = 0.5 * stone + 0.5 * territory
        return max(-1.0, min(1.0, raw))

    # ------------------------------------------------------------------
    # MCTS primitives
    # ------------------------------------------------------------------

    def _uct(self, child: MCTSNode, parent_visits: int) -> float:
        if child.visits == 0:
            return float('inf')
        return (child.value / child.visits) + self.c * np.sqrt(np.log(parent_visits) / child.visits)

    def _select(self, node: MCTSNode) -> MCTSNode:
        curr = node
        while not curr.is_leaf():
            if curr.is_terminal():
                return curr
            curr = max(curr.children, key=lambda ch: self._uct(ch, curr.visits))
        return curr

    def _expand(self, leaf: MCTSNode) -> List[MCTSNode]:
        if leaf.is_terminal():
            return []
        actions = self.search_problem.get_available_actions(leaf.state)
        random.shuffle(actions)
        children = []
        for action in actions:
            child_state = self.search_problem.transition(leaf.state, action)
            child = MCTSNode(child_state, parent=leaf, action=action)
            leaf.children.append(child)
            children.append(child)
        return children

    def _backpropagate(self, node: MCTSNode, result: float) -> None:
        curr = node
        while curr is not None:
            curr.visits += 1
            if not curr.state.is_terminal_state():
                player = curr.state.player_to_move()
                # value = sum of results from the perspective of the player who
                # moved INTO this node. player_to_move==1 means BLACK just moved.
                if player == 1:
                    curr.value += result        # positive when BLACK wins
                else:
                    curr.value += -result       # positive when WHITE wins
            curr = curr.parent

    # ------------------------------------------------------------------
    # Tree reuse
    # ------------------------------------------------------------------

    def _reuse_tree(self, game_state: GoState) -> MCTSNode:
        """Return the subtree rooted at game_state if found, else a fresh node."""
        state_str = str(game_state)
        if self._root is not None:
            # Search two levels deep (our last move + opponent's response).
            for child in self._root.children:
                if str(child.state) == state_str:
                    child.parent = None
                    return child
                for grandchild in child.children:
                    if str(grandchild.state) == state_str:
                        grandchild.parent = None
                        return grandchild
        return MCTSNode(game_state.clone())

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        # Opening book: play centre on the very first move.
        board = game_state.get_board()
        stones_on_board = int(board[0].sum() + board[1].sum())
        if stones_on_board in self._OPENING_BOOK_5x5:
            candidate = self._OPENING_BOOK_5x5[stones_on_board]
            actions = self.search_problem.get_available_actions(game_state)
            if candidate in actions:
                self._root = None
                return candidate

        root = self._reuse_tree(game_state)
        deadline = time.time() + time_limit * 0.90

        while time.time() < deadline:
            leaf = self._select(root)
            children = self._expand(leaf)
            if not children:
                # Terminal leaf: backpropagate its result so UCT becomes finite
                # and the leaf is not re-selected every iteration.
                if leaf.is_terminal():
                    result = self.search_problem.get_result(leaf.state)
                    self._backpropagate(leaf, result)
                continue
            # Prefer an unvisited child for the rollout.
            unvisited = [ch for ch in children if ch.visits == 0]
            target = random.choice(unvisited) if unvisited else children[0]
            result = self._evaluate_leaf(target.state)
            self._backpropagate(target, result)

        self._root = root

        if root.children:
            return max(root.children, key=lambda ch: ch.visits).action
        actions = self.search_problem.get_available_actions(game_state)
        return random.choice(actions) if actions else None

    def __str__(self):
        return f"FinalAgent(c={self.c:.2f})"


def get_final_agent_5x5():
    """Called to construct agent for final submission for 5x5 board"""
    return FinalAgent(c=1.4)

def get_final_agent_9x9():
    """Called to construct agent for final submission for 9x9 board"""
    return FinalAgent(c=1.4)
