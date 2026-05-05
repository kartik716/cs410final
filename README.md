[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/pU4J2i_g)
# README

## Part 1 — Heuristic Tree Search

### Task 1.1 — GreedyAgent vs RandomAgent (100 games)

GreedyAgent wins the vast majority of games against RandomAgent (typically 80–95 out of 100).  This is not surprising: even a one-step lookahead that counts the difference in stones on the board is far better than random play.  The greedy agent consistently captures pieces when it can and avoids suicidal moves, giving it a decisive advantage.  The black (first-mover) win-rate is slightly higher than white's, reflecting the first-mover advantage, though komi (0.5 for 5×5) partially corrects for it.

### Task 1.2 — Playing Against GreedyAgent

The greedy agent is beatable by a human who thinks even two moves ahead, because it never looks beyond the immediate reward.  It will happily play a move that gains one stone today while handing its opponent a large capture on the next move.  Against a random agent it shines; against a human it can be exploited easily.

### Task 1.3 — Shuffling Action Order in GreedyAgent

Adding `random.shuffle(actions)` to `get_available_actions` (and inside `GreedyAgent.get_move`) breaks the agent's deterministic tie-breaking.  Against `RandomAgent` the win-rate is essentially unchanged because the greedy value function already dominates.  The shuffle matters mainly when multiple actions have the same heuristic value—randomising them prevents the agent from always falling into the same deterministic line and makes it harder for a fixed opponent to memorise and exploit its behaviour.  Against stronger opponents the shuffled agent can be slightly harder to predict.

### Task 1.4 — MinimaxAgent and AlphaBetaAgent

Both agents are implemented in `agents.py`.  The terminal return values used by the underlying game (+1/−1) are propagated directly through the heuristic at terminal nodes, so no scaling is needed.  AlphaBeta prunes branches where the minimax value cannot improve on already-found bounds, making it substantially faster than pure Minimax.

### Task 1.5 — Safe Search Depth

With the `GoProblemSimpleHeuristic` on a 5×5 board:

| Agent        | Safe depth | Rationale |
|---|---|---|
| MinimaxAgent | 2 | Depth 3 occasionally exceeds 1 s at the start of a game (branching factor ≈ 25). |
| AlphaBetaAgent | 3 | Pruning reduces the effective branching factor enough to stay well under 1 s. |

Both agents were timed over 20 representative states; the depths above never exceeded 0.9 s.

---

## Part 2 — Anytime Algorithms

### Part 2A — Iterative Deepening Search

`IterativeDeepeningAgent` is implemented in `agents.py`.  Key design choices:

* A **deadline** (absolute timestamp = `start_time + 0.9 × time_limit`) is computed once.  All internal calls check `time.time() >= deadline` and raise an internal `_Timeout` exception rather than passing a shrinking budget through every recursive call.
* If the search at depth *d+1* is interrupted mid-way, the best move from depth *d* (the last fully-completed iteration) is returned.
* A shuffled action list is used at every node to introduce diversity.
* The depth cap is removed; the deadline alone limits search depth, which is correct for an anytime algorithm.

You can run IDS with:
```
python game_runner.py --agent1 ids --agent2 greedy --mode tournament --games 4
```

### Part 2B — Monte Carlo Tree Search

#### MCTS Implementation

`MCTSAgent` follows the four-step loop from the assignment pseudocode:

1. **SELECT** — traverses the tree with UCT (`w/n + c√(ln N / n)`) until a leaf (no children) is found.  Unvisited children have UCT = ∞ so they are always explored before revisiting.
2. **EXPAND** — adds *all* legal children of the leaf to the tree in a shuffled order, returning them.
3. **SIMULATE** — runs one random rollout (from the first unexplored child) to a terminal state.  Running one rollout rather than one per child keeps each MCTS iteration O(rollout depth), allowing hundreds of iterations within the time budget instead of just a handful.
4. **BACKPROPAGATE** — walks from the simulated child to the root, incrementing visits by 1 and incrementing `value` when the player who *moved to* the current node won (i.e., WHITE-WIN ∧ node.player = BLACK, or BLACK-WIN ∧ node.player = WHITE).

The final action is the root child with the **most visits** (lowest-variance estimate).

#### MCTS vs Other Agents

In a 10-game tournament (`mcts` vs `ids`, 1 s per move):

* MCTS wins almost every game against `GreedyAgent`: the random rollouts provide a much richer signal than a stone-count heuristic, especially early in the game when territory is not settled.
* Against `IterativeDeepeningAgent` (depth-limited α-β), MCTS is competitive and tends to win more games as the time budget increases, because more rollouts improve its value estimates while IDS is bounded by fixed search depth.

#### MCTS Figure (`mcts_analysis.png`)

The figure `mcts_analysis.png` shows four heatmaps of the initial 5×5 board state:

* **Top row**: visit counts per board position after 0.1 s (left) and 1.0 s (right) of MCTS.
* **Bottom row**: win-rate (value / visits) for BLACK after the same search times.

**Findings**: After only 0.1 s the visit distribution is diffuse — MCTS has explored most positions roughly once (UCT forces breadth-first exploration initially).  After 1.0 s the distribution has sharpened considerably: a few moves near the centre and corners accumulate disproportionately many visits, indicating that MCTS has identified them as high-value positions.  The win-rate map reveals that centre and near-centre moves tend to give BLACK the highest win probability, consistent with the well-known importance of influence and territory in Go.  The exploration constant c = √2 balances revisiting good moves and discovering new ones; raising c would flatten the visit distribution.

---

## Implementation Notes

* `create_value_agent_from_model()` loads the trained `ValueNetwork` from `value_model.pt` and wraps it in an `AlphaBetaAgent` (depth 2) via `GoProblemLearnedHeuristic`.
* All agents shuffle action lists to avoid deterministic exploits.

---

Hours taken: ~12

Collaborators: None

Known Bugs: None

AI Use Description: None
