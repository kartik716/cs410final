[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/pU4J2i_g)
# README

## Part 1 — Heuristic Tree Search

### Task 1.1 — GreedyAgent vs RandomAgent (100 games)

GreedyAgent wins around 80–90 games out of 100. Makes sense — even looking one step ahead at stone count is way better than pure random. One thing that surprised me a little: Black wins more consistently than White, which tracks with first-mover advantage, though the 0.5 komi does help keep it from being totally lopsided.

### Task 1.2 — Playing Against GreedyAgent

Playing a few games manually, GreedyAgent is pretty easy to exploit once you understand what it's doing. It'll always take the locally-best stone capture but has no concept of setting up future captures. So you can bait it into bad positions. Against a random opponent it looks great, against anyone who thinks ahead even two moves it becomes very beatable.

### Task 1.3 — Shuffling Action Order in GreedyAgent

Shuffling barely changed the win rate against RandomAgent — the greedy heuristic dominates regardless of tie-breaking order. The difference matters more when playing against agents that can actually respond strategically. Deterministic tie-breaking can also create exploitable patterns; randomizing makes the agent less predictable even if the overall strength is similar.

### Task 1.4 — MinimaxAgent and AlphaBetaAgent

Both are implemented in agents.py. Terminal state values from pyspiel are +1/-1, so no scaling needed — the heuristic just propagates naturally. AlphaBeta prunes branches where alpha >= beta, which cuts down the search space dramatically especially at higher depths.

### Task 1.5 — Safe Search Depth

On a 5x5 board with GoProblemSimpleHeuristic:

- MinimaxAgent: depth 2 is consistently under 1 second. Depth 3 sometimes exceeds it near the start when there are ~25 legal moves.
- AlphaBetaAgent: depth 3 comfortably fits under 1 second thanks to pruning. Depth 4 can go over occasionally.

---

## Part 2 — Anytime Algorithms

### Part 2A — Iterative Deepening Search

IterativeDeepeningAgent is in agents.py. Key implementation decisions:

- A deadline = start_time + 0.85 * time_limit is set once. Every recursive call checks the deadline and raises a `_Timeout` exception if it's hit.
- If depth d+1 gets interrupted, the agent returns the best move found at depth d.
- Actions are shuffled at each node to add diversity.
- No hard depth cap — the deadline alone controls depth, which is what an anytime algorithm should do.

### Part 2B — Monte Carlo Tree Search

**MCTS Implementation**

MCTSAgent follows the four-step loop from the assignment pseudocode:

- **SELECT**: Walk down the tree using UCT until a leaf is reached. Unvisited children get UCT = ∞ so unexplored moves are always tried first.
- **EXPAND**: Add ALL legal children of the leaf. If the leaf is a terminal state, return [leaf] so the result still gets backpropagated.
- **SIMULATE**: Run one random rollout per child to terminal state. For a terminal child, use get_result() directly without a rollout.
- **BACKPROPAGATE**: Walk back from each simulated child to the root. Increment visits at every node. Increment value when the player who moved INTO that node ended up winning the game.

The final move is the root child with the most visits — this gives the lowest-variance estimate.

**MCTS vs Other Agents**

In 10-game tournaments at 1 second per move, MCTS beats GreedyAgent pretty convincingly. Random rollouts give way more signal than a one-step stone count, especially in early-game positions. Against IterativeDeepeningAgent, MCTS is competitive and tends to pull ahead as time per move increases.

**MCTS Figure (mcts_analysis.png)**

The figure shows four heatmaps of the starting 5x5 board:

- Top row: number of times each position was visited after 0.1s (left) vs 1.0s (right) of search.
- Bottom row: win rate for Black from each position at those same search budgets.

At 0.1s, visits are spread out everywhere — MCTS is exploring broadly. At 1.0s, visits concentrate on center and near-center positions. The win-rate maps confirm that center and adjacent positions give Black the best winning chances, which aligns with real Go strategy. The exploration constant c = sqrt(2) was used, which balances trying known-good moves versus discovering new ones.

---

## Part 3 — Final Agent

### Approach

FinalAgent is an enhanced MCTS agent that combines several improvements over the base MCTSAgent:

**Value Network Evaluation**

The biggest improvement is replacing random rollouts with the trained ValueNetwork from Part 1. Instead of playing a full random game to estimate who wins, FinalAgent feeds the leaf state through the neural network and gets an immediate value estimate. This is much faster per evaluation and more accurate than noisy random play, allowing many more MCTS iterations within the same time budget.

When value_model.pt is available, `_evaluate_leaf` calls the learned heuristic. If not (e.g., the file is missing), it falls back to random rollouts so the agent always works.

**Tree Reuse**

FinalAgent saves the MCTS tree between moves. After making a move and seeing the opponent's response, it finds the subtree rooted at the new board position instead of starting from scratch. This effectively gives the agent "free" thinking time — it builds on work done in previous turns.

**Opening Book**

On the very first move of a 5x5 game (0 stones on board), FinalAgent always plays the center (action 12, position 2,2). This is known to be the strongest opening and skips all search time.

**Smarter Rollouts (fallback)**

When the value network isn't used, rollouts prefer non-pass moves with 80% probability. This makes rollouts terminate faster and produce more meaningful results than pure uniform random.

**Continuous Backpropagation**

Backpropagation uses the actual float value from the network (or rollout) rather than binary win/loss counts. This means a position the network rates as "slightly winning" (+0.4) still contributes useful signal, whereas a binary scheme would discard it.

**Terminal Node Handling**

A terminal leaf node that gets selected by UCT is immediately backpropagated with its true game result and given a visit count of 1, so its UCT score becomes finite. Without this, a terminal leaf has UCT = ∞ forever and gets selected in an infinite loop, wasting all search time.

### Experiments

Tested FinalAgent against:
- RandomAgent: consistent wins
- GreedyAgent: consistent wins  
- Staff AlphaBetaAgent: roughly even (value network helps here)
- Staff MCTSAgent: competitive but challenging; staff MCTS may have additional optimizations

The value network substitution for rollouts was the single largest improvement. Switching from random rollouts to network evaluation roughly tripled the effective number of MCTS iterations within the same time budget.

### Known Limitations

Against very strong agents (SuperAgent1, SuperAgent2), FinalAgent struggles. These likely use significantly more sophisticated search or better trained models. Given more time, improvements would include: training on self-play data, implementing a policy network to focus MCTS search on promising moves, or using transposition tables.

---

Hours taken: 24

Collaborators: None

Known Bugs: None

AI Use Description: None
