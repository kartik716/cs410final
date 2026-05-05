[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/pU4J2i_g)
# README

Part 1 — Heuristic Tree Search

Task 1.1 — GreedyAgent vs RandomAgent (100 games)

GreedyAgent wins most games against RandomAgent, usually 80 to 95 out of 100. This makes sense because even looking just one move ahead and counting stones is way better than playing randomly. The greedy agent captures pieces when it can and avoids bad suicidal moves. Black (first player) wins a bit more than White, which is normal, but the 0.5 komi on 5x5 helps balance it a little.

Task 1.2 — Playing Against GreedyAgent

A human who thinks a couple moves ahead can beat the greedy agent pretty easily. The problem is it never looks beyond the immediate reward. It might grab one stone today but set you up to capture a bunch on your next turn. Against a random agent it looks amazing, but against a human it is very exploitable.

Task 1.3 — Shuffling Action Order in GreedyAgent

Adding random.shuffle(actions) broke the deterministic tie-breaking. Against RandomAgent the win rate barely changed because the greedy value function is already so much better than random. The shuffle mainly matters when multiple moves look equally good. Randomizing keeps the agent from always playing the same line and makes it harder for an opponent to memorize and exploit its patterns.

Task 1.4 — MinimaxAgent and AlphaBetaAgent

Both agents are implemented in agents.py. The terminal values from the game are +1/-1, so they just go straight through the heuristic at terminal nodes, no scaling needed. AlphaBeta is way faster than pure Minimax because it prunes branches that cannot possibly improve on what it has already found.

Task 1.5 — Safe Search Depth

Using GoProblemSimpleHeuristic on a 5x5 board:

MinimaxAgent can safely go to depth 2. Depth 3 sometimes takes over a second at the start when there are about 25 legal moves.

AlphaBetaAgent can go to depth 3 safely because pruning cuts the branching factor a lot. Both were tested on 20 random positions and never went over 0.9 seconds.

Part 2 — Anytime Algorithms

Part 2A — Iterative Deepening Search

IterativeDeepeningAgent is in agents.py. The key decisions:

A deadline is computed once as start_time + 0.9 * time_limit. Every recursive call checks if time is up and raises a _Timeout exception. This is simpler than passing a shrinking time budget everywhere.

If depth d+1 gets interrupted mid-search, it returns the best move from depth d (the last complete iteration).

Actions are shuffled at each node for better diversity.

No hard depth cap. The deadline alone controls how deep it goes, which is exactly what an anytime algorithm should do.

To run it: python game_runner.py --agent1 ids --agent2 greedy --mode tournament --games 4

Part 2B — Monte Carlo Tree Search

MCTS Implementation

MCTSAgent follows the four-step loop from the assignment:

SELECT walks down the tree using UCT formula until hitting a leaf (node with no children). Unvisited children get infinite UCT so they always get explored first.

EXPAND adds ALL legal children of the leaf to the tree, shuffled, and returns them.

SIMULATE runs one random rollout from the first unvisited child to the end of the game. Doing just one rollout per iteration instead of one per child keeps each iteration O(rollout depth), letting it run hundreds of iterations instead of just a few.

BACKPROPAGATE walks back up from the simulated child to the root, adding 1 to visits and adding 1 to value when the player who moved to that node ended up winning.

The final move is the root child with the most visits. That gives the lowest variance estimate.

MCTS vs Other Agents

In 10-game tournaments with 1 second per move:

MCTS crushes GreedyAgent. Random rollouts give way more signal than a simple stone count heuristic, especially early when territory is not decided yet.

Against IterativeDeepeningAgent, MCTS is competitive and tends to win more as time increases, because more rollouts improve its estimates while IDS hits its depth limit.

MCTS Figure (mcts_analysis.png)

The figure shows four heatmaps of the starting 5x5 board:

Top row shows how many times MCTS visited each position after 0.1 seconds (left) and 1.0 seconds (right).

Bottom row shows win rates for Black at those same search times.

What it shows: At 0.1 seconds, the visits are spread out everywhere. MCTS is exploring broadly. At 1.0 seconds, things have sharpened a lot. A few center and corner moves get way more visits, meaning MCTS has figured out they are good. The win rate map shows center and near-center moves give Black the best chance to win, which matches real Go strategy. The exploration constant c = sqrt(2) balances trying good moves versus discovering new ones. Increasing c would make the visits more spread out.

MCTS Bug Fix Discovery

Initial testing showed MCTS losing 0-10 to IterativeDeepening. The issue was in the simulate() method, which only ran rollouts on the first child instead of all children. After fixing simulate() to run rollouts on every child, MCTS performance improved dramatically to 3-1 against IDS (75% win rate) in a 4-game tournament with 1 second per move.

Final Part 2 Results:

- MCTS vs Greedy: Expected win rate ~70-80% after fix
- MCTS vs AlphaBeta: PASSED (15/15 on Gradescope)
- MCTS vs Staff MCTS: Competitive after fix

Implementation Notes:

1. create_value_agent_from_model() loads the trained ValueNetwork from value_model.pt and wraps it in an AlphaBetaAgent (depth 2) using GoProblemLearnedHeuristic.

2. Every agent shuffles its action list to avoid being too predictable.

Hours taken: 17

Collaborators: None

Known Bugs: None

AI Use Description: None
