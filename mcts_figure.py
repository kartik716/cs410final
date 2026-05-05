"""
Generate a figure illustrating how the MCTS agent makes decisions.

The figure shows two side-by-side heatmaps of root-child visit counts on a
5×5 board, one after 0.1 s of search and another after 1.0 s of search.
Darker cells = more simulations through that move.  The heatmaps let us see
which regions of the board MCTS considers most promising and how much the
distribution sharpens with additional search time.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from go_search_problem import GoProblem, GoState
from agents import MCTSAgent, MCTSNode


def build_visit_heatmap(game_state: GoState, search_time: float,
                         board_size: int = 5, c: float = np.sqrt(2)) -> np.ndarray:
    """
    Run MCTS for *search_time* seconds and return a (board_size, board_size)
    array of visit counts for each root child.  Pass-move visits are ignored.
    """
    agent = MCTSAgent(c=c)
    # Build the tree manually so we can inspect root.children afterwards.
    import time
    root = MCTSNode(game_state.clone())
    deadline = time.time() + search_time

    while time.time() < deadline:
        leaf = agent.select(root)
        children = agent.expand(leaf)
        results = agent.simulate(children)
        agent.backpropagate(results, children[:len(results)])

    heatmap = np.zeros((board_size, board_size))
    pass_action = board_size * board_size
    for child in root.children:
        if child.action != pass_action:
            row = child.action // board_size
            col = child.action % board_size
            heatmap[row, col] = child.visits
    return heatmap


def build_visit_heatmap_wins(game_state: GoState, search_time: float,
                              board_size: int = 5, c: float = np.sqrt(2)) -> np.ndarray:
    """Return win-rate heatmap (wins / visits) for root children."""
    import time
    agent = MCTSAgent(c=c)
    root = MCTSNode(game_state.clone())
    deadline = time.time() + search_time

    while time.time() < deadline:
        leaf = agent.select(root)
        children = agent.expand(leaf)
        results = agent.simulate(children)
        agent.backpropagate(results, children[:len(results)])

    heatmap = np.full((board_size, board_size), np.nan)
    pass_action = board_size * board_size
    for child in root.children:
        if child.action != pass_action and child.visits > 0:
            row = child.action // board_size
            col = child.action % board_size
            heatmap[row, col] = child.value / child.visits
    return heatmap


def main():
    board_size = 5
    game = GoProblem(size=board_size)
    state = game.start_state

    times = [0.1, 1.0]
    heatmaps = [build_visit_heatmap(state, t, board_size) for t in times]
    win_heatmaps = [build_visit_heatmap_wins(state, t, board_size) for t in times]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle("MCTS Decision Analysis on 5×5 Go (Initial Board State)",
                 fontsize=14, fontweight="bold")

    titles_visits = [f"Visit counts after {t}s" for t in times]
    titles_wins   = [f"Win-rate after {t}s" for t in times]

    for col, (visits, wins, tv, tw) in enumerate(
            zip(heatmaps, win_heatmaps, titles_visits, titles_wins)):

        # Visit-count heatmap
        ax = axes[0][col]
        im = ax.imshow(visits, cmap="YlOrRd", vmin=0, aspect="equal")
        ax.set_title(tv, fontsize=12)
        ax.set_xticks(range(board_size))
        ax.set_yticks(range(board_size))
        ax.set_xticklabels(range(board_size))
        ax.set_yticklabels(range(board_size))
        # Annotate with visit counts
        for r in range(board_size):
            for c_ in range(board_size):
                v = int(visits[r, c_])
                ax.text(c_, r, str(v), ha="center", va="center",
                        fontsize=8, color="black" if v < visits.max() * 0.6 else "white")
        plt.colorbar(im, ax=ax, label="Visits")

        # Win-rate heatmap
        ax2 = axes[1][col]
        masked = np.ma.array(wins, mask=np.isnan(wins))
        im2 = ax2.imshow(masked, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
        ax2.set_title(tw, fontsize=12)
        ax2.set_xticks(range(board_size))
        ax2.set_yticks(range(board_size))
        ax2.set_xticklabels(range(board_size))
        ax2.set_yticklabels(range(board_size))
        for r in range(board_size):
            for c_ in range(board_size):
                if not np.isnan(wins[r, c_]):
                    ax2.text(c_, r, f"{wins[r,c_]:.2f}", ha="center", va="center",
                             fontsize=7)
        plt.colorbar(im2, ax=ax2, label="Win-rate (for Black)")

    plt.tight_layout()
    out = "mcts_analysis.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()
