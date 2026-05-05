"""
Generate a figure illustrating how the MCTS agent makes decisions.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import random

from go_search_problem import GoProblem, GoState
from agents import MCTSAgent, MCTSNode


def run_mcts_and_get_visits(state: GoState, search_time: float, c: float = np.sqrt(2)):
    """Run MCTS and return visit counts and win rates for root children."""
    
    # Create a fresh MCTS agent
    agent = MCTSAgent(c=c)
    
    # Manually run the MCTS loop (same as agent.get_move but we keep the tree)
    root = MCTSNode(state.clone())
    deadline = time.time() + search_time
    
    iterations = 0
    while time.time() < deadline and iterations < 1000:
        # Selection
        leaf = agent.select(root)
        
        # Expansion
        children = agent.expand(leaf)
        
        # Simulation
        if children:
            results = agent.simulate(children)
            # Backpropagation
            agent.backpropagate(results, children)
        
        iterations += 1
    
    # Extract visit counts and win rates from root children
    board_size = state.size
    visits = np.zeros((board_size, board_size))
    win_rates = np.zeros((board_size, board_size))
    
    pass_action = board_size * board_size
    for child in root.children:
        if child.action != pass_action and child.visits > 0:
            row = child.action // board_size
            col = child.action % board_size
            visits[row, col] = child.visits
            win_rates[row, col] = child.value / child.visits
    
    return visits, win_rates, iterations


def main():
    board_size = 5
    game = GoProblem(size=board_size)
    state = game.start_state
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    times = [0.1, 1.0]
    c_value = 1.414  # sqrt(2)
    
    # Collect data
    visit_maps = []
    win_maps = []
    
    for t in times:
        print(f"Running MCTS for {t}s...")
        visits, wins, iters = run_mcts_and_get_visits(state, t, c_value)
        visit_maps.append(visits)
        win_maps.append(wins)
        print(f"  Completed {iters} iterations")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("MCTS Decision Analysis on 5x5 Go (Initial Board State)",
                 fontsize=14, fontweight="bold")
    
    for col, (t, visits, wins) in enumerate(zip(times, visit_maps, win_maps)):
        
        # Visit count heatmap (top row)
        ax = axes[0, col]
        im = ax.imshow(visits, cmap="YlOrRd", aspect="equal")
        ax.set_title(f"Visit counts after {t}s", fontsize=12)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        
        # Add text labels
        for r in range(board_size):
            for c in range(board_size):
                v = int(visits[r, c])
                if v > 0:
                    text_color = "white" if v > visits.max() * 0.6 else "black"
                    ax.text(c, r, str(v), ha="center", va="center", fontsize=9, color=text_color)
        
        plt.colorbar(im, ax=ax, label="Number of visits")
        
        # Win rate heatmap (bottom row)
        ax2 = axes[1, col]
        im2 = ax2.imshow(wins, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
        ax2.set_title(f"Win rate after {t}s (for Black)", fontsize=12)
        ax2.set_xlabel("Column")
        ax2.set_ylabel("Row")
        
        # Add text labels
        for r in range(board_size):
            for c in range(board_size):
                w = wins[r, c]
                if w > 0:
                    ax2.text(c, r, f"{w:.2f}", ha="center", va="center", fontsize=8)
        
        plt.colorbar(im2, ax=ax2, label="Win rate")
    
    plt.tight_layout()
    plt.savefig("mcts_analysis.png", dpi=150, bbox_inches="tight")
    print("Figure saved to mcts_analysis.png")
    
    # Print summary statistics
    print("\nSummary:")
    for t, visits in zip(times, visit_maps):
        total_visits = np.sum(visits)
        nonzero = np.count_nonzero(visits)
        print(f"  {t}s: {total_visits} total visits across {nonzero} positions")


if __name__ == "__main__":
    main()