"""
CMSC474 Project 2 - Reversi (88-cell variant)

Reads the board from stdin, prints "row col" (1-indexed, input coordinates).
Plan: minimax with alpha-beta pruning + iterative deepening, heuristic eval.

To Do
- [ ] Board geometry: confirm how rows line up (Figure 1 of the spec)
- [ ] Rules: flips_for, find_valid_moves, apply_move, game_over
- [ ] Utility + evaluation function (corners, mobility, discs, ...)
- [ ] Alpha-beta search with move ordering
- [ ] Iterative deepening under the 3 s time limit
"""

import sys
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMPTY, ME, OPP = 0, 1, 2

ROW_LENGTHS = [8, 10, 12, 14, 14, 12, 10, 8]
# Column shift of each row so rows line up geometrically (rows are centered).
# Check against the board picture in the spec before relying on this.
ROW_OFFSETS = [(max(ROW_LENGTHS) - n) // 2 for n in ROW_LENGTHS]

# Every cell as (row, col) in 0-indexed *input* coordinates, in input order.
CELLS = [(r, c) for r, n in enumerate(ROW_LENGTHS) for c in range(n)]
NUM_CELLS = len(CELLS)  # 88

# 8 directions as (d_row, d_col) in aligned/geometric coordinates.
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]

TIME_LIMIT = 3.0       # seconds, from the spec
SAFETY_MARGIN = 0.4    # stop searching this long before the limit
WIN_SCORE = 10_000     # utility of a won game; must beat any heuristic value


class SearchTimeout(Exception):
    """Raised inside the search when the deadline passes."""


# ---------------------------------------------------------------------------
# Board representation / IO
# ---------------------------------------------------------------------------
# A board is a flat list of NUM_CELLS ints (EMPTY / ME / OPP), indexed like CELLS.

def read_input(text):
    """Parse the 88 integers from `text` into a flat board list."""
    raise NotImplementedError


def cell_index(row, col):
    """(row, col) input coords -> index into the board, or None if off-board."""
    raise NotImplementedError


def to_aligned(row, col):
    """Input coords -> aligned (geometric) coords using ROW_OFFSETS."""
    raise NotImplementedError


def from_aligned(row, acol):
    """Aligned coords -> input coords, or None if that spot isn't on the board."""
    raise NotImplementedError


def build_neighbor_table():
    """For each cell and each direction, the index of the next cell (or None).

    Precompute once so the search never redoes coordinate math.
    Returns: list[NUM_CELLS] of list[len(DIRECTIONS)] of int | None
    """
    raise NotImplementedError


NEIGHBORS = None  # = build_neighbor_table() once implemented


def format_move(i):
    """Cell index -> output string "row col" (1-indexed input coords)."""
    raise NotImplementedError


def other(player):
    """ME <-> OPP."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Game rules
# ---------------------------------------------------------------------------

def flips_for(board, i, player):
    """Indices of opponent disks flipped if `player` plays at cell `i`.

    Empty list means the move is illegal.
    """
    raise NotImplementedError


def find_valid_moves(board, player=ME):
    """List of cell indices where `player` has a legal move."""
    raise NotImplementedError


def apply_move(board, i, player):
    """Return a NEW board with `player`'s disk at `i` and flips applied."""
    raise NotImplementedError


def game_over(board):
    """True if neither player has a legal move."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def utility(board):
    """Exact value of a finished game from ME's perspective (+/- WIN_SCORE, 0)."""
    raise NotImplementedError


def evaluate(board):
    """Heuristic value of an unfinished board from ME's perspective.

    Combine features below with weights (possibly depending on game phase).
    """
    raise NotImplementedError


def disc_difference(board):
    """ME disks - OPP disks."""
    raise NotImplementedError


def mobility(board):
    """# ME legal moves - # OPP legal moves."""
    raise NotImplementedError


def corner_cells():
    """Indices of the 'corner' cells for this board shape (can't be flipped)."""
    raise NotImplementedError


def corner_score(board):
    """Corners held by ME - corners held by OPP."""
    raise NotImplementedError


def danger_squares_score(board):
    """Penalty for disks next to empty corners (X/C-square idea)."""
    raise NotImplementedError


def game_phase(board):
    """0.0 (opening) -> 1.0 (endgame), e.g. by number of empty cells."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def rank_moves(board, moves, player):
    """Order moves best-first (corners first, etc.) to improve pruning."""
    raise NotImplementedError


def alphabeta(board, depth, alpha, beta, player, deadline):
    """Minimax value of `board` with `player` to move.

    ME maximizes, OPP minimizes. Handle passes (no legal moves).
    Raise SearchTimeout if time.time() > deadline.
    """
    raise NotImplementedError


def search_root(board, depth, deadline):
    """Best move for ME at a fixed depth. Returns (move, value)."""
    raise NotImplementedError


def iterative_deepening(board, deadline):
    """Search depth 1, 2, 3, ... until time runs out; return last finished best move."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Read stdin, pick a move within the time limit, print it."""
    raise NotImplementedError


if __name__ == "__main__":
  main()