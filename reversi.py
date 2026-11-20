"""
CMSC474 Project 2 - Reversi (88-cell variant)

Reads the board from stdin and prints our move as "row col".
Uses minimax with alpha-beta pruning and iterative deepening.
"""

import sys
import time

# ===========================================================================
# CONSTANTS
# ===========================================================================

EMPTY, ME, OPP = 0, 1, 2  # cell values in the input

ROW_LENGTHS = [8, 10, 12, 14, 14, 12, 10, 8]  # cells per row

# How far each row is shifted right on the real board (rows are centered)
ROW_OFFSETS = [(max(ROW_LENGTHS) - n) // 2 for n in ROW_LENGTHS]

# (row, col) of every cell, 0-indexed, in input order
CELLS = [(r, c) for r, n in enumerate(ROW_LENGTHS) for c in range(n)]
NUM_CELLS = len(CELLS)  # 88

# 8 directions as (row step, col step) in aligned coordinates
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),   # up-left,   up,    up-right
              (0, -1),           (0, 1),    # left,             right
              (1, -1),  (1, 0),  (1, 1)]    # down-left, down,  down-right

TIME_LIMIT = 3.0       # seconds per turn
SAFETY_MARGIN = 0.6    # stop searching this many seconds before the limit
WIN_SCORE = 10_000     # score for a won game (larger than any evaluate() value)

class SearchTimeout(Exception):
    """Raised by the search when the time limit is reached."""


# ===========================================================================
# BOARD + INPUT/OUTPUT
# ===========================================================================

def read_input(text):
    """Parse the 88 input numbers into a flat board list."""
    if text is not None:
        return [int(x) for x in text.split()[:NUM_CELLS]]
    
    board = []
    for line in sys.stdin:
        for token in line.split():
            board.append(int(token))
            if len(board) == NUM_CELLS:
                return board
    return board

ROW_STARTS = [sum(ROW_LENGTHS[:r]) for r in range(len(ROW_LENGTHS))]

def cell_index(row, col):
    """Convert input (row, col) to a board index, or None if off the board."""
    if row < 0 or row >= len(ROW_LENGTHS):
        return None
    if col < 0 or col >= ROW_LENGTHS[row]:
        return None
    return ROW_STARTS[row] + col


def to_aligned(row, col):
    """Convert input (row, col) to aligned board coordinates."""
    return row, col + ROW_OFFSETS[row]


def from_aligned(row, aligned_col):
    """Convert aligned coordinates to a board index, or None if off the board."""
    if row < 0 or row >= len(ROW_LENGTHS):
        return None
    col = aligned_col - ROW_OFFSETS[row]
    return cell_index(row, col)


def build_neighbor_table():
    """Build NEIGHBORS[i][d]: the index of the next cell from i in direction d."""

    all_neighbors = []

    for i in range(NUM_CELLS):
        neighbors = []
        for dr, dc in DIRECTIONS:
            r, c = to_aligned(*CELLS[i])
            index = from_aligned(r + dr, c + dc)
            neighbors.append(index)

        all_neighbors.append(neighbors)

    return all_neighbors


# Neighbor lookup table, built once at startup
NEIGHBORS = build_neighbor_table()


def format_move(i):
    """Convert a board index to the output string "row col" (1-indexed)."""
    row, column = CELLS[i]
    return f"{row + 1} {column + 1}"


def other(player):
    """Return the other player."""
    return OPP if player == ME else ME


# ===========================================================================
# GAME RULES
# ===========================================================================

def flips_for(board, i, player):
    """Return the cells flipped if player plays at i (empty list = illegal move)."""
    # invalid move: can't play on a taken cell
    if board[i] != EMPTY:
        return []

    opponent = other(player)
    to_be_flipped = []

    for d in range(8): # look in each direction
        maybes = []    # possible cells to flip in current direction
        j = NEIGHBORS[i][d]
        while j is not None and board[j] == opponent:
            maybes.append(j)
            j = NEIGHBORS[j][d]

        if j is not None and board[j] == player and maybes:
            to_be_flipped.extend(maybes)

    return to_be_flipped


def find_valid_moves(board, player=ME):
    """Return all cells where player has a legal move."""
    valid_moves = [] # stores all the valid moves player can choose
    for i in range(NUM_CELLS):
        # if cell is occupied, invalid move
        if board[i] != EMPTY:
            continue

        # stores cells that will be flipped
        cell_flips = flips_for(board, i, player)
        # if no cells are flipped, invalid move
        if not cell_flips:
            continue

        valid_moves.append(i)

    return valid_moves


def apply_move(board, i, player):
    """Return a new board with player's move at i and its flips applied."""
    new_board = board[:] # copy old board
    new_board[i] = player # flip choosen cell to player

    # find all cells that flipped when choosing this cell
    flips = flips_for(board, i, player)

    # set all flipped cells to player
    for j in flips:
        new_board[j] = player

    return new_board


def game_over(board):
    """Return True if neither player has a legal move."""
    player_moves = find_valid_moves(board, ME)
    opp_moves = find_valid_moves(board, OPP)

    return not player_moves and not opp_moves


# ===========================================================================
# SCORING
# All scores are from OUR point of view: bigger = better for ME.
# ===========================================================================


CORNER_CELLS = [cell_index(0, 0),  cell_index(0, 7),
                cell_index(3, 0),  cell_index(3, 13),
                cell_index(4, 0),  cell_index(4, 13),
                cell_index(7, 0),  cell_index(7, 7)]

#percentage of board left opened, used to determine game phase, adjust as needed
ENDGAME_MARK = 0.7

def utility(board):
    """Score a finished game from ME's point of view."""
    player_discs = board.count(ME)
    opp_discs = board.count(OPP)
    disc_diff = player_discs - opp_discs

    if player_discs > opp_discs:
        return disc_diff + WIN_SCORE #we have more discs
    elif player_discs < opp_discs:
        return disc_diff - WIN_SCORE #opp has more discs
    else:
        return 0 #tie

def evaluate(board):
    """Heuristic score of an unfinished board from ME's point of view."""
    #Adjust weights as needed

    # early game weights
    corner_weight = 25
    mobility_weight = 5
    disc_difference_weight = 1
    danger_squares_weight = -10

    # endgame prioritizes disk count
    if game_phase(board) > ENDGAME_MARK:
        disc_difference_weight = 10 # disk count becomes important
        mobility_weight = 2

    # score calculation
    corner = corner_weight * corner_score(board)
    mob = mobility_weight * mobility(board)
    discs = disc_difference_weight * disc_difference(board)
    danger_squares = danger_squares_weight * danger_squares_score(board)
    total_score = corner + mob + discs + danger_squares

    return total_score


def disc_difference(board):
    """Return ME's disk count minus OPP's."""
    return board.count(ME) - board.count(OPP)


def mobility(board):
    """Return ME's number of legal moves minus OPP's."""
    num_player_valid_moves = len(find_valid_moves(board, ME))
    num_opp_valid_moves = len(find_valid_moves(board, OPP))

    return num_player_valid_moves - num_opp_valid_moves


def corner_score(board):
    """Return corners held by ME minus corners held by OPP."""
    player_corner = 0
    opp_corner = 0

    for i in CORNER_CELLS:
        if board[i] == ME:
            player_corner+=1

        if board[i] == OPP:
            opp_corner+=1

    return player_corner - opp_corner


def danger_squares_score(board):
    """Return ME's disks next to empty corners minus OPP's."""
    player_disks_near_corner = 0
    opp_disks_near_corner = 0

    for c in CORNER_CELLS:
        if board[c] != EMPTY:
            continue

        for n in NEIGHBORS[c]:
            if n is None or n in CORNER_CELLS:
                continue

            if board[n] == ME:
                player_disks_near_corner += 1

            if board[n] == OPP:
                opp_disks_near_corner += 1

    return player_disks_near_corner - opp_disks_near_corner


def game_phase(board):
    """Return how full the board is, from 0.0 (start) to 1.0 (end)."""
    return 1 - (board.count(EMPTY) / NUM_CELLS)


# ===========================================================================
# SEARCH (minimax with alpha-beta pruning)
# ===========================================================================

def rank_moves(board, moves, player):
    """Order moves best-first: corners, normal moves, then next to empty corners."""
    corners = []
    corner_neighbors = []
    normal_moves = []
    ranked = []
    for move in moves:
        # check if move is a corner cell
        if move in CORNER_CELLS:
            corners.append(move)
            continue

        # if the move's neighbor is a corner cell, then it is a neighbor of a corner
        neighbors = NEIGHBORS[move]
        near_empty_corner = False
        for n in neighbors:
            if n is not None and n in CORNER_CELLS and board[n] == EMPTY:
                near_empty_corner = True
                break

        if near_empty_corner:
            corner_neighbors.append(move)
        else:
            normal_moves.append(move)

    # rank moves priorty
    # 1. corner cells
    # 2. normal moves
    # 3. corner
    ranked = corners + normal_moves + corner_neighbors
    return ranked

def alphabeta(board, depth, alpha, beta, player, deadline):
    """Minimax value of board with alpha-beta pruning (ME maximizes, OPP minimizes)."""
    if time.time() > deadline:
        raise SearchTimeout
    if game_over(board):
        return utility(board)
    if depth == 0:
        return evaluate(board)
    
    valid_moves = find_valid_moves(board, player)

    # Skip this turn if no valid moves
    if not valid_moves:
        return alphabeta(board, depth - 1, alpha, beta, other(player), deadline)

    ranked_moves = rank_moves(board, valid_moves, player)

    if player == ME:
        max_score = float('-inf')
        for move in ranked_moves:
            score = alphabeta(apply_move(board, move, player), depth - 1, alpha, beta, other(player), deadline)
            if score > max_score:
                max_score = score
            alpha = max(alpha, max_score)
            if alpha >= beta:
                break
        return max_score

    else:  # player == OPP
        min_score = float('inf')
        for move in ranked_moves:
            score = alphabeta(apply_move(board, move, player), depth - 1, alpha, beta, other(player), deadline)
            if score < min_score:
                min_score = score
            beta = min(beta, min_score)
            if beta <= alpha:
                break
        return min_score


def search_root(board, depth, deadline):
    """Return (best_move, score) for ME at a fixed search depth."""
    alpha, beta = float("-inf"), float("inf")
    best_move = None
    best_score = float("-inf")

    # finds all the best possible moves on current board for player
    moves = rank_moves(board, find_valid_moves(board, ME), ME)
    # searches tree for the best move out of all the best possible moves
    for move in moves:
        child = apply_move(board, move, ME)
        score = alphabeta(child, depth - 1, alpha, beta, OPP, deadline)
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, best_score)

    return best_move, best_score


def iterative_deepening(board, deadline):
    """Search deeper each round until time runs out; return the best move found."""
    moves = find_valid_moves(board, ME)
    best_move = rank_moves(board, moves, ME)[0]
    depth = 1
    while depth <= board.count(EMPTY):
        try:
            best_move, _ = search_root(board, depth, deadline)
        except SearchTimeout:
            break
        depth += 1
    return best_move

# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    """Read the board, choose a move within the time limit, and print it."""
    start = time.time()
    board = read_input(sys.stdin.read())
    valid_moves = find_valid_moves(board, ME)

    if not valid_moves:
        return

    if len(valid_moves) == 1:
        print(format_move(valid_moves[0]))
        return

    move = valid_moves[0]
    deadline = start + TIME_LIMIT - SAFETY_MARGIN
    try:
        # Passing None for deadline until timer logic is implemented
        best = iterative_deepening(board, deadline)
        if best in valid_moves:
            move = best
    except Exception:
        pass

    print(format_move(move))


if __name__ == "__main__":
    main()
