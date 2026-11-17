"""
CMSC474 Project 2 - Reversi (88-cell variant)

HOW THE PROGRAM IS RUN
    The game server runs this file once per turn. It sends the current board
    on stdin (88 numbers) and expects one line on stdout: "row col" for the
    cell we want to play. We have 3 seconds per turn.

OVERALL PLAN
    1. Read the board from stdin.                       -> read_input
    2. Find every legal move we can make.               -> find_valid_moves
    3. Look ahead with minimax + alpha-beta pruning,
       going deeper and deeper until time is almost up. -> iterative_deepening
    4. Print the best move we found.                    -> format_move / main

HOW THE BOARD IS STORED
    The board is a plain Python list of 88 ints, one per cell, in the same
    order the cells appear in the input (row 1 left-to-right, then row 2, ...).
        board[i] == EMPTY (0), ME (1) or OPP (2)
    A "move" is just the index i of the cell we place our disk in.
    CELLS[i] tells you which (row, col) that index is.

To Do
- [ ] Board geometry: confirm how rows line up (Figure 1 of the spec)
- [ ] Rules: flips_for, find_valid_moves, apply_move, game_over
- [ ] Utility + evaluation function (corners, mobility, discs, ...)
- [ ] Alpha-beta search with move ordering
- [ ] Iterative deepening under the 3 s time limit
"""

import sys
import time

# ===========================================================================
# CONSTANTS
# ===========================================================================

# What each number in the input means.
EMPTY, ME, OPP = 0, 1, 2

# How many cells are in each of the 8 rows (top to bottom), from the spec.
ROW_LENGTHS = [8, 10, 12, 14, 14, 12, 10, 8]

# The input lines every row up on the LEFT, but on the real board the shorter
# rows are pushed in toward the middle. ROW_OFFSETS says how far each row is
# shifted right so that columns line up the way they do on the real board:
#     ROW_OFFSETS = [3, 2, 1, 0, 0, 1, 2, 3]
# So input cell (row 0, col 0) actually sits above input cell (row 1, col 1).
# !! This assumes the board is centered. Check it against Figure 1 of the spec.
ROW_OFFSETS = [(max(ROW_LENGTHS) - n) // 2 for n in ROW_LENGTHS]

# Every cell as (row, col), 0-indexed, in the same order as the input.
#     CELLS[0]  == (0, 0)   first cell of row 1
#     CELLS[8]  == (1, 0)   first cell of row 2
#     CELLS[87] == (7, 7)   last cell of row 8
# To go from (row, col) back to an index: CELLS.index((row, col)) (slow) or
# cell_index(row, col) (fast, once you write it).
CELLS = [(r, c) for r, n in enumerate(ROW_LENGTHS) for c in range(n)]
NUM_CELLS = len(CELLS)  # 88

# The 8 directions a line of flipped disks can run in, as (row step, col step).
# These are in ALIGNED coordinates (after applying ROW_OFFSETS), because that's
# where "straight up" and "diagonal" make sense geometrically.
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),   # up-left,   up,    up-right
              (0, -1),           (0, 1),    # left,             right
              (1, -1),  (1, 0),  (1, 1)]    # down-left, down,  down-right

TIME_LIMIT = 3.0       # seconds per turn, from the spec
SAFETY_MARGIN = 0.4    # stop searching this many seconds early (startup + printing)
WIN_SCORE = 10_000     # score for a won game. Must be bigger than anything
                       # evaluate() can return, so a real win always beats a
                       # position that only "looks good".

class SearchTimeout(Exception):
    """Thrown from deep inside the search when we run out of time.

    The search is recursive, so the easiest way to stop immediately is to raise
    this exception and catch it in iterative_deepening(), which then falls back
    to the best move from the last depth that finished completely.
    """


# ===========================================================================
# BOARD + INPUT/OUTPUT
# ===========================================================================

def read_input(text):
    """Turn the raw input text into a board list.

    Input:  text - the whole stdin as one string, e.g.
                "0 0 0 0 0 0 0 0\\n0 0 0 0 0 0 0 0 0 0\\n ..."
            Rows have different lengths and spacing may vary, so don't rely on
            line structure. Just grab all the numbers in order.
    Output: a list of 88 ints (EMPTY / ME / OPP), in input order.

    Example: the first 8 numbers become board[0]..board[7] (row 1),
             the next 10 become board[8]..board[17] (row 2), and so on.
    """
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
    """Find the board index for an input-style (row, col) position.

    Input:  row, col - 0-indexed input coordinates (row 0..7, col 0..len-1).
    Output: the index i such that CELLS[i] == (row, col),
            or None if that (row, col) isn't on the board
            (row out of 0..7, or col past the end of that row).

    Example: cell_index(1, 0) == 8   (row 1 starts after the 8 cells of row 0)
    Tip: add up ROW_LENGTHS of the rows above, then add col.
    """
    if row < 0 or row >= len(ROW_LENGTHS):
        return None
    if col < 0 or col >= ROW_LENGTHS[row]:
        return None
    return ROW_STARTS[row] + col


def to_aligned(row, col):
    """Convert input coordinates to "real board" (aligned) coordinates.

    Input:  row, col - 0-indexed input coordinates.
    Output: (row, aligned_col) where aligned_col = col + ROW_OFFSETS[row].

    Why: in input coordinates, the cell "directly above" isn't at the same col
    because rows are shifted. In aligned coordinates it is, so the DIRECTIONS
    work normally.
    Example: to_aligned(0, 0) == (0, 3)
    """
    return row, col + ROW_OFFSETS[row]


def from_aligned(row, aligned_col):
    """Undo to_aligned: go from aligned coordinates back to a board index.

    Input:  row, aligned_col - a position on the real board grid.
    Output: the board index of that cell, or None if there's no cell there
            (off the top/bottom, or in the cut-off corners of a short row).

    Example: from_aligned(0, 3) == 0  (first cell of row 0)
             from_aligned(0, 0) is None  (row 0 doesn't reach that far left)
    """
    if row < 0 or row >= len(ROW_LENGTHS):
        return None
    col = aligned_col - ROW_OFFSETS[row]
    return cell_index(row, col)


def build_neighbor_table():
    """Precompute "what's the next cell in direction d from cell i?".

    Output: a list with one entry per cell. Entry i is a list of 8 values,
            one per direction in DIRECTIONS. Each value is the index of the
            neighboring cell in that direction, or None if it's off the board.

        NEIGHBORS[i][d]  ->  index of the cell next to i in direction d

    Why: flips_for() walks along lines constantly during search. Doing the
    coordinate math once at startup and then just looking it up is much faster.

    How: for each cell, convert it to aligned coords, add the direction step,
    and convert back with from_aligned().
    """

    all_neighbors = []

    for i in range(NUM_CELLS):
        neighbors = []
        for dr, dc in DIRECTIONS:
            r, c = to_aligned(*CELLS[i])
            index = from_aligned(r + dr, c + dc)
            neighbors.append(index)

        all_neighbors.append(neighbors)

    return all_neighbors


# Filled in once at startup. Uncomment when build_neighbor_table() works.
NEIGHBORS = build_neighbor_table()


def format_move(i):
    """Turn a move (board index) into the text the server expects.

    Input:  i - a board index.
    Output: a string "row col", 1-indexed, in INPUT coordinates
            (not aligned coordinates).

    Example: if CELLS[i] == (6, 5), return "7 6"
             (this is the sample answer from the spec).
    """
    row, column = CELLS[i]
    return f"{row + 1} {column + 1}"


def other(player):
    """Return the other player: other(ME) == OPP, other(OPP) == ME."""
    return OPP if player == ME else ME


# ===========================================================================
# GAME RULES
# ===========================================================================

def flips_for(board, i, player):
    """Work out which disks would flip if `player` placed a disk at cell i.

    Input:  board  - current board list
            i      - the cell we're thinking about playing in
            player - ME or OPP (who is moving)
    Output: a list of board indices that would flip to `player`'s color.
            An EMPTY list means the move is illegal.

    The rule: from cell i, look in each of the 8 directions. If you see one or
    more of the opponent's disks in a row, followed by one of `player`'s own
    disks, then all of those opponent disks in between flip. If the line hits
    an empty cell or the edge of the board first, nothing flips that way.

    Examples along one direction (P = player, O = opponent, . = empty):
        i O O P   -> flips both O's
        i O O .   -> flips nothing (no P to close the line)
        i P       -> flips nothing (need at least one O in between)

    Also: if cell i isn't EMPTY, the move is illegal, so return [].
    """
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
    """List every cell where `player` is allowed to play.

    Input:  board, player (ME by default)
    Output: a list of board indices. Empty list = this player has to pass.

    A cell is valid if it's empty and flips_for() returns at least one disk.
    Example: on the spec's sample board, our valid moves (formatted) are
             3 5, 4 6, 4 9, 5 10, 5 11, 7 6.
    """
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
    """Make a move and return the resulting board.

    Input:  board, i (where to play), player (who's playing)
    Output: a NEW list. Don't change the original board, because the search
            needs it again when it tries the next move.

    The new board has `player`'s disk at i, and every disk from
    flips_for(board, i, player) changed to `player`.
    Example: on the sample board, playing "7 6" flips 4 opponent disks, so we
             gain 5 disks (4 flipped + 1 placed) and they lose 4.
    """
    new_board = board[:] # copy old board
    new_board[i] = player # flip choosen cell to player

    # find all cells that flipped when choosing this cell
    flips = flips_for(board, i, player)

    # set all flipped cells to player
    for j in flips:
        new_board[j] = player

    return new_board


def game_over(board):
    """Check whether the game has ended.

    Output: True if NEITHER player has a valid move (board full, or nobody can
            flip anything). False otherwise.

    Note: one player having no moves is NOT game over. They just pass.
    """
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


def utility(board):
    """Exact score of a FINISHED game. Only call this when game_over() is True.

    Output: +WIN_SCORE if we have more disks, -WIN_SCORE if they do, 0 for a tie.
    Optional: add the disk difference (e.g. WIN_SCORE + diff) so the agent
    prefers winning by more.
    """
    raise NotImplementedError


def evaluate(board):
    """Estimate how good an UNFINISHED board is for us.

    This is used when the search hits its depth limit and can't see the end of
    the game. It's the most important function for playing well.

    Output: a number (any scale), bigger = better for ME. Must stay well below
            WIN_SCORE in size.

    How: combine the feature functions below with weights, e.g.
        score = w1 * corner_score + w2 * mobility + w3 * disc_difference + ...
    Good idea: change the weights with game_phase(). Early on, mobility and
    corners matter most. Near the end, disk count matters most.
    """
    raise NotImplementedError


def disc_difference(board):
    """How many more disks we have than the opponent.

    Output: (number of ME cells) - (number of OPP cells). Can be negative.
    Note: this matters most near the end. Early on, having FEWER disks is often
    better because it gives the opponent fewer moves.
    """
    player_disc = 0
    opp_disc = 0
    for i in range(NUM_CELLS):
        if board[i] == ME:
            player_disc+=1

        if board[i] == OPP:
            opp_disc+=1

    return player_disc - opp_disc


def mobility(board):
    """How many more moves we have than the opponent.

    Output: len(our valid moves) - len(their valid moves).
    Why: if the opponent has few options, they're often forced into bad moves.
    Note: this calls find_valid_moves twice, so it's expensive. Keep an eye on
    speed.
    """
    num_player_valid_moves = len(find_valid_moves(board, ME))
    num_opp_valid_moves = len(find_valid_moves(board, OPP))

    return num_player_valid_moves - num_opp_valid_moves


def corner_score(board):
    """How many more corners we hold than the opponent.

    Output: (corners that are ME) - (corners that are OPP).
    Corners are very valuable because they can never be flipped back.
    """
    player_corner = 0
    opp_corner = 0

    for i in CORNER_CELLS:
        if board[i] == ME:
            player_corner+=1

        if board[i] == OPP:
            opp_corner+=1

    return player_corner - opp_corner


def danger_squares_score(board):
    """Penalty for disks next to an EMPTY corner.

    If we put a disk right next to an empty corner, we often give the opponent
    a way to take that corner. Once the corner is taken, the cells next to it
    are no longer dangerous.

    Output: (our disks next to empty corners) - (their disks next to empty
            corners). Use it with a NEGATIVE weight in evaluate().
    """
    raise NotImplementedError


def game_phase(board):
    """How far into the game we are, from 0.0 to 1.0.

    Output: 0.0 at the start (almost all cells empty), 1.0 at the end
            (no empty cells). For example: 1 - (empty cells / NUM_CELLS).
    Used by evaluate() to change weights as the game goes on.
    """
    raise NotImplementedError


# ===========================================================================
# SEARCH (minimax with alpha-beta pruning)
# ===========================================================================

def rank_moves(board, moves, player):
    """Sort moves so the most promising ones are tried first.

    Input:  board, the list of valid moves, and whose turn it is.
    Output: the same moves, reordered best-first.

    Why: alpha-beta prunes much more when it sees good moves early, which lets
    the search go deeper in the same time. The ordering doesn't change the
    answer, only the speed.
    Cheap ideas: corners first, cells next to empty corners last, or sort by
    a quick evaluate() of the board after the move.
    """
    raise NotImplementedError


def alphabeta(board, depth, alpha, beta, player, deadline):
    """Score `board` by looking `depth` moves ahead.

    Input:
        board    - position to score
        depth    - how many more moves to look ahead (stop at 0)
        alpha    - best score ME is already guaranteed elsewhere in the tree
        beta     - best (lowest) score OPP is already guaranteed elsewhere
        player   - whose turn it is on this board (ME maximizes, OPP minimizes)
        deadline - time.time() value we must stop by
    Output: a number, the minimax value of this board for ME.

    Steps:
      1. If time.time() > deadline: raise SearchTimeout.
      2. If game_over(board): return utility(board).
      3. If depth == 0: return evaluate(board).
      4. Get valid moves for `player`. If there are none, the player passes:
         return alphabeta(same board, depth - 1, alpha, beta, other(player), ...).
      5. If player is ME: try each move, keep the MAX score, update alpha.
         If player is OPP: try each move, keep the MIN score, update beta.
      6. If alpha >= beta at any point, stop looping (prune). The other player
         would never let the game reach this position.
    """
    raise NotImplementedError


def search_root(board, depth, deadline):
    """Find our best move by searching to a fixed depth.

    Input:  board (our turn), depth, deadline
    Output: (best_move, best_score)

    How: for each of our valid moves (in rank_moves order), apply it and call
    alphabeta() on the result with player=OPP and depth-1. Keep the move with
    the highest score. Pass alpha along between moves so later moves get
    pruned too.
    """
    raise NotImplementedError


def iterative_deepening(board, deadline):
    """Search deeper and deeper until time runs out, then return the best move.

    Input:  board, deadline
    Output: a board index (the move to play)

    How:
        best = some valid move (so we ALWAYS have an answer)
        for depth = 1, 2, 3, ...:
            try:    best, _ = search_root(board, depth, deadline)
            except SearchTimeout: stop. Keep `best` from the last FINISHED depth.
    Why: we don't know ahead of time how deep we can get in 3 seconds. This
    way we always have a complete answer ready. A half-finished deeper search
    is thrown away because its result can't be trusted.
    Extra: if the depth gets past the number of empty cells, the search
    already reaches the end of the game, so stop early.
    """
    raise NotImplementedError


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    """Run one turn: read the board, choose a move, print it."""
    board = read_input(sys.stdin.read())
    valid_moves = find_valid_moves(board, ME)

    if not valid_moves:
        return

    if len(valid_moves) == 1:
        print(format_move(valid_moves[0]))
        return

    move = valid_moves[0]
    try:
        # Passing None for deadline until timer logic is implemented
        best = iterative_deepening(board, None)
        if best in valid_moves:
            move = best
    except Exception:
        pass

    print(format_move(move))


if __name__ == "__main__":
    main()
