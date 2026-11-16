import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import reversi  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "reversi.py")

SAMPLE = """\
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 2 1 0 0 0 0 0
0 0 0 0 0 0 2 2 0 0 0 0 0 0
0 0 0 0 0 1 2 2 2 0 0 0 0 0
0 0 0 0 0 2 2 2 2 0 0 0
0 0 0 0 0 0 1 0 0 0
0 0 0 0 0 0 0 0
"""


def moves_as_text(board, player=reversi.ME):
    return sorted(reversi.format_move(m) for m in reversi.find_valid_moves(board, player))


def test_board_has_88_cells():
    assert reversi.NUM_CELLS == 88


def test_sample_legal_moves():
    board = reversi.read_input(SAMPLE)
    assert moves_as_text(board) == sorted(
        ["3 5", "4 6", "4 9", "5 10", "5 11", "7 6"])


def test_sample_answer_flips_four():
    board = reversi.read_input(SAMPLE)
    r, c = 6, 5                      # "7 6" in 0-indexed input coords
    i = reversi.CELLS.index((r, c))
    assert len(reversi.flips_for(board, i, reversi.ME)) == 4


def test_apply_move_flips():
    board = reversi.read_input(SAMPLE)
    i = reversi.CELLS.index((6, 5))
    after = reversi.apply_move(board, i, reversi.ME)
    assert after.count(reversi.OPP) == board.count(reversi.OPP) - 4
    assert after.count(reversi.ME) == board.count(reversi.ME) + 5


def test_program_outputs_legal_move_in_time():
    start = time.time()
    out = subprocess.run([sys.executable, SCRIPT], input=SAMPLE,
                         capture_output=True, text=True, timeout=5).stdout.strip()
    assert time.time() - start < reversi.TIME_LIMIT
    assert out in moves_as_text(reversi.read_input(SAMPLE))
