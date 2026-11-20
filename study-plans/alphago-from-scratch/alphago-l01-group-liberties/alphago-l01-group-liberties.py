import numpy as np

def go_group_liberties(board: list, row: int, col: int) -> tuple:
    """
    Returns: a tuple of two sorted coordinate lists: group and liberties.
    """
    total_rows = len(board)
    total_cols = len(board[0])
    stone_colour = int(board[row][col])

    group = set()
    liberties = set()
    stack = [(row, col)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        current_row, current_col = stack.pop()

        if (current_row, current_col) in group:
            continue

        group.add((current_row, current_col))

        for row_step, col_step in directions:
            next_row = current_row + row_step
            next_col = current_col + col_step

            if next_row < 0 or next_row >= total_rows:
                continue
            if next_col < 0 or next_col >= total_cols:
                continue

            neighbour = int(board[next_row][next_col])

            if neighbour == stone_colour:
                if (next_row, next_col) not in group:
                    stack.append((next_row, next_col))
            elif neighbour == 0:
                liberties.add((next_row, next_col))

    return sorted(group), sorted(liberties)