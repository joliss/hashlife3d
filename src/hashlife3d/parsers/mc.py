"""
Parser for the Macrocell (.mc) format.
"""

import re

import numpy as np
from ranges import Range

from ..grid import Grid, empty_quadtree
from ..state import State
from ..node import Quadtree, QuadtreeBranch
from ..extent import RectangleExtent


def _parse_leaf_pattern(pattern):
    grid = np.empty((8, 8), dtype=object).view(Grid)
    grid.fill(State.DEAD)
    x = y = 0
    for c in pattern:
        if c == '.':
            grid[y, x] = State.DEAD
        elif c == '*':
            grid[y, x] = State.ALIVE
        elif c == '$':
            y += 1
            x = 0
        else:
            raise ValueError(f'Invalid character: {c}')
    assert y == 8 and x == 0
    return Quadtree.from_grid(grid)


def _parse_branch_pattern(pattern, quadtrees):
    side_length_shift, nw, ne, sw, se = (int(s) for s in pattern.split())
    assert side_length_shift >= 4
    side_length = 2 ** int(side_length_shift)
    def get_quadtree(i):
        if i == 0:
            return empty_quadtree(
                RectangleExtent(Range(0, side_length // 2), Range(0, side_length // 2))
            )
        else:
            return quadtrees[i]
    quadtree = QuadtreeBranch(np.array([
        [get_quadtree(nw), get_quadtree(ne)],
        [get_quadtree(sw), get_quadtree(se)]
    ], dtype=object))
    assert quadtree.width() == side_length
    return quadtree


def parse(s):
    empty_grid = np.empty((8, 8), dtype=object)
    empty_grid.fill(State.DEAD)
    quadtrees = [
        Quadtree.from_grid(empty_grid.view(Grid))
    ]
    lines = [line for line in s.splitlines()]
    assert lines.pop(0).startswith('[M2]')
    for line in lines:
        if line.startswith('#R'):
            assert line == '#R B3/S23'
    i = 0
    for line in lines:
        if re.match(r'^\s*(#.*)?$', line):
            continue
        line = line.strip()
        if line[0] in ('.', '*', '$'):
            quadtrees.append(_parse_leaf_pattern(line))
        else:
            quadtrees.append(_parse_branch_pattern(line, quadtrees))
    return quadtrees[-1]


def parse_header(line):
    header = {}
    for field in line.split(','):
        field = field.strip()
        (key, value) = field.split('=')
        key = key.strip()
        value = value.strip()
        if key == 'x' or key == 'y':
            value = int(value)
        header[key] = value
    return header
