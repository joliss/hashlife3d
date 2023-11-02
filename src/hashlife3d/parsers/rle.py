import re

import numpy as np

from ..grid import Grid
from ..state import State


def parse(s):
    lines = [line for line in s.splitlines() if not re.match(r'^\s*(#.*)?$', line)]
    assert lines[0].startswith('x')
    header = parse_header(lines[0])
    assert header['x'] > 0
    assert header['y'] > 0
    if 'rule' in header:
        assert header['rule'] == 'B3/S23'
    width = header['x']
    height = header['y']

    pattern = re.sub(r'\s', '', ''.join(lines[1:]))
    pattern = re.sub(r'!.*', '!', pattern, flags=re.S)
    assert pattern.endswith('!')
    pattern = pattern[:-1]

    grid = Grid.dead(width, height)

    item_re = re.compile(r'(\d*)([bo$A-X]|[p-y][A-X])')
    x = 0
    y = 0
    for match in item_re.finditer(pattern):
        (count, tag) = match.groups()
        count = int(count or 1)
        if tag == 'b' or tag == '.':
            grid[y, x:x+count] = State.DEAD
            x += count
        elif tag == 'o' or tag == 'A':
            grid[y, x:x+count] = State.ALIVE
            x += count
        elif tag == '$':
            y += count
            x = 0
        else:
            raise ValueError(f'Invalid tag: {tag}')
        assert x <= width
        assert y < height or (y == height and x == 0)

    return grid


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
