from hashlife3d.parsers.rle import parse

r_pentomino = """
x = 3, y = 3, rule = B3/S23
b2o$2o$bo!
"""


def test_rle():
    grid = parse(r_pentomino)
    assert str(grid) == """
_XX
XX_
_X_
""".strip()
