from hashlife3d.camera import *


def test_find_required_quadtree():
    assert find_required_quadtree(0, 8, 0, 8, 1) == (Extent(-4, -4, 16), Extent(0, 0, 8))
    assert find_required_quadtree(100, 120, -400, -390, 1) == (Extent(44, -468, 128), Extent(76, -436, 64))
