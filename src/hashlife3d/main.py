import sys
import os

import numpy as np
from ranges import Range
from PIL import Image

from .video import create_video
from .grid import LazyGrid
from .parsers.rle import parse as parse_rle
from .extent import Point2D, RectangleExtent, CuboidExtent
from .camera import snapshot_from_grid


def _make_speed_fn(at_0, at_10):
    """
    Return an exponential function f such that f(0) = at_0 and f(10) = at_10.
    """
    # Calculate the base b using the provided points
    b = (at_10 / at_0) ** (1 / 10)
    def exp_fn(x):
        return at_0 * (b ** x)
    return exp_fn


def main():
    pattern_file = sys.argv[1]
    output_file = sys.argv[2]
    lazy_grid = LazyGrid()
    pattern = open(pattern_file, 'r', encoding='utf-8').read()
    grid = parse_rle(pattern)
    lazy_grid.add_grid(Point2D(0, 0), grid)
    (height, width) = grid.shape
    print(f"Grid size: {width}x{height}")
    resolution = Point2D(192, 108)
    if height < width / resolution.x * resolution.y:
        height = width / resolution.x * resolution.y
    else:
        width = height / resolution.y * resolution.x
    rectangle = RectangleExtent(
        Range(0, width),
        Range(0, height)
    )
    print(f"Rectangle size: {width:.1f}x{height:.1f}")
    densities = snapshot_from_grid(lazy_grid, CuboidExtent(rectangle.x_range, rectangle.y_range, Range(0, 1)), resolution);
    assert densities.shape == (resolution.y, resolution.x)
    image = Image.fromarray((densities * 255).astype(np.uint8), 'L')
    image.save('output.png')
    return
    duration = 1
    fps = 2
    speed_fn = _make_speed_fn(2, 60)
    create_video(
        grid=lazy_grid,
        speed_fn=speed_fn,
        view_fn=lambda sec: rectangle,
        resolution=resolution,
        duration=duration,
        fps=fps,
        output=output_file,
    )
