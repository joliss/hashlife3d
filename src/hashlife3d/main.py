import sys
import os

import numpy as np
from ranges import Range
from PIL import Image

from .video import create_video
from .grid import LazyGrid
from .parsers import parse_file
from .extent import Point2D, RectangleExtent, CuboidExtent
from .camera import snapshot_from_grid


def make_speed_fn(x0_val, y0_val, x1_val, y1_val, x2_val, y2_val):
    """
    Return a function f(t) of the form a + b * x^2 + c * x^3 such that

    f(x0_val) = y0_val
    f(x1_val) = y1_val
    f(x2_val) = y2_val
    """

    # Implemented by ChatGPT.
    # https://chat.openai.com/share/c9881422-f217-4209-8c5f-fcc4af01514e

    from sympy import symbols, Eq, solve

    # Define the symbols
    a, b, c, x, y2 = symbols('a b c x y2')

    # Define the equations based on the given function and points
    eq1 = Eq(a + b*x0_val**2 + c*x0_val**3, y0_val)
    eq2 = Eq(a + b*x1_val**2 + c*x1_val**3, y1_val)
    eq3 = Eq(a + b*x2_val**2 + c*x2_val**3, y2_val)

    # Solve the system of equations
    solutions = solve((eq1, eq2, eq3), (a, b, c))

    # Check if solutions for b and c are positive
    if solutions[b] <= 0:
        raise ValueError("b must be greater than 0.")
    if solutions[c] <= 0:
        # We solve for the new y2 that makes c >= 0
        eq3_new_c = Eq(solutions[a] + solutions[b] * x2_val**2, y2)
        new_y2 = solve(eq3_new_c, y2)[0]
        raise ValueError(f"c must be greater than 0. Try y2 >= {new_y2.evalf()} to ensure c >= 0.")

    # Define the function with the found coefficients
    a = float(solutions[a])
    b = float(solutions[b])
    c = float(solutions[c])
    def f(x):
        return a + b * x**2 + c * x**3

    return f


def main():
    pattern_file = sys.argv[1]
    output_file = sys.argv[2]
    grid, extent = parse_file(pattern_file)
    resolution = Point2D(3840, 2160)
    # resolution = Point2D(1920, 1080)
    # resolution = Point2D(1280, 720)
    if extent.height < extent.width / resolution.x * resolution.y:
        new_height = extent.width / resolution.x * resolution.y
        rectangle = RectangleExtent(
            extent.x_range,
            Range(
                extent.y_range.start - (new_height - extent.height) / 2,
                extent.y_range.start - (new_height - extent.height) / 2 + new_height)
        )
    else:
        new_width = extent.height / resolution.y * resolution.x
        rectangle = RectangleExtent(
            Range(
                extent.x_range.start - (new_width - extent.width) / 2,
                extent.x_range.start - (new_width - extent.width) / 2 + new_width),
            extent.y_range
        )
    print(f"Rectangle size: {rectangle.width:.1f}x{rectangle.height:.1f}")
    densities = snapshot_from_grid(grid, CuboidExtent(rectangle.x_range, rectangle.y_range, Range(0, 1)), resolution);
    assert densities.shape == (resolution.y, resolution.x)
    image = Image.fromarray((densities * 255).astype(np.uint8), 'L')
    image.save('output.png')
    # return
    duration = 60
    fps = 30
    # speed_fn = make_speed_fn(0, 1, 10, 7, 60, 1000)
    speed_fn = make_speed_fn(0, 1, 10, 500, 60, 100000)``
    create_video(
        grid=grid,
        speed_fn=speed_fn,
        view_fn=lambda sec: rectangle,
        resolution=resolution,
        duration=duration,
        fps=fps,
        output=output_file,
        over_depth=lambda sec: 0.5
    )
