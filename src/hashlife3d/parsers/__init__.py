from ranges import Range

from .mc import parse as parse_mc
from .rle import parse as parse_rle
from ..grid import LazyGrid, GridFromQuadtree
from ..extent import RectangleExtent, Point2D, Range

def parse_file(filename):
    """
    Parse an .mc or .rle file. Returns (grid, rectangle_extent).
    """
    if filename.endswith('.gz'):
        import gzip
        f = gzip.open(filename, 'rt', encoding='utf-8')
    else:
        f = open(filename, 'r', encoding='utf-8')
    if filename.endswith('.mc') or filename.endswith('.mc.gz'):
        quadtree = parse_mc(f.read())
        grid = GridFromQuadtree(quadtree)
        _, extent = grid.initial_quadtree_and_extent()
        return (grid, extent)
    else:
        grid = parse_rle(f.read())
        lazy_grid = LazyGrid()
        lazy_grid.add_grid(Point2D(0, 0), grid)
        (height, width) = grid.shape
        return (lazy_grid, RectangleExtent(Range(0, width), Range(0, height)))
