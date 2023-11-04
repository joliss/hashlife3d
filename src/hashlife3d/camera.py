"""
This modules helps you take pictures of arbitrary rectangles, temporally
anti-aliased across any number of generations.
"""

from collections import defaultdict
import math

import numpy as np
from skimage.measure import block_reduce
from skimage.transform import warp, AffineTransform
from ranges import Range

from .canonical import CacheStats
from .node import BinaryTree, BinaryTreeBranch, BinaryTreeLeaf, flatten_quadtree_array, Quadtree, QuadtreeBranch
from .extent import CuboidExtent, RectangleExtent, Point2D
from .grid import LazyGrid, GridFromQuadtree


def _octree_extent_from_quadtree(quadtree_extent: RectangleExtent):
    assert quadtree_extent.width % 4 == 0
    quarter = quadtree_extent.width // 4
    octree_extent = CuboidExtent(
        Range(quadtree_extent.x_range.start + quarter, quadtree_extent.x_range.end - quarter),
        Range(quadtree_extent.y_range.start + quarter, quadtree_extent.y_range.end - quarter),
        Range(1, quarter + 1)
    )
    octree_extent_and_base = CuboidExtent(
        octree_extent.x_range,
        octree_extent.y_range,
        Range(0, octree_extent.t_range.end)
    )
    return (octree_extent, octree_extent_and_base)



def find_required_quadtree(cuboid):
    """
    Find extent of quadtree that will generate an octree containing the
    specified cuboid.

    Returns a tuple of extents (quadtree: RectangleExtent, octree:
    CuboidExtent).
    """
    # A square denoting a quadtree extent, containing the origin.
    origin_square = RectangleExtent(Range(0, 4), Range(0, 4))
    # Keep extending the origin_square 4x, alternating between north-west
    # direction and south-east direction. At each step, tile the quadtree on an
    # infinite plane. If one of the tiles would generate an octree containing
    # the given range, return its extent.
    def containing_tile(origin_square):
        assert origin_square.width == origin_square.height
        assert origin_square.width % 4 == 0
        # tiled_square is origin_square translated by an integer multiple so as
        # to contain the top left corner of the cuboid.
        x_offset = int(cuboid.x_range.start - origin_square.x_range.start) // origin_square.width * origin_square.width
        y_offset = int(cuboid.y_range.start - origin_square.y_range.start) // origin_square.height * origin_square.height
        tiled_square = RectangleExtent(
            Range(origin_square.x_range.start + x_offset, origin_square.x_range.end + x_offset),
            Range(origin_square.y_range.start + y_offset, origin_square.y_range.end + y_offset)
        )
        (octree_extent, octree_extent_and_base) = _octree_extent_from_quadtree(tiled_square)
        if cuboid not in octree_extent_and_base:
            return None
        return (tiled_square, octree_extent)
    while True:
        tile = containing_tile(origin_square)
        if tile is not None:
            return tile
        if (origin_square.x_range.start + origin_square.x_range.end) // 2 > 0:
            # Midpoint is positive; expand north-west
            origin_square.x_range.start -= origin_square.width
            origin_square.y_range.start -= origin_square.height
        else:
            # Midpoint is negative; expand south-east
            origin_square.x_range.end += origin_square.width
            origin_square.y_range.end += origin_square.height

# density_cache[original_depth][quadtree][side_length] is a numpy array of shape (side_length, side_length)
def defaultdict_of_dicts():
    return defaultdict(dict)
density_cache = defaultdict(defaultdict_of_dicts)
density_cache_max_side_length = 128
density_cache_stats = CacheStats()

def densities_square(population_quadtree, original_depth, resolution: int):
    return densities_square_up_to_max_side_length(population_quadtree, original_depth, resolution)

    children = np.array([[population_quadtree]], dtype=object)
    child_resolution = resolution
    block_count = 1 # per dimension
    while resolution > density_cache_max_side_length and children[0, 0].level > 0:
        children = flatten_quadtree_array(children)
        assert child_resolution % 2 == 0
        child_resolution //= 2
        block_count *= 2
    print("calculating child blocks")

    # child_blocks = [[
    #     densities_square_up_to_max_side_length(child, original_depth, child_resolution)
    #     for child in row] for row in children]
    # child_blocks = np.vectorize(lambda child: densities_square_up_to_max_side_length(child, original_depth, child_resolution), signature='()->(n,n)')(children)
    # print(f"got child_blocks of shape {len(child_blocks)}x{len(child_blocks[0])}")
    # # array_4d = np.array(child_blocks).reshape(block_count, block_count, child_resolution, child_resolution)
    # array_4d = child_blocks.reshape(block_count, block_count, child_resolution, child_resolution)
    # print("calculating total block")
    # concatenated_rows = np.concatenate(array_4d, axis=2)
    # total_block = np.concatenate(concatenated_rows, axis=0)

    total_block = np.empty((resolution, resolution), dtype=np.float32)
    for y in range(block_count):
        for x in range(block_count):
            child = children[y, x]
            child_block = densities_square_up_to_max_side_length(child, original_depth, child_resolution)
            total_block[y*child_resolution:(y+1)*child_resolution, x*child_resolution:(x+1)*child_resolution] = child_block

    # child_blocks = [[
    #     densities_square_up_to_max_side_length(child, original_depth, child_resolution)
    #     for child in row] for row in children]
    # total_block = np.block(child_blocks)

    print("  done")
    return total_block


def densities_square_up_to_max_side_length(population_quadtree, original_depth, resolution: int):
    """
    Return a ndarray with shape (resolution, resolution) of float32 values in
    the range [0, 1] representing the population density in the
    population_quadtree. The population_quadtree must have been originally
    generated from a binary tree with depth `original_depth`.
    """
    # print(resolution)
    def recurse():
        # children = [[
        #     densities_square_up_to_max_side_length(child, original_depth, resolution // 2)
        #     for child in row] for row in population_quadtree.children]
        # return np.block(children)
        block = np.empty((resolution, resolution), dtype=np.float32)
        for y in range(2):
            for x in range(2):
                child = population_quadtree.children[y, x]
                child_block = densities_square_up_to_max_side_length(child, original_depth, resolution // 2)
                block[y*resolution//2:(y+1)*resolution//2, x*resolution//2:(x+1)*resolution//2] = child_block
        return block
    def single_value():
        return np.full((resolution, resolution), population_quadtree.total_population() / population_quadtree.width() ** 2 / original_depth, dtype=np.float32)
    if resolution > density_cache_max_side_length:
        if population_quadtree.level >= 1:
            return recurse()
        else:
            return single_value()
    snapshot = density_cache[original_depth][population_quadtree].get(resolution)
    if snapshot is not None:
        density_cache_stats.hits += 1
        return snapshot
    density_cache_stats.misses += 1
    for res2, snap2 in sorted(density_cache[original_depth][population_quadtree].keys()):
        if res2 > resolution:
            # Downscale
            assert res2 % resolution == 0
            snapshot = block_reduce(snap2, (res2 // resolution, res2 // resolution), np.mean)
            break
    else:
        if resolution == 1 or population_quadtree.level == 0:
            # Cannot recurse further, or do not need to.
            snapshot = single_value()
        else:
            snapshot = recurse()
    density_cache[original_depth][population_quadtree][resolution] = snapshot
    return snapshot

def densities_snapshot(binary_tree, original_cuboid, target_cuboid, target_resolution: Point2D):
    """
    Create a densities snapshot of a rectangle contained within a binary,
    averaged across multiple generations; thus of a cuboid.
    """
    assert target_cuboid in original_cuboid
    assert original_cuboid.depth == binary_tree.depth()
    if binary_tree.level == 0:
        return _snapshot_from_population_quadtree(binary_tree.population_quadtree(), original_cuboid, target_cuboid, target_resolution)
    cuboid_pair = original_cuboid.split_t()
    snapshot_pair = [None, None]
    for t in (0, 1):
        intersection = cuboid_pair[t].t_range.intersection(target_cuboid.t_range)
        if intersection:
            new_target = CuboidExtent(target_cuboid.x_range, target_cuboid.y_range, intersection)
            if intersection == cuboid_pair[t].t_range:
                # Spans entire depth; base case
                population_quadtree = binary_tree.children[t].population_quadtree()
                snapshot_pair[t] = _snapshot_from_population_quadtree(population_quadtree, cuboid_pair[t], new_target, target_resolution)
            else:
                # Doesn't span entire depth; recurse further
                snapshot_pair[t] = densities_snapshot(binary_tree.children[t], cuboid_pair[t], new_target, target_resolution)
            if intersection.length() < target_cuboid.t_range.length():
                snapshot_pair[t] *= intersection.length() / target_cuboid.t_range.length()
    assert snapshot_pair[0] is not None or snapshot_pair[1] is not None
    if snapshot_pair[0] is None:
        return snapshot_pair[1]
    if snapshot_pair[1] is None:
        return snapshot_pair[0]
    return snapshot_pair[0] + snapshot_pair[1]

def _snapshot_from_population_quadtree(population_quadtree, original_cuboid, target_cuboid, target_resolution: Point2D):
    """
    Return a snapshot (2D ndarray) of the population_quadtree at resolution `target_resolution`.
    """
    assert target_cuboid in original_cuboid
    sl = population_quadtree.side_length()
    assert sl == original_cuboid.width == original_cuboid.height
    if population_quadtree.level >= 2:
        # Try to narrow down
        grand_children = flatten_quadtree_array(population_quadtree.children)
        for x in (0, 1, 2):
            for y in (0, 1, 2):
                x_start = original_cuboid.x_range.start + x * sl // 4
                y_start = original_cuboid.y_range.start + y * sl // 4
                new_cuboid = CuboidExtent(
                    Range(x_start, x_start + sl // 2),
                    Range(y_start, y_start + sl // 2),
                    original_cuboid.t_range
                )
                if target_cuboid in new_cuboid:
                    return _snapshot_from_population_quadtree(QuadtreeBranch(grand_children[y:y+2, x:x+2]), new_cuboid, target_cuboid, target_resolution)
    resolution = _get_resolution(original_cuboid, target_cuboid, target_resolution)
    densities = densities_square(population_quadtree, original_cuboid.depth, resolution)
    factor = resolution / sl
    x_range = Range(factor * (target_cuboid.x_range.start - original_cuboid.x_range.start), factor * (target_cuboid.x_range.end - original_cuboid.x_range.start))
    y_range = Range(factor * (target_cuboid.y_range.start - original_cuboid.y_range.start), factor * (target_cuboid.y_range.end - original_cuboid.y_range.start))
    translation = (x_range.start, y_range.start)
    scale = (x_range.length() / target_resolution.x, y_range.length() / target_resolution.y)
    transform = AffineTransform(translation=translation, scale=scale)
    out = warp(densities, transform, output_shape=(target_resolution.y, target_resolution.x), order=1, preserve_range=True)
    return out

def _get_resolution(original_cuboid, target_cuboid, target_resolution: Point2D) -> int:
    """
    Return a resolution for the densities array of the original_cuboid that is
    high enough that we can sample target_cuboid with target_resolution from it
    with good quality.
    """
    assert target_cuboid in original_cuboid
    minimum_factor = 1.5
    minimum_resolutions = [
        target_resolution.x * original_cuboid.width / target_cuboid.width,
        target_resolution.y * original_cuboid.height / target_cuboid.height
    ]
    resolution = max(minimum_resolutions) * minimum_factor
    resolution = _round_up_to_power_of_2(resolution)
    return resolution

def _round_up_to_power_of_2(x: float) -> int:
    return 2 ** math.ceil(math.log(x, 2))

def _repeat_binary_tree(binary_tree: BinaryTree, n: int):
    if n == 1:
        return binary_tree
    assert n % 2 == 0
    child = _repeat_binary_tree(binary_tree, n // 2)
    return BinaryTreeBranch(np.array([child, child], dtype=object))

def _extend_binary_tree(binary_tree: BinaryTree, octree_extent: CuboidExtent, base_quadtree: Quadtree):
    grandchildren = flatten_quadtree_array(base_quadtree.children)
    base_center = QuadtreeBranch(grandchildren[1:3, 1:3])
    extended_binary_tree = BinaryTreeBranch(np.array([
        _repeat_binary_tree(BinaryTreeLeaf(base_center), octree_extent.depth),
        binary_tree
    ], dtype=object))
    extended_octree = CuboidExtent(
        octree_extent.x_range,
        octree_extent.y_range,
        Range(octree_extent.t_range.start - octree_extent.depth, octree_extent.t_range.end)
    )
    return (extended_binary_tree, extended_octree)

def _get_containing_quadtree_for_grid_from_quadtree(grid: GridFromQuadtree, target_cuboid: CuboidExtent):
    (quadtree, quadtree_extent) = grid.initial_quadtree_and_extent()
    while True:
        (octree_extent, octree_extent_and_base) = _octree_extent_from_quadtree(quadtree_extent)
        if target_cuboid in octree_extent_and_base:
            return quadtree, octree_extent
        (quadtree, quadtree_extent) = grid.expand_quadtree_and_extent(quadtree, quadtree_extent)

def snapshot_from_grid(grid, target_cuboid, target_resolution: Point2D):
    """
    Create a densities snapshot of a rectangle contained within a grid,
    averaged across multiple generations; thus of a cuboid.
    """
    if isinstance(grid, LazyGrid):
        (quadtree_extent, octree_extent) = find_required_quadtree(target_cuboid)
        print(f"quadtree_extent: {quadtree_extent}")
        base_quadtree = grid.get_quadtree(quadtree_extent)
    else:
        assert isinstance(grid, GridFromQuadtree)
        (base_quadtree, octree_extent) = _get_containing_quadtree_for_grid_from_quadtree(grid, target_cuboid)
    octree = base_quadtree.generate_octree()
    binary_tree = octree.quadtrees()
    # The octree we generated starts at t=1. In order to be able to snapshot the
    # baselayer, we awkwardly extend the binary_tree by repeating the part of
    # the base layer that's underneath the octree.
    (binary_tree, octree_extent) = _extend_binary_tree(binary_tree, octree_extent, base_quadtree)
    snapshot = densities_snapshot(binary_tree, octree_extent, target_cuboid, target_resolution)
    print('density cache', density_cache_stats)
    return snapshot
