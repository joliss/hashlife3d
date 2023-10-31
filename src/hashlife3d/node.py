from dataclasses import dataclass, field

import numpy as np

from .canonical import Canonical, interned
from .state import State
from .grid import Grid


class QuadTree(Canonical):
    """
    A quadtree, specifically a tree pyramid with a square base.
    """
    def width(self):
        return 2 ** self.level

    @staticmethod
    def from_str(s):
        grid = Grid.from_str(s)
        return QuadTree.from_grid(grid)

    def __str__(self):
        return str(self.to_grid())

    @staticmethod
    def from_grid(grid):
        if not isinstance(grid, Grid):
            grid = Grid.from_list(grid)
        width = grid.shape[0]
        assert width == grid.shape[1]
        if width == 1:
            return QuadTreeLeaf(grid[0, 0])
        assert width % 2 == 0
        nw = QuadTree.from_grid(grid[:width//2, :width//2])
        ne = QuadTree.from_grid(grid[:width//2, width//2:])
        sw = QuadTree.from_grid(grid[width//2:, :width//2])
        se = QuadTree.from_grid(grid[width//2:, width//2:])
        return QuadTreeBranch(nw, ne, sw, se)


@dataclass(frozen=True, eq=False)
class QuadTreeLeaf(QuadTree):
    state: State
    level = 0

    def __post_init__(self):
        assert isinstance(self.state, State)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.state})"

    _STRINGS = ['_', 'X', '.']

    def to_grid(self):
        return Grid.from_list([[self.state]])

QuadTreeLeaf.DEAD = QuadTreeLeaf(State.DEAD)
QuadTreeLeaf.ALIVE = QuadTreeLeaf(State.ALIVE)
QuadTreeLeaf.UNINHABITABLE = QuadTreeLeaf(State.UNINHABITABLE)


@dataclass(frozen=True, eq=False)
class QuadTreeBranch(QuadTree):
    nw: QuadTree
    ne: QuadTree
    sw: QuadTree
    se: QuadTree
    level: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "level", self._checklevel(self.nw, self.ne, self.sw, self.se))

    @classmethod
    def _checklevel(cls, nw, ne, sw, se):
        child_level = nw.level
        if child_level != ne.level or child_level != sw.level or child_level != se.level:
            raise ValueError("Child nodes must have the same level")
        return child_level + 1

    @property
    def children(self):
        return np.array([
            [self.nw, self.ne],
            [self.sw, self.se]
        ], dtype=object)

    @classmethod
    def from_children(cls, children):
        return cls(children[0, 0], children[0, 1], children[1, 0], children[1, 1])

    def __repr__(self):
        return f"{self.__class__.__name__}(level={self.level})"

    def __iter__(self):
        yield from [self.nw, self.ne]
        yield from [self.sw, self.se]

    def to_grid(self):
        grids = np.vectorize(lambda child: child.to_grid(), otypes=[object])(self.children)
        return np.block(grids.tolist()).view(Grid)

    @interned
    def generate_octree(self):
        """
        Given a (4n, 4n) QuadTree, generate a (n, 2n, 2n) Octree, computing n
        generations.
        """
        assert self.level >= 2
        if self.level == 2:
            # n = 1 case
            return self._generate_octree_leaf()

        # n is a multiple of 2 from here on out.

        # See the color coding in Figure 4 on
        # https://www.drdobbs.com/jvm/an-algorithm-for-compressing-space-and-t/184406478?pgno=1

        # First, generate the red squares as octrees.
        four_by_four = flatten_quad_tree_array(self.children)
        eight_by_eight = flatten_quad_tree_array(four_by_four)
        assert four_by_four.shape == (4, 4)
        assert eight_by_eight.shape == (8, 8)
        # The red_square_containers aren't in the figure, but imagine each red
        # square being contained in a 4x4 square.
        red_square_containers = np.array(
            [[QuadTreeBranch.from_children(four_by_four[y:y+2, x:x+2]) for x in range(3)] for y in range(3)],
            dtype=object
        )
        assert red_square_containers.shape == (3, 3)
        # Instead of the red squares in the figure, we generate cuboids,
        # advancing n/2 generations. This is unlike the HashLife algorithm,
        # which instead of advancing just slices to generate its red squares.
        generate_octree = np.vectorize(lambda container: container.generate_octree(), otypes=[object])
        red_octrees = generate_octree(red_square_containers)
        assert red_octrees.shape == (3, 3)
        red_squares = np.vectorize(lambda octree: octree.most_recent_quad_tree(), otypes=[object])(red_octrees)
        assert red_squares.shape == (3, 3)
        green_square_containers = np.array(
            [[QuadTreeBranch.from_children(red_squares[y:y+2, x:x+2]) for x in range(2)] for y in range(2)],
            dtype=object
        )
        assert green_square_containers.shape == (2, 2)
        # Once again instead of the green squares in the figure, we generate
        # cuboids, advancing another n/2 generations.
        green_octrees = generate_octree(green_square_containers)
        assert green_octrees.shape == (2, 2)
        # Finally, assemble the green cuboids and the parts of the red cuboids that
        # are behind them into a single octree.
        if red_octrees[0, 0].level >= 1:
            # The 6x6 large red square in the figure is made up of two layers for our HashLife 3D algorithm.
            two_by_six_by_six = flatten_octree_array(red_octrees.reshape(1, 3, 3))
            assert two_by_six_by_six.shape == (2, 6, 6)
            red_parts = np.array(
                [[OctreeBranch(two_by_six_by_six[0:2, y:y+2, x:x+2])
                  for x in range(1, 5, 2)] for y in range(1, 5, 2)],
                dtype=object)
        else:
            assert red_octrees[0, 0].level == 0
            assert self.level == 3
            # Geometrically, this is the same as the level >= 1 case, but we
            # need to work the QuadTrees in the leaf nodes here.
            six_by_six = flatten_quad_tree_array(np.vectorize(lambda o: o.quad_tree, otypes=[object])(red_octrees))
            assert six_by_six.shape == (6, 6)
            red_parts = np.array(
                [[OctreeLeaf(QuadTreeBranch.from_children(six_by_six[y:y+2, x:x+2]))
                  for x in range(1, 5, 2)] for y in range(1, 5, 2)],
                dtype=object)
        return OctreeBranch(np.stack([red_parts, green_octrees], axis=0))

    def _generate_octree_leaf(self):
        assert self.level == 2
        leafs = flatten_quad_tree_array(self.children)
        matrix = np.vectorize(lambda leaf: leaf.state, otypes=[object])(leafs)
        def neighbor_count(x, y):
            neighborhood = np.copy(matrix[y-1:y+2, x-1:x+2])
            neighborhood[1][1] = State.DEAD # don't count self
            return np.count_nonzero(neighborhood == State.ALIVE)
        new_states = [[matrix[y][x].next_state(neighbor_count(x, y)) for x in range(1, 3)] for y in range(1, 3)]
        quad_tree = QuadTree.from_grid(new_states)
        return OctreeLeaf(quad_tree)


class Octree(Canonical):
    def width(self):
        return 2 * 2 ** self.level

    def depth(self):
        return 2 ** self.level

    def __repr__(self):
        return f"{self.__class__.__name__}(level={self.level})"

    def __str__(self):
        return "\n\n".join([str(quad_tree) for quad_tree in self.quad_trees()])

    @interned
    def most_recent_quad_tree(self):
        return self.quad_trees()[-1]


@dataclass(frozen=True, eq=False)
class OctreeLeaf(Octree):
    quad_tree: QuadTree
    level = 0

    def __post_init__(self):
        if self.quad_tree.level != 1:
            raise ValueError(f"QuadTree must be level 1")

    def quad_trees(self):
        arr = np.array([self.quad_tree], dtype=object)
        assert arr.shape == (1,)
        return arr


@dataclass(frozen=True, eq=False)
class OctreeBranch(Octree):
    children: np.ndarray # 2x2x2 array of Octrees, children[t][y][x]
    level: int = field(init=False)

    def __post_init__(self):
        assert self.children.shape == (2, 2, 2)
        object.__setattr__(self, "level", self._checklevel(self.children))

    @classmethod
    def _checklevel(cls, children):
        child_level = children[0, 0, 0].level
        get_level = np.vectorize(lambda child: child.level, otypes=[int])
        if not np.all(get_level(children) == child_level):
            raise ValueError("Child nodes must have the same level")
        return child_level + 1

    def __iter__(self):
        yield from self.children

    def quad_trees(self):
        assert self.level <= 4, "this becomes too slow"
        def reassemble_quad_tree(octree_quad):
            to_quad_trees = np.vectorize(lambda octree: octree.quad_trees(), otypes=[object])
            [[nws, nes], [sws, ses]] = to_quad_trees(octree_quad)
            return np.array([QuadTreeBranch(nw, ne, sw, se) for (nw, ne, sw, se) in zip(nws, nes, sws, ses)], dtype=object)
        return np.concatenate([
            reassemble_quad_tree(self.children[t])
            for t in range(2)
        ])


def flatten_quad_tree_array(arr):
    """
    Flatten an m by n array of quad trees into a 2m by 2n array.
    """
    assert arr.ndim == 2
    assert arr[0][0].level >= 1
    height = arr.shape[0]
    width = arr.shape[1]
    flattened_arr = np.empty((2 * width, 2 * height), dtype=object)
    for y in range(height):
        for x in range(width):
            flattened_arr[2*y:2*y+2, 2*x:2*x+2] = arr[y][x].children
    return flattened_arr


def flatten_octree_array(arr):
    """
    Flatten an m by n by o array of octrees into a 2m by 2n by 2o array.
    """
    assert arr.ndim == 3
    assert arr[0][0][0].level >= 1
    depth = arr.shape[0]
    height = arr.shape[1]
    width = arr.shape[2]
    assert width == arr.shape[1] == arr.shape[2]
    flattened_arr = np.empty((2 * depth, 2 * height, 2 * width), dtype=object)
    for t in range(depth):
        for y in range(height):
            for x in range(width):
                flattened_arr[2*t:2*t+2, 2*y:2*y+2, 2*x:2*x+2] = arr[t][y][x].children
    return flattened_arr
