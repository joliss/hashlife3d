from dataclasses import dataclass, field

import numpy as np

from .canonical import Canonical
from .state import State
from .grid import Grid


class QuadTree(Canonical):
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

    def next_generation_octree(self):
        assert self.level >= 2
        if self.level == 2:
            return self._next_generation_octree_leaf()

        # See the color coding in Figure 4 on
        # https://www.drdobbs.com/jvm/an-algorithm-for-compressing-space-and-t/184406478?pgno=1

        # First, generate the red squares as octrees; unlike the HashLife
        # algorithm, we don't just slice, but also advance a generation
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
        # Instead of the red squares in the figure, we generate cubes
        red_octrees = np.array(
            [[container.next_generation_octree() for container in row] for row in red_square_containers],
            dtype=object
        )
        assert red_octrees.shape == (3, 3)
        # green_square_containers = np.array(
        #     [[QuadTreeBranch.from_children(red_squares[y:y+2, x:x+2]) for x in range(2)] for y in range(2)],
        #     dtype=object
        # )
        # assert green_square_containers.shape == (2, 2)
        # # Once again instead of the green squares in the figure, we generate cubes

    def _next_generation_octree_leaf(self):
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
    Flatten a 2x2 array of quad tree children into 4x4, and a 4x4 array into 8x8.
    """
    assert arr.ndim == 2
    assert arr[0][0].level >= 1
    width = arr.shape[0]
    assert width == arr.shape[1]
    flattened_arr = np.empty((2 * width, 2 * width), dtype=object)
    for y in range(width):
        for x in range(width):
            flattened_arr[2*y:2*y+2, 2*x:2*x+2] = arr[y][x].children
    return flattened_arr


def flatten_octree_array(arr):
    """
    Flatten a 2x2x2 array of octree children into 4x4x4, and a 4x4x4 array into 8x8x8.
    """
    assert arr.ndim == 3
    assert arr[0][0][0].level >= 1
    width = arr.shape[0]
    assert width == arr.shape[1] == arr.shape[2]
    flattened_arr = np.empty((2 * width, 2 * width, 2 * width), dtype=object)
    for t in range(width):
        for y in range(width):
            for x in range(width):
                flattened_arr[2*t:2*t+2, 2*y:2*y+2, 2*x:2*x+2] = arr[t][y][x].children
    return flattened_arr
