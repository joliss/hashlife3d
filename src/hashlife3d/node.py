from dataclasses import dataclass, field

import numpy as np

from .canonical import Canonical, interned, interned_with_progress, unwrap_progress_iterable
from .state import State
from .grid import Grid


class KdQuadtree(Canonical):
    """
    A k-dimensional generalization of the quadtree.

    Unlike a true k-d tree, each node of this splits in each dimension
    simultaneously. Thus in 2D, this is a quadtree; in 3D, an octree.
    """

    __slots__ = ()

    def __repr__(self):
        return f"{self.__class__.__name__}(level={self.level})"

    def side_length(self):
        return 2 ** self.level


@dataclass(frozen=True, eq=False, repr=False)
class KdQuadtreeLeaf(KdQuadtree):
    __slots__ = ("element",)
    element: object
    level = 0

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.element)})"


# @dataclass(frozen=True, eq=False, repr=False)
class KdQuadtreeBranch(KdQuadtree):
    # __slots__ = ("children", "level")
    __slots__ = ()
    children: np.ndarray # k-dimensional array with shape (2, 2, 2, ...)
    # level: int = field(init=False)

    def __init__(self, children):
        self.children = children
        object.__setattr__(self, "level", self._checklevel())
        assert self.children.ndim == self.ndim
        # This assertion is too slow to run in production.
        # assert all(shape == 2 for shape in self.children.shape)

    def _checklevel(self):
        child_level = self.children.flat[0].level
        # This assertion is too slow to run in production.
        # assert all(child.level == child_level for child in self.children.ravel()), "Child nodes must have the same level"
        return child_level + 1

    def __iter__(self):
        yield from self.children.ravel()


class Quadtree(KdQuadtree):
    """
    A quadtree, specifically a tree pyramid with a square base.
    """

    __slots__ = ()
    ndim = 2

    def width(self):
        return self.side_length()

    @staticmethod
    def from_str(s):
        grid = Grid.from_str(s)
        return Quadtree.from_grid(grid)

    def __str__(self):
        return str(self.to_grid())

    @staticmethod
    def from_grid(grid):
        if not isinstance(grid, Grid):
            grid = Grid.from_list(grid)
        width = grid.shape[0]
        assert width == grid.shape[1]
        if width == 1:
            return QuadtreeLeaf(grid[0, 0])
        assert width % 2 == 0
        nw = Quadtree.from_grid(grid[:width//2, :width//2])
        ne = Quadtree.from_grid(grid[:width//2, width//2:])
        sw = Quadtree.from_grid(grid[width//2:, :width//2])
        se = Quadtree.from_grid(grid[width//2:, width//2:])
        return QuadtreeBranch(np.array([[nw, ne], [sw, se]], dtype=object))

    @interned
    def total_population(self):
        """
        Return the total population of the quadtree.
        """
        if self.level == 0:
            return self.element
        else:
            child_populations = np.vectorize(lambda child: child.total_population(), otypes=[object])(self.children)
            return add_population_counts(
                add_population_counts(child_populations[0, 0], child_populations[0, 1]),
                add_population_counts(child_populations[1, 0], child_populations[1, 1])
            )

    @staticmethod
    @interned
    def empty(level: int, default: State = State.DEAD):
        """
        Return an empty quadtree of the given level.
        """
        if level == 0:
            return QuadtreeLeaf(default)
        else:
            return QuadtreeBranch(np.array([
                [Quadtree.empty(level - 1, default), Quadtree.empty(level - 1, default)],
                [Quadtree.empty(level - 1, default), Quadtree.empty(level - 1, default)]
            ], dtype=object))


class QuadtreeLeaf(Quadtree, KdQuadtreeLeaf):
    __slots__ = ()
    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.element, State)

    def to_grid(self):
        return Grid.from_list([[self.element]])


class QuadtreeBranch(Quadtree, KdQuadtreeBranch):
    __slots__ = ('nw', 'ne', 'sw', 'se', 'level')

    def __init__(self, children):
        assert children.shape == (2, 2)
        self.nw, self.ne = children[0]
        self.sw, self.se = children[1]
        self.level = self._checklevel()

    @property
    def children(self):
        return np.array([[self.nw, self.ne], [self.sw, self.se]], dtype=object)

    def to_grid(self):
        grids = np.vectorize(lambda child: child.to_grid(), otypes=[object])(self.children)
        return np.block(grids.tolist()).view(Grid)

    @interned_with_progress(do_log=False)
    def generate_octree_with_progress(self):
        """
        Given a (4n, 4n) Quadtree, generate a (n, 2n, 2n) Octree, computing n
        generations.
        """
        do_yield = self != Quadtree.empty(self.level)
        assert self.level >= 2
        if self.level == 2:
            # n = 1 case
            return self._generate_octree_leaf()
        progress_suffix = (0,) * (self.level - 3)

        # n is a multiple of 2 from here on out.

        # See the color coding in Figure 4 on
        # https://www.drdobbs.com/jvm/an-algorithm-for-compressing-space-and-t/184406478?pgno=1

        # First, generate the red squares as octrees.
        four_by_four = flatten_quadtree_array(self.children)
        eight_by_eight = flatten_quadtree_array(four_by_four)
        assert four_by_four.shape == (4, 4)
        assert eight_by_eight.shape == (8, 8)
        # The red_square_containers aren't in the figure, but imagine each red
        # square being contained in a 4x4 square.
        red_square_containers = np.array(
            [[QuadtreeBranch(four_by_four[y:y+2, x:x+2]) for x in range(3)] for y in range(3)],
            dtype=object
        )
        assert red_square_containers.shape == (3, 3)
        current_progress = 0
        def generate_octrees(octree_ndarray):
            """Basically np.vectorize(self.generate_octree), but with progress reporting."""
            nonlocal current_progress
            assert octree_ndarray.ndim == 2
            r = np.empty(octree_ndarray.shape, dtype=object)
            for y in range(octree_ndarray.shape[0]):
                for x in range(octree_ndarray.shape[1]):
                    next_progress = current_progress
                    progress_base = (current_progress,)
                    progress_end = (current_progress + 1,) + progress_suffix
                    iterator = octree_ndarray[y, x].generate_octree_with_progress(progress_range=(progress_base, progress_end))
                    try:
                        while True:
                            progress = next(iterator)
                            next_progress = current_progress + 1
                            if do_yield:
                                yield progress
                    except StopIteration as e:
                        generated_octree = e.value
                    r[y, x] = generated_octree
                    current_progress = next_progress
            return r
        # Instead of the red squares in the figure, we generate cuboids,
        # advancing n/2 generations. This is unlike the HashLife algorithm,
        # which instead of advancing just slices to generate its red squares.
        red_octrees = yield from generate_octrees(red_square_containers)
        assert red_octrees.shape == (3, 3)
        red_squares = np.vectorize(Octree.most_recent_quadtree, otypes=[object])(red_octrees)
        assert red_squares.shape == (3, 3)
        green_square_containers = np.array(
            [[QuadtreeBranch(red_squares[y:y+2, x:x+2]) for x in range(2)] for y in range(2)],
            dtype=object
        )
        assert green_square_containers.shape == (2, 2)
        # Once again instead of the green squares in the figure, we generate
        # cuboids, advancing another n/2 generations.
        green_octrees = yield from generate_octrees(green_square_containers)
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
            # need to work the Quadtrees in the leaf nodes here.
            six_by_six = flatten_quadtree_array(np.vectorize(lambda o: o.quadtree, otypes=[object])(red_octrees))
            assert six_by_six.shape == (6, 6)
            red_parts = np.array(
                [[OctreeLeaf(QuadtreeBranch(six_by_six[y:y+2, x:x+2]))
                  for x in range(1, 5, 2)] for y in range(1, 5, 2)],
                dtype=object)
        return OctreeBranch(np.stack([red_parts, green_octrees], axis=0))

    def generate_octree(self):
        return unwrap_progress_iterable(self.generate_octree_with_progress())

    def _generate_octree_leaf(self):
        assert self.level == 2
        leafs = flatten_quadtree_array(self.children)
        matrix = np.vectorize(lambda leaf: leaf.element, otypes=[object])(leafs)
        def neighbor_count(x, y):
            neighborhood = np.copy(matrix[y-1:y+2, x-1:x+2])
            neighborhood[1][1] = State.DEAD # don't count self
            return np.count_nonzero(neighborhood == State.ALIVE)
        new_states = [[matrix[y][x].next_state(neighbor_count(x, y)) for x in range(1, 3)] for y in range(1, 3)]
        quadtree = Quadtree.from_grid(new_states)
        return OctreeLeaf(quadtree)


class Octree(KdQuadtree):
    """
    An octree, representing how the grid evolves over time.

    Indexed by self.children[t][y][x], where t is time, and x and y are the grid
    coordinates. We call the t dimension "depth".

    The leaf elements of the octree are 2x2 quad trees containing states. Thus
    the octree has a cuboid shape, storing how a grid of 2n by 2n states evolves
    over n generations.
    """

    __slots__ = ()
    ndim = 3

    def width(self):
        return 2 * self.side_length()

    def depth(self):
        return self.side_length()

    def __str__(self):
        return "\n\n".join([str(quadtree) for quadtree in self.quadtrees()])

    @interned
    def most_recent_quadtree(self):
        return self.quadtrees().most_recent()


class OctreeLeaf(Octree, KdQuadtreeLeaf):
    __slots__ = ()
    @property
    def quadtree(self):
        return self.element

    def __post_init__(self):
        super().__post_init__()
        if self.quadtree.level != 1:
            raise ValueError(f"Quadtree must be level 1, got {self.quadtree.level}")

    def quadtrees(self):
        return BinaryTreeLeaf(self.quadtree)


class OctreeBranch(Octree, KdQuadtreeBranch):
    __slots__ = ('nw0', 'ne0', 'sw0', 'se0', 'nw1', 'ne1', 'sw1', 'se1', 'level')

    def __init__(self, children):
        assert children.shape == (2, 2, 2)
        self.nw0, self.ne0 = children[0, 0]
        self.sw0, self.se0 = children[0, 1]
        self.nw1, self.ne1 = children[1, 0]
        self.sw1, self.se1 = children[1, 1]
        self.level = self._checklevel()

    @property
    def children(self):
        return np.array([
            [[self.nw0, self.ne0], [self.sw0, self.se0]],
            [[self.nw1, self.ne1], [self.sw1, self.se1]]
        ], dtype=object)

    @interned
    def quadtrees(self):
        return BinaryTreeBranch(np.array([
            BinaryTree.compose(np.vectorize(lambda octree: octree.quadtrees(), otypes=[object])(self.children[t]))
            for t in range(2)
        ], dtype=object))


class BinaryTree(KdQuadtree):
    """
    This binary tree is the 1-dimensional equivalent to the quadtree (in 2D) or octree (in 3D).
    """

    __slots__ = ()
    ndim = 1

    def depth(self):
        return self.side_length()

    @classmethod
    @interned
    def compose(cls, quadrants: np.ndarray):
        """
        Compose a 2x2 array of binary trees into a single binary tree.
        """
        assert quadrants.shape == (2, 2)
        level = quadrants[0][0].level
        if level == 0:
            # Quadrants are leaves
            return BinaryTreeLeaf(QuadtreeBranch(np.vectorize(lambda leaf: leaf.element, otypes=[object])(quadrants)))
        else:
            return BinaryTreeBranch(np.array([
                BinaryTree.compose(np.vectorize(lambda binary_tree: binary_tree.children[t], otypes=[object])(quadrants))
                for t in range(2)
            ], dtype=object))

    @interned
    def population_quadtree(self):
        """
        Return a quadtree of the population of the binary tree summed across time.
        """
        if self.level == 0:
            return state_quadtree_to_population_quadtree(self.element)
        else:
            return add_population_quadtrees(self.children[0].population_quadtree(), self.children[1].population_quadtree())


class BinaryTreeLeaf(BinaryTree, KdQuadtreeLeaf):
    __slots__ = ()
    def __iter__(self):
        yield self.element

    def most_recent(self):
        return self.element


class BinaryTreeBranch(BinaryTree, KdQuadtreeBranch):
    __slots__ = ('first', 'last', 'level')

    def __init__(self, children):
        assert children.shape == (2,)
        self.first, self.last = children
        self.level = self._checklevel()

    @property
    def children(self):
        return np.array([self.first, self.last], dtype=object)

    def __iter__(self):
        yield from self.children[0]
        yield from self.children[1]

    @interned
    def most_recent(self):
        return self.children[1].most_recent()


def flatten_binary_tree_array(arr):
    """
    Flatten an array of n binary trees into one of 2n binary trees.
    """
    assert arr.ndim == 1
    assert arr[0].level >= 1
    length = arr.shape[0]
    flattened_arr = np.empty(2 * length, dtype=object)
    for i in range(length):
        flattened_arr[2*i:2*i+2] = arr[i].children
    return flattened_arr


def flatten_quadtree_array(arr):
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


@interned
def state_quadtree_to_population_quadtree(state_quadtree):
    """
    Given a quadtree of states, return a quadtree of populations. This is a
    simple cast to integer.
    """
    if state_quadtree.level == 0:
        return QuadtreeLeaf(int(state_quadtree.element))
    else:
        return QuadtreeBranch(np.vectorize(state_quadtree_to_population_quadtree, otypes=[object])(state_quadtree.children))


@interned
def add_population_quadtrees(tree1, tree2):
    """
    Given two quadtree of populations, return a quadtree of the sum of the
    populations.
    """
    if tree1.level == 0:
        return QuadtreeLeaf(add_population_counts(tree1.element, tree2.element))
    else:
        return QuadtreeBranch(np.vectorize(add_population_quadtrees, otypes=[object])(tree1.children, tree2.children))

def add_population_counts(i1, i2):
    if i1 == -1:
        assert i2 == -1
        return -1
    else:
        return i1 + i2
