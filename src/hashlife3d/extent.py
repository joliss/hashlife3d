from dataclasses import dataclass

from ranges import Range


@dataclass
class Point2D:
    """
    A point in the infinite plane.
    """
    x: int | float
    y: int | float


@dataclass
class Point3D:
    """
    A point in the infinite 3D space.
    """
    x: int | float
    y: int | float
    t: int | float


@dataclass
class RectangleExtent:
    """
    A rectangle region of the infinite plane.
    """
    x_range: Range
    y_range: Range

    @property
    def width(self):
        return self.x_range.end - self.x_range.start

    @property
    def height(self):
        return self.y_range.end - self.y_range.start

    def __contains__(self, other):
        if isinstance(other, Point2D):
            return other.x in self.x_range and other.y in self.y_range
        return other.x_range in self.x_range and other.y_range in self.y_range

    def intersects(self, other):
        return bool(self.x_range.intersection(other.x_range) and self.y_range.intersection(other.y_range))

    def split_x_y(self):
        """
        Return two pairs of rectangles with half the width and height.
        """
        assert self.width % 2 == 0
        assert self.height % 2 == 0
        midpoint_x = self.x_range.start + self.width // 2
        midpoint_y = self.y_range.start + self.height // 2
        return (
            (
                RectangleExtent(Range(self.x_range.start, midpoint_x), Range(self.y_range.start, midpoint_y)),
                RectangleExtent(Range(midpoint_x, self.x_range.end), Range(self.y_range.start, midpoint_y))
            ),
            (
                RectangleExtent(Range(self.x_range.start, midpoint_x), Range(midpoint_y, self.y_range.end)),
                RectangleExtent(Range(midpoint_x, self.x_range.end), Range(midpoint_y, self.y_range.end))
            )
        )


@dataclass
class CuboidExtent:
    """
    A cuboid region of the infinite 3D space.
    """
    x_range: Range
    y_range: Range
    t_range: Range

    @property
    def width(self):
        return self.x_range.end - self.x_range.start

    @property
    def height(self):
        return self.y_range.end - self.y_range.start

    @property
    def depth(self):
        return self.t_range.end - self.t_range.start

    def __contains__(self, other):
        if isinstance(other, Point3D):
            return other.x in self.x_range and other.y in self.y_range and other.t in self.t_range
        return other.x_range in self.x_range and other.y_range in self.y_range and other.t_range in self.t_range

    def base(self):
        """
        Return the rectangular base of this cuboid.
        """
        return RectangleExtent(self.x_range, self.y_range)

    def split_t(self):
        """
        Return a pair of cuboids with half the depth.
        """
        assert self.depth % 2 == 0
        midpoint = self.t_range.start + self.depth // 2
        return (
            CuboidExtent(self.x_range, self.y_range, Range(self.t_range.start, midpoint)),
            CuboidExtent(self.x_range, self.y_range, Range(midpoint, self.t_range.end))
        )
