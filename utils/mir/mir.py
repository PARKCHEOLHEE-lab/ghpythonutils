import copy
import math

import Rhino.Geometry as rg
import ghpythonlib.parallel

from ghpythonutils.utils.utils import (
    ConstsCollection,
    LineHelper,
    NumericHelper,
)


class MaximalInnerRectangleHelper:
    def _get_maximal_rectangle_of_binary_grid_indices(self, binary_grid):
        """Compute the maximum rectangle of a 2d binary grid

        Args:
            binary_grid (List[List[str]]): binary grid to compute the maximum rectangle indices

        Returns:
            Tuple[Tuple[int]]: top left indices and bottom right indices
        """

        if not binary_grid or not binary_grid[0]:
            return (), ()
            
        _, col_num = len(binary_grid), len(binary_grid[0])
        height = [0] * (col_num + 1)
        max_area = 0
        top_left = ()
        bottom_right = ()

        for ri, row in enumerate(binary_grid):
            for hi in range(col_num):
                height[hi] = (
                    height[hi] + 1
                    if row[hi] == ConstsCollection.BINARY_SOLID
                    else 0
                )

            stack = [-1]
            for ci in range(col_num + 1):
                while height[stack[-1]] > height[ci]:
                    hi = stack.pop()
                    h = height[hi]
                    w = ci - stack[-1] - 1

                    area = h * w
                    if max_area < area:
                        max_area = area

                        top_left = (ri - h + 1, stack[-1] + 1)
                        bottom_right = (ri, ci - 1)

                stack.append(ci)

        return top_left, bottom_right

    def _get_maximal_inner_rectangle(
        self, polygon, rotation_degree, grid_size, is_strict=False
    ):
        """Estimates the maximum inner rectangle of the given polygon by `rotation_degree` and `grid_size`

        Args:
            polygon (Rhino.Geometry.PolylineCurve): polygon to estimate
            rotation_degree (float): rotation step
            grid_size (float): size of the grid each cell
            is_strict (bool, optional): if true, only uses fully inner cells. Defaults to False.

        Returns:
            Rhino.Geometry.PolylineCurve: estimated maximal rectangle

        Reference:
            https://leetcode.com/problems/maximal-rectangle/solutions/3407011/ex-amazon-explains-a-solution-with-a-video-python-javascript-java-and-c/
            https://chat.openai.com/share/50607e72-da71-4938-8a4d-61d94b097ede
        """

        anchor = rg.AreaMassProperties.Compute(polygon).Centroid

        maximal_rectangle = None
        inverse_angle = 0

        for each_degree in range(0, 359, rotation_degree):
            each_angle = math.radians(each_degree)

            rotation_transform = rg.Transform.Rotation(each_angle, anchor)

            rotated_polygon = copy.copy(polygon)
            rotated_polygon.Transform(rotation_transform)

            grid = LineHelper.get_2d_grid_by_aabb(
                rotated_polygon, grid_size=grid_size, return_to_tree=False
            )

            binary_grid = NumericHelper.get_binary_grid(
                rotated_polygon, grid, is_centroid=not is_strict
            )

            # pylint: disable=unbalanced-tuple-unpacking
            (
                top_left,
                bottom_right,
            ) = self._get_maximal_rectangle_of_binary_grid_indices(binary_grid)
            # pylint: enable=unbalanced-tuple-unpacking

            row_range = range(top_left[0], bottom_right[0] + 1)
            col_range = range(top_left[1], bottom_right[1] + 1)

            maximal_rectangle_elements = []
            for ri in row_range:
                for ci in col_range:
                    maximal_rectangle_elements.append(grid[::-1][ri][ci])

            each_maximal_rectangle = rg.Curve.CreateBooleanUnion(
                maximal_rectangle_elements
            )[0]

            if maximal_rectangle is None:
                maximal_rectangle = each_maximal_rectangle

            largest_rectangle_area = rg.AreaMassProperties.Compute(
                maximal_rectangle
            ).Area

            each_rectangle_area = rg.AreaMassProperties.Compute(
                each_maximal_rectangle
            ).Area

            if each_rectangle_area > largest_rectangle_area:
                maximal_rectangle = each_maximal_rectangle
                inverse_angle = -each_angle

        if maximal_rectangle is not None:
            inverse_transform = rg.Transform.Rotation(inverse_angle, anchor)
            maximal_rectangle.Transform(inverse_transform)

        return maximal_rectangle
    
    def _generate_mir(
        self, polygon, anchor, rotation_angle, grid_size, is_strict
    ):

        grid = LineHelper.get_2d_grid_by_aabb(
            polygon, grid_size=grid_size, return_to_tree=False
        )

        binary_grid = NumericHelper.get_binary_grid(
            polygon, grid, is_centroid=not is_strict
        )

        # pylint: disable=unbalanced-tuple-unpacking
        (
            top_left,
            bottom_right,
        ) = self._get_maximal_rectangle_of_binary_grid_indices(binary_grid)
        # pylint: enable=unbalanced-tuple-unpacking

        row_range = range(top_left[0], bottom_right[0] + 1)
        col_range = range(top_left[1], bottom_right[1] + 1)

        maximal_rectangle_elements = []
        for ri in row_range:
            for ci in col_range:
                maximal_rectangle_elements.append(grid[::-1][ri][ci])

        maximal_rectangle = rg.Curve.CreateBooleanUnion(
            maximal_rectangle_elements
        )[0]
        
        inverse_transform = rg.Transform.Rotation(-rotation_angle, anchor)
        maximal_rectangle.Transform(inverse_transform)

        return maximal_rectangle
        
    

class MaximalInnerRectangle(MaximalInnerRectangleHelper):
    def __init__(self, polygon, rotation_degree, grid_size, is_strict=False):
        self.polygon = polygon
        self.rotation_degree = rotation_degree
        self.grid_size = grid_size
        self.is_strict = is_strict

        self._generate()

    def _generate(self):
        self.mir = self._get_maximal_inner_rectangle(
            self.polygon, self.rotation_degree, self.grid_size, self.is_strict
        )



def get_maximal_rectangle_of_binary_grid_indices(binary_grid):
    """Compute the maximum rectangle of a 2d binary grid

    Args:
        binary_grid (List[List[str]]): binary grid to compute the maximum rectangle indices

    Returns:
        Tuple[Tuple[int]]: top left indices and bottom right indices
    """

    if not binary_grid or not binary_grid[0]:
        return (), ()
        
    _, col_num = len(binary_grid), len(binary_grid[0])
    height = [0] * (col_num + 1)
    max_area = 0
    top_left = ()
    bottom_right = ()

    for ri, row in enumerate(binary_grid):
        for hi in range(col_num):
            height[hi] = (
                height[hi] + 1
                if row[hi] == ConstsCollection.BINARY_SOLID
                else 0
            )

        stack = [-1]
        for ci in range(col_num + 1):
            while height[stack[-1]] > height[ci]:
                hi = stack.pop()
                h = height[hi]
                w = ci - stack[-1] - 1

                area = h * w
                if max_area < area:
                    max_area = area

                    top_left = (ri - h + 1, stack[-1] + 1)
                    bottom_right = (ri, ci - 1)

            stack.append(ci)

    return top_left, bottom_right


    
def generate_mir(args):
    polygon, anchor, rotation_angle, grid_size, is_strict = args

    grid = LineHelper.get_2d_grid_by_aabb(
        polygon, grid_size=grid_size, return_to_tree=False
    )

    binary_grid = NumericHelper.get_binary_grid(
        polygon, grid, is_centroid=not is_strict
    )

    # pylint: disable=unbalanced-tuple-unpacking
    (
        top_left,
        bottom_right,
    ) = get_maximal_rectangle_of_binary_grid_indices(binary_grid)
    # pylint: enable=unbalanced-tuple-unpacking

    row_range = range(top_left[0], bottom_right[0] + 1)
    col_range = range(top_left[1], bottom_right[1] + 1)

    maximal_rectangle_elements = []
    for ri in row_range:
        for ci in col_range:
            maximal_rectangle_elements.append(grid[::-1][ri][ci])

    maximal_rectangle = rg.Curve.CreateBooleanUnion(
        maximal_rectangle_elements
    )[0]
    
    inverse_transform = rg.Transform.Rotation(-rotation_angle, anchor)
    maximal_rectangle.Transform(inverse_transform)

    return maximal_rectangle



if __name__ == "__main__":
    
    import time
    
    polygon = x
    
    rotation_degree = 5
    grid_size = 10
    is_strict = True
    
    anchor = rg.AreaMassProperties.Compute(polygon).Centroid
    rotation_polygons = []
    args_list = []
    
    for each_degree in range(0, 359, rotation_degree):
        each_args = []
        
        each_angle = math.radians(each_degree)

        rotation_transform = rg.Transform.Rotation(each_angle, anchor)

        rotated_polygon = copy.copy(polygon)
        rotated_polygon.Transform(rotation_transform)
        
        each_args = [rotated_polygon, anchor, each_angle, grid_size, is_strict]
        args_list.append(each_args)
    
    
    s = time.time()
    a = ghpythonlib.parallel.run(generate_mir, args_list, True)
    print(time.time() - s)  # 1.60836791992
    
    s = time.time()
    MaximalInnerRectangle(polygon, rotation_degree, grid_size, True).mir
    print(time.time() - s)  # 6.23183441162
    