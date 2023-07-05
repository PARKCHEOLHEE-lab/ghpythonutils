import copy
import math
import random
import re
import time

import Rhino.Display as rd
import Rhino.Geometry as rg
import rhinoscriptsyntax as rs
from ghpythonlib.treehelpers import list_to_tree

CUSTOM_DISPLAY = "custom_display"


def runtime_calculator(func):
    """This is a calculation function
    for the time taken as the shape of `decorator`
    """

    def runtime_calculator_wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        time_taken = end - start
        print("Run time: {}".format(time_taken))

        return result

    return runtime_calculator_wrapper


class ConstsCollection:
    TOLERANCE = 0.001

    HALF = 0.5
    INF = 1e15


class ColorsCollection:
    COLOR_BLACK = rd.ColorHSL(1, 0, 0, 0)
    COLOR_GRAY = rd.ColorHSL(0.5, 0, 0, 0.5)
    COLOR_RED = rd.ColorHSL(1, 0, 1, 0.5)
    COLOR_GREEN = rd.ColorHSL(1, 0.333, 1, 0.5)
    COLOR_BLUE = rd.ColorHSL(1, 0.666, 1, 0.5)

    @staticmethod
    def get_random_colors_per_each_branch(tree):
        """Random colors generator as much as the number of branches

        Args:
            tree (Grasshopper.DataTree): Data for coloring

        Returns:
            Grasshopper.DataTree: Colors tree, generated randomly
        """

        colors = []

        for branch in tree.Branches:
            r = random.randint(0, 150)
            g = random.randint(0, 150)
            b = random.randint(0, 150)

            colors.append([rs.CreateColor(r, g, b)] * len(branch))

        return list_to_tree(colors)


class Enum:
    @staticmethod
    def enum(*names):
        """Enumerated type creation helper

        Returns:
            Enum: custom enumerated type
        """

        enums = {}
        for ni, name in enumerate(names):
            enums[name] = ni

        return type("Enum", (), enums)


class LineHelper:
    @staticmethod
    def get_line_2d_angle(linestring, is_radians=True):
        """Calculate angle between two points

        Args:
            linestring (Rhino.Geometry.Curve): Segment with only two points
            is_radians (bool, optional): If False return to degree. Defaults to True.

        Returns:
            float: Angle of given linestring
        """

        x1, y1, _ = linestring.PointAtStart
        x2, y2, _ = linestring.PointAtEnd

        angle = math.atan2(y2 - y1, x2 - x1)
        if is_radians:
            return angle

        return math.degrees(angle)

    @staticmethod
    def get_2d_obb_from_line(linestring, geom):
        """Create an oriented bounding box by given axis of linestring

        Args:
            linestring (Rhino.Geometry.Curve): Alignment axis
            geom (Rhino.Geometry.Curve): Geometry to create the obb

        Returns:
            Rhino.Geometry.Rectangle3d: Oriented bounding box aligned to given line axis
        """
        angle = LineHelper.get_line_2d_angle(linestring)

        anchor = PointHelper.get_points_cloud_centroid(
            LineHelper.get_curve_vertices(linestring)
        )

        if linestring.IsClosed:
            anchor = geom.ToPolyline().CenterPoint()

        negative_transform = rg.Transform.Rotation(-angle, anchor)
        positive_transform = rg.Transform.Rotation(angle, anchor)

        copied_geom = copy.copy(geom)
        copied_geom.Transform(negative_transform)

        obb = copied_geom.GetBoundingBox(rg.Plane.WorldXY)
        obb = rg.Rectangle3d(rg.Plane.WorldXY, obb.Min, obb.Max)
        obb.Transform(positive_transform)

        return obb

    @staticmethod
    def get_sorted_segment(linestring, is_sort_by_longest=False):
        """Get sorted segments based on the length of each segment

        Args:
            linestring (Rhino.Geometry.Curve): To sort curve
            is_sort_by_longest (bool, optional): Sorting criteria. Defaults to False.

        Returns:
            List[Rhino.Geometry.Curve]: Sorted segments
        """

        exploded_linestring = linestring.DuplicateSegments()
        sorted_linestring = sorted(
            exploded_linestring,
            key=lambda l: l.GetLength(),
            reverse=is_sort_by_longest,
        )

        return sorted_linestring

    @staticmethod
    def get_2d_offset_polygon(linestring, distance, plane=rg.Plane.WorldXY):
        """Create a polygon through the given opened curve

        Args:
            linestring (Rhino.Geometry.Curve): Curve to make as a polygon
            distance (float): Distance to offset
            plane (Rhino.Geometry.Plane, optional): Offset plane
                                                    Defaults to WorldXY

        Raises:
            Exception: When closed curve

        Returns:
            Rhino.Geometry.Curve: Offset polygon
        """

        if linestring.IsClosed:
            raise Exception("Given linestring has been closed")

        offset_linestring_list = list(
            linestring.Offset(
                plane,
                distance,
                ConstsCollection.TOLERANCE,
                rg.CurveOffsetCornerStyle.Sharp,
            )
        )

        vertices = []
        exploded_linestring = linestring.DuplicateSegments()
        for li, each_line in enumerate(exploded_linestring):
            vertices.append(each_line.PointAtStart)
            if li == len(exploded_linestring) - 1:
                vertices.append(each_line.PointAtEnd)

        offset_vertices = []
        for offset_line in offset_linestring_list:
            exploded_offset_linestring = offset_line.DuplicateSegments()

            for oli, each_offset_line in enumerate(exploded_offset_linestring):
                offset_vertices.append(each_offset_line.PointAtStart)
                if oli == len(exploded_offset_linestring) - 1:
                    offset_vertices.append(each_offset_line.PointAtEnd)

        return rg.PolylineCurve(vertices + offset_vertices[::-1] + vertices[:1])

    @staticmethod
    def get_2d_buffered_linestring(linestring, distance, is_single_side=False):
        """Create a linestring has width

        Args:
            linestring (Rhino.Geometry.Curve): Target curve
            distance (float): Width

        Returns:
            List[Rhino.Geometry.Curve]: Buffered linestring
        """

        linestring = LineHelper.get_simplified_curve(linestring)
        exploded_linestring = []
        for l in linestring:
            exploded_linestring.extend(l.DuplicateSegments())

        first_line = exploded_linestring[0]

        is_plane_creation_succeed, section_plane = first_line.FrameAt(
            first_line.GetLength() + 1
        )

        if not is_plane_creation_succeed:
            angle = LineHelper.get_line_2d_angle(first_line)
            transform = rg.Transform.Rotation(angle, section_plane.Origin)

            section_plane.Transform(transform)
            section_plane.Origin = first_line.PointAtStart

        vector_1 = section_plane.YAxis * distance / 2
        vector_2 = -section_plane.YAxis * distance / 2

        if is_single_side:
            vector_1 = section_plane.YAxis * distance
            vector_2 = -section_plane.YAxis * 0

        for each_linestring in linestring:
            section = rg.Line(
                each_linestring.PointAtStart + vector_1,
                each_linestring.PointAtStart + vector_2,
            ).ToNurbsCurve()

            buffered_linestring = rg.Brep.CreateFromSweep(
                rail=each_linestring,
                shape=section,
                closed=True,
                tolerance=ConstsCollection.TOLERANCE,
            )

            buffered_linestring_outer = []
            for line in buffered_linestring:
                buffered_linestring_outer.extend(
                    list(line.DuplicateNakedEdgeCurves(True, False))
                )

        return list(rg.Curve.JoinCurves(buffered_linestring_outer))

    @staticmethod
    def get_curve_vertices(linestring):
        """Get vertices from given linestring

        Args:
            linestring (Rhino.Geometry.Curve): Given curve

        Returns:
            List[Rhino.Geometry.Point3d]: Vertices to curve
        """

        vertices = []
        exploded_linestring = linestring.DuplicateSegments()
        is_closed = linestring.IsClosed

        if len(exploded_linestring) == 1:
            vertices.extend(
                [
                    exploded_linestring[0].PointAtStart,
                    exploded_linestring[0].PointAtEnd,
                ]
            )

        else:
            for li, line in enumerate(exploded_linestring):
                vertices.append(line.PointAtStart)

                if li == len(exploded_linestring) - 1 and not is_closed:
                    vertices.append(line.PointAtEnd)

        return vertices

    @staticmethod
    def get_2d_convexhull(points):
        """Create 2d convex hull about given points

        Args:
            points (Rhino.Geometry.Point3d): Points to create convex hull

        Returns:
            Rhino.Geometry.Curve: 2d Convex hull
        """

        def cross(a, b, c):
            return (b.X - a.X) * (c.Y - a.Y) - (b.Y - a.Y) * (c.X - a.X)

        start = min(points, key=lambda p: (p.Y, p.X))
        sorted_points = sorted(
            points,
            key=lambda p: (math.atan2(p.Y - start.Y, p.X - start.X), p.X, p.Y),
        )
        hull = [start]
        for p in sorted_points:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
                hull.pop()

            hull.append(p)

        return rg.PolylineCurve(hull + [start])

    @staticmethod
    def get_2d_minimum_obb(geom):
        """Create the minimum oriented bounding box about given geometry

        Args:
            geom (Rhino.Geometry.Curve): Geometry to make minimum obb

        Returns:
            Rhino.geometry.Rectangle3d: Minimum obb
        """
        geom_vertices = LineHelper.get_curve_vertices(geom)
        geom_convexhull_segements = list(
            LineHelper.get_2d_convexhull(geom_vertices).DuplicateSegments()
        )

        obbs = []
        for hull_segment in geom_convexhull_segements:
            each_obb = LineHelper.get_2d_obb_from_line(hull_segment, geom)
            obbs.append(each_obb)

        return sorted(obbs, key=lambda b: b.Area)[0]

    @staticmethod
    def get_removed_overlapped_curves(curves, is_needed_for_points=False):
        """Remove overlapping curves

        Args:
            curves (List[Rhino.Geometry.Curve]): Target curves to remove duplicates

        Returns:
            List[Rhino.Geometry.Curve]: Curves with duplicates removed
        """

        result_curves = []
        for edge in curves:
            exploded_edge = edge.DuplicateSegments()
            result_curves.extend(exploded_edge)

        ei = 0
        while ei < len(result_curves):
            edge = result_curves[ei]
            edge_centroid = edge.PointAtLength(edge.GetLength() / 2)
            other_curves = result_curves[:ei] + result_curves[ei + 1 :]

            for other_edge in other_curves:
                other_edge_centroid = other_edge.PointAtLength(
                    other_edge.GetLength() / 2
                )

                intersections = rg.Intersect.Intersection.CurveCurve(
                    edge,
                    other_edge,
                    ConstsCollection.TOLERANCE,
                    ConstsCollection.TOLERANCE,
                )

                is_same_centroid = PointHelper.is_same_points(
                    edge_centroid, other_edge_centroid
                )

                for intsc in intersections:
                    if (
                        intsc.IsOverlap
                        and is_same_centroid
                        and edge in result_curves
                    ):
                        result_curves.remove(edge)
                        ei += -1

            ei += 1

        if is_needed_for_points:
            result_points = []
            for edge in result_curves:
                result_points.extend([edge.PointAtStart, edge.PointAtEnd])
            result_points = rg.Point3d.CullDuplicates(
                result_points, ConstsCollection.TOLERANCE
            )

            return result_curves, result_points

        return result_curves

    @staticmethod
    def get_simplified_curve(linestring):
        """Custom curve simplification method

        Args:
            linestring (Rhino.Geometry.Curve): Curve to simplify

        Returns:
            Rhino.Geometry.Curve: Simplified curve
        """

        exploded_input_poly = linestring.DuplicateSegments()

        simplified = [exploded_input_poly[0]]

        si = 0
        while si < len(exploded_input_poly):
            curr_segment = exploded_input_poly[si]
            prev_segment = simplified[-1]
            next_segment = simplified[0]

            is_last_si = si == len(exploded_input_poly) - 1

            curr_segment_angle = LineHelper.get_line_2d_angle(curr_segment)
            prev_segment_angle = LineHelper.get_line_2d_angle(prev_segment)
            next_segment_angle = LineHelper.get_line_2d_angle(next_segment)

            is_needed_merge_curr_and_prev = NumericHelper.is_close(
                curr_segment_angle, prev_segment_angle
            ) and PointHelper.is_same_points(
                prev_segment.PointAtEnd, curr_segment.PointAtStart
            )

            is_needed_merge_curr_and_next = (
                NumericHelper.is_close(curr_segment_angle, next_segment_angle)
                and PointHelper.is_same_points(
                    curr_segment.PointAtEnd, next_segment.PointAtStart
                )
                and is_last_si
            )

            if is_needed_merge_curr_and_prev and is_needed_merge_curr_and_next:
                vertex_1 = LineHelper.get_curve_vertices(simplified[-1])[0]
                vertex_2 = LineHelper.get_curve_vertices(simplified[0])[-1]
                simplified.append(rg.PolylineCurve([vertex_1, vertex_2]))

                del simplified[0]
                del simplified[-2]

            elif is_needed_merge_curr_and_prev:
                vertex_1 = LineHelper.get_curve_vertices(prev_segment)[0]
                vertex_2 = LineHelper.get_curve_vertices(curr_segment)[-1]
                simplified[-1] = rg.PolylineCurve([vertex_1, vertex_2])

            elif is_needed_merge_curr_and_next:
                vertex_1 = LineHelper.get_curve_vertices(curr_segment)[0]
                vertex_2 = LineHelper.get_curve_vertices(next_segment)[-1]
                simplified.append(rg.PolylineCurve([vertex_1, vertex_2]))

                del simplified[0]

            else:
                if si != 0:
                    simplified.append(curr_segment)

            si += 1

        return rg.Curve.JoinCurves(simplified)

    @staticmethod
    def get_curve_sublinestring(linestring, start, length):
        """Get the given linestring's substring

        Args:
            linestring (Rhino.Geometry.Curve): Linestring to get the substring
            start (float): Start location
            length (float): Length of Substring

        Returns:
            Rhino.Geometry.Curve: Sublinestring
        """

        splitted_curve = linestring.Split([float(start), start + float(length)])

        if start <= 0:
            return splitted_curve[0]

        return splitted_curve[1]

    @staticmethod
    def get_extended_linestring(linestring, start=0, end=0):
        """Get extended linestring with start and end length

        Args:
            linestring (Rhino.Geometry.Curve): Linestring to get the substring
            start (float): Start length to extend
            end (float): End length to extend

        Returns:
            Rhino.Geometry.Curve: Extended linestring
        """

        vector_to_extend = PointHelper.get_normalized_vector(
            linestring.PointAtEnd - linestring.PointAtStart
        )

        extended_start = vector_to_extend * -start + linestring.PointAtStart
        extended_end = vector_to_extend * end + linestring.PointAtEnd

        return rg.PolylineCurve([extended_start, extended_end])

    @staticmethod
    def get_2d_grid_by_aabb(
            axis_line, polygon, grid_size, return_to_tree=False
        ):
            
        if not polygon.IsClosed:
            raise Exception("A given polygon is opened")
        
        world_x_line = rg.LineCurve(
            rg.Point3d(0, 0, 0), rg.Point3d(1, 0, 0)
        )
        aabb = LineHelper.get_2d_obb_from_line(world_x_line, polygon)
        aabb = aabb.ToNurbsCurve()
        aabb_exploded = aabb.DuplicateSegments()
        
        x_seg, _, _, y_seg = aabb_exploded
        x_count = int(math.ceil(x_seg.GetLength() / grid_size))
        y_count = int(math.ceil(y_seg.GetLength() / grid_size))
        
        x_seg_vector = PointHelper.get_normalized_vector(
            x_seg.PointAtEnd - x_seg.PointAtStart
        )
        
        y_seg_vector = PointHelper.get_normalized_vector(
            y_seg.PointAtStart - y_seg.PointAtEnd
        )
        
        start_pt = x_seg.PointAtStart
        
        grid = []
        for yc in range(y_count):
            curr_y_vector = y_seg_vector * yc * grid_size
            
            grid_row = []
            for xc in range(x_count):
                curr_x_vector = x_seg_vector * xc * grid_size
                
                grid_p1 = start_pt + curr_y_vector + curr_x_vector
                grid_p2 = grid_p1 + x_seg_vector * grid_size
                grid_p3 = grid_p2 + y_seg_vector * grid_size
                grid_p4 = grid_p3 - x_seg_vector * grid_size
                
                cell = rg.PolylineCurve(
                    [grid_p1, grid_p2, grid_p3, grid_p4, grid_p1]
                )
                
                grid_row.append(cell)
            grid.append(grid_row)
        
        if return_to_tree:
            return list_to_tree(grid)
        
        return grid


class NumericHelper:
    @staticmethod
    def is_close(n1, n2, tolerance=ConstsCollection.TOLERANCE):
        """Check if two numbers are within margin of error

        Args:
            n1 (float): Number to compare
            n2 (float): Number to compare
            tolerance (float, optional): Permissible range. Defaults to ConstsCollection.TOLERANCE.

        Returns:
            bool: Compare result about whether two numbers are equal
        """

        return abs(n1 - n2) <= tolerance

    @staticmethod
    def all_close(nums, target, tolerance=ConstsCollection.TOLERANCE):
        """Check whether all numbers are the same as the target

        Args:
            nums (List[float]): Numbers to compare
            target (float): Target number to compare
            tolerance (float, optional): Permissible range. Defaults to ConstsCollection.TOLERANCE.

        Returns:
            bool: Compare result about whether all numbers are equal
        """

        return all(
            NumericHelper.is_close(num, target, tolerance) for num in nums
        )
        
    @staticmethod
    def get_binary_grid(polygon, grid, is_centroid=False):
        
        binary_void = "0"
        binary_solid = "1"
        
        binary_grid = []
        for grid_row in reversed(grid):
            binary_grid_row = []
            
            for cell in grid_row:
                if is_centroid:
                    cell_centroid = rg.AreaMassProperties.Compute(cell).Centroid
                    if polygon.Contains(cell_centroid) == rg.PointContainment.Inside:
                        binary_grid_row.append(binary_solid)
                    else:
                        binary_grid_row.append(binary_void)
                    
                else:
                    cell_vertices = LineHelper.get_curve_vertices(cell)
                    if all(
                        polygon.Contains(vertex) == rg.PointContainment.Inside 
                        for vertex in cell_vertices
                    ):
                        binary_grid_row.append(binary_solid)
                    else:
                        binary_grid_row.append(binary_void)
                        
            binary_grid.append(binary_grid_row)
        
        return binary_grid


class SurfaceHelper:
    @staticmethod
    def get_reparameterized_surface(srf):
        """Change domain to 0 ~ 1 given surface

        Args:
            srf (Rhino.Geometry.Surface): Surface to reparameterize

        Returns:
            Rhino.Geometry.Surface: Reparameterized surface
        """

        interval = rg.Interval(0, 1)

        copied_srf = copy.copy(srf)
        copied_srf.SetDomain(0, interval)
        copied_srf.SetDomain(1, interval)

        return copied_srf


class PointHelper:
    @staticmethod
    def get_points_cloud_centroid(points):
        """Find a centroid of given points.

        Args:
            points (List[Rhino.Geometry.Point3d]): Points cloud

        Returns:
            Rhino.Geometry.Point3d: Centroid of points cloud
        """

        return rg.Point3d(
            *[sum(coord_list) / len(points) for coord_list in zip(*points)]
        )

    @staticmethod
    def get_projected_point_on_curve(anchor, vector, geometry):
        """Calculate the projected point on the curve through the given point and vector

        Args:
            anchor (Rhino.Geometry.Point3d): Point to project
            vector (Rhino.Geometry.Vector3d): Projection direction
            geometry (Rhino.Geometry.Curve): Geometry to project

        Raises:
            Exception: When given geometry is not curve

        Returns:
            Rhino.Geometry.Point3d: Projected point. if not intersects, return None.
        """

        if not isinstance(geometry, rg.Curve):
            raise Exception("Unsupported geometry.")

        ray = rg.PolylineCurve([anchor, anchor + vector * ConstsCollection.INF])
        intersection = rg.Intersect.Intersection.CurveCurve(
            geometry,
            ray,
            ConstsCollection.TOLERANCE,
            ConstsCollection.TOLERANCE,
        )

        projected_point = None
        for intersect in intersection:
            projected_point = intersect.PointA
            break

        return projected_point

    @staticmethod
    def is_same_points(p1, p2):
        """Check whether p1 and p2 are the same

        Args:
            p1 (Rhino.Geometry.Point3d): Point to check
            p2 (Rhino.Geometry.Point3d): Point to check

        Returns:
            bool: Whether two points are the same or not the same
        """

        distance = p1.DistanceTo(p2)
        is_same_points = (
            distance <= 0
            or NumericHelper.is_close(distance, 0)
            or p1.Equals(p2)
        )

        return is_same_points

    @staticmethod
    def get_normalized_vector(vector):
        """Normalize to vector that length 1.0

        Args:
            vector (Rhino.Geometry.Vector3d): Vector to normalize

        Returns:
            Rhino.Geometry.Vector3d: Normalized vector
        """

        return vector / vector.Length


class VisualizeHelper:
    """
    NOTE: If you RUN(F5) in ghpython scripting window, the text may not remove.
    """

    @classmethod
    def __initialize(cls):
        if CUSTOM_DISPLAY not in globals():
            globals()[CUSTOM_DISPLAY] = rd.CustomDisplay(True)

    @classmethod
    def __dispose(cls):
        globals()[CUSTOM_DISPLAY].Dispose()
        del globals()[CUSTOM_DISPLAY]

    @staticmethod
    def visualize_text(
        string,
        height,
        toggle,
        color=ColorsCollection.COLOR_BLACK,
        string_place_origin=rg.Point3d(0, 0, 0),
        string_place_plane=rg.Plane.WorldXY,
    ):
        """Text visualization helper

        Args:
            string (str): Text to visualize
            height (float): Text size
            toggle (bool): Whether text visualization ON or OFF
            color (Rhino.Display.ColorHSL, optional): Text color. Defaults to ColorHelper.COLOR_BLACK.
            string_place_origin (Rhino.Geometry.Point3d, optional): Location of text. Defaults to rg.Point3d(0, 0, 0).
            string_place_plane (Rhino.Geometry.Plane, optional): Text direction. Defaults to rg.Plane.WorldXY.
        """

        VisualizeHelper.__initialize()
        if not toggle:
            VisualizeHelper.__dispose()
        else:
            string_place_plane.Origin = string_place_origin
            text_3d = rd.Text3d(string, string_place_plane, height)
            globals()[CUSTOM_DISPLAY].AddText(text_3d, color)

    @staticmethod
    def visualize_curve(
        curve, toggle, color=ColorsCollection.COLOR_BLACK, thickness=1
    ):
        """Curve Visualization helper

        Args:
            curve (Rhino.Geometry.Curve): Curve to visualize
            toggle (bool): Whether curve visualization ON or OFF
            color (Rhino.Display.ColorHSL, optional): Curve color. Defaults to ColorHelper.COLOR_BLACK.
            thickness (int, optional): Curve display thickness. Defaults to 1.
        """

        VisualizeHelper.__initialize()
        if not toggle:
            VisualizeHelper.__dispose()
        else:
            globals()[CUSTOM_DISPLAY].AddCurve(curve, color, thickness)

    @staticmethod
    def visualize_polygon(
        polygon,
        toggle,
        fill_color=ColorsCollection.COLOR_GRAY,
        edge_color=ColorsCollection.COLOR_BLACK,
        draw_fill=True,
        draw_edge=True,
    ):
        """Polygon visualization helper

        Args:
            polygon (List[Point3d]): Polygon to visualize
            toggle (bool): Whether polygon visualization ON or OFF
            fill_color (Rhino.Display.ColorHSL, optional): Polygon's inner color. Defaults to ColorHelper.COLOR_GRAY.
            edge_color (Rhino.Display.ColorHSL, optional): Polygon's edge color. Defaults to ColorHelper.COLOR_BLACK.
            draw_fill (bool, optional): Whether polygon inside fill. Defaults to True.
            draw_edge (bool, optional): Whether polygon edge visualization. Defaults to True.

        NOTE:
            This function is incomplete. Sometimes the concave part of the polygon is convexly filled
        """

        VisualizeHelper.__initialize()
        if not toggle:
            VisualizeHelper.__dispose()
        else:
            globals()[CUSTOM_DISPLAY].AddPolygon(
                polygon, fill_color, edge_color, draw_fill, draw_edge
            )


class StringHelper:
    @staticmethod
    def get_geometries_from_wkt(wkt):
        """Parses WKT and converts to Rhino geometries
           https://chat.openai.com/share/fbc9a714-8711-4854-8c4e-65cea90aff76

        Args:
            wkt (str): Well-known text representation of geometry

        Returns:
            Rhino.Geometry.Polyline: Converted geometry to Polyline

        TODO:
            `POINT`, `LINESTRING` handling. Handles only `POLYGON` currently
        """

        geometry_list = []

        wkt_geometries = wkt.split(";")

        for wkt_geometry in wkt_geometries:
            match = re.match(r"(\w+)\s*\((.*)\)", wkt_geometry)

            if match:
                geometry_type = match.group(1).upper()
                coordinates = match.group(2)

                if geometry_type == "GEOMETRYCOLLECTION":
                    coordinates = coordinates[:-1]

                    sub_geometries = coordinates.split("POLYGON")[1:]

                    sub_geometries_wkt = []
                    for wi, sw in enumerate(sub_geometries):
                        sub_wkt = "POLYGON" + sw

                        if wi == len(sub_geometries) - 1:
                            sub_wkt = sub_wkt + ")"
                        else:
                            sub_wkt = sub_wkt[:-2]

                        sub_geometries_wkt.append(sub_wkt)

                    sub_geometry_list = [
                        StringHelper.get_geometries_from_wkt(sw)[0]
                        for sw in sub_geometries_wkt
                    ]

                    geometry_list.extend(sub_geometry_list)

                elif geometry_type == "POLYGON":
                    coordinates = coordinates[:-1]

                    rings = coordinates.split("),")

                    polygon_vertices = []

                    for ring in rings:
                        ring = ring[1:]

                        if ring.endswith(")"):
                            ring = ring[:-1]

                        vertices = [
                            map(float, coord.split())
                            for coord in ring.split(",")
                        ]

                        polygon_vertices.extend(
                            [rg.Point3d(x, y, 0) for x, y in vertices]
                        )

                    polyline = rg.Polyline(polygon_vertices)
                    geometry_list.append(polyline)

        return geometry_list


grid = LineHelper.get_2d_grid_by_aabb(
    y,
    x,
    grid_size=1.1,
    return_to_tree=False
)

a = LineHelper.get_2d_grid_by_aabb(
    y,
    x,
    grid_size=1.1,
    return_to_tree=True
)

print(NumericHelper.get_binary_grid(x, grid, is_centroid=False))


