import Rhino.Geometry as rg
import Rhino.Display as rd
import math
import copy


CUSTOM_DISPLAY = "custom_display"


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
            Rhino.Geometry.Curve: Oriented bounding box aligned to given line axis
        """
        angle = LineHelper.get_line_2d_angle(linestring)
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

        linestring = linestring.Simplify(
            rg.CurveSimplifyOptions.Merge,
            ConstsCollection.TOLERANCE,
            ConstsCollection.TOLERANCE,
        )

        exploded_linestring = linestring.DuplicateSegments()
        first_line = exploded_linestring[0]

        is_plane_creation_succeed, section_plane = first_line.FrameAt(0)
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

        section = rg.Line(
            linestring.PointAtStart + vector_1,
            linestring.PointAtStart + vector_2,
        ).ToNurbsCurve()

        buffered_linestring = rg.Brep.CreateFromSweep(
            rail=linestring,
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
