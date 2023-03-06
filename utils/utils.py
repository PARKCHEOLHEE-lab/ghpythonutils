import rhinoscriptsyntax as rs
import Rhino.Geometry as rg
import Rhino.Display as rd
import math
import copy

import ghpythonlib.components as gh

CUSTOM_DISPLAY = "custom_display"


class ConstsCollection:
    TOLERANCE = 0.001


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


class SurfaceHelper:
    @staticmethod
    def surface_reparameterize(srf):
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


class ColorHelper:
    COLOR_BLACK = rd.ColorHSL(1, 0, 0, 0)
    COLOR_GRAY = rd.ColorHSL(0.5, 0, 0, 0.5)
    COLOR_RED = rd.ColorHSL(1, 0, 1, 0.5)
    COLOR_GREEN = rd.ColorHSL(1, 0.333, 1, 0.5)
    COLOR_BLUE = rd.ColorHSL(1, 0.666, 1, 0.5)


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
        color=ColorHelper.COLOR_BLACK,
        string_place_origin=rg.Point3d(0, 0, 0), 
        string_place_plane=rg.Plane.WorldXY
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
        curve, toggle, color=ColorHelper.COLOR_BLACK, thickness=1
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
        fill_color=ColorHelper.COLOR_GRAY, 
        edge_color=ColorHelper.COLOR_BLACK, 
        draw_fill=True,
        draw_edge=True,
    ):
        """Polygon visualization helper

        Args:
            polygon (List[Point3d]): Polygon to visualize
            toggle (bool): Whether polygon visualization ON or OFF
            fill_color (Rhino.Display.ColorHSL, optional): _description_. Defaults to ColorHelper.COLOR_GRAY.
            edge_color (Rhino.Display.ColorHSL, optional): _description_. Defaults to ColorHelper.COLOR_BLACK.
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


if __name__ == "__main__":
    a = LineHelper.get_2d_obb_from_line(x, y)