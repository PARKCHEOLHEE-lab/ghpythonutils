import Rhino.Geometry as rg
import Rhino.Display as rd
import math


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
            linestring (
                Union[
                    Rhino.Geometry.Curve, 
                    Rhino.Geometry.PolylineCurve, 
                    Rhino.Geometry.LineCurve,
                    Rhino.Geometry.PolyCurve,
                ]
            ): Segment with only two points
            is_radians (bool, optional): If False return to degree. Defaults to True.

        Returns:
            float: Angle of given linestring
        """
                
        x1, y1, _ = linestring.PointAtStart
        x2, y2, _ = linestring.PointAtEnd
        
        angle = math.atan2(y2 - y1, x2 - x1)
        if is_radians:
            return angle
        
        return math.degrees(self.sun_facing_angle)


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
        srf.SetDomain(0, interval)
        srf.SetDomain(1, interval)
        
        return srf


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


class TextHelper:
    def __init__(
        self, 
        string,
        height,
        string_place_origin=rg.Point3d(0, 0, 0), 
        string_place_plane=rg.Plane.WorldXY
    ):  
        self.string = string
        self.height = height
        self.string_place_origin = string_place_origin
        self.string_place_plane = string_place_plane
        
        if "custom_display" not in globals():
            self.custom_display = rd.CustomDisplay(True)
        
    def visualize_text(self, toggle):
        
        if not toggle:
            self.custom_display.Dispose()
        
        else:
            self.string_place_plane.Origin = self.string_place_origin
            text_3d = rd.Text3d(self.string, self.string_place_plane, self.height)
            text_color = rd.ColorHSL(0, 0, 0)
            self.custom_display.AddText(text_3d, text_color)
            

text_helper = TextHelper(
    string="test", height=2, string_place_origin=rg.Point3d(0,20,0)
)

text_helper.visualize_text(x)

a = 1
asdf = 123123
print(globals()["a"], globals()["asdf"], globals()["text_helper"].custom_display)
print(vars())