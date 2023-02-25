import Rhino.Geometry as rg
import math


class ConstsCollection:
    TOLERANCE = 0.001


class Enum:
    @staticmethod
    def enum(*names):
        """Enumerated type creation helper
        Args:
            names: str
        Return:
            Enum
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
            linestring: Union[
                Rhino.Geometry.Curve, 
                Rhino.Geometry.PolylineCurve, 
                Rhino.Geometry.LineCurve,
                Rhino.Geometry.PolyCurve,
            ]
        Return:
            float
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
            n1: float
            n2: float
            tolerance: float
        Return:
            bool
        """
        
        return abs(n1 - n2) <= tolerance


class SurfaceHelper:
    @staticmethod
    def surface_reparameterize(srf):
        """Change domain to 0 ~ 1 given surface
            Args:
                srf: Rhino.Geometry.Surface
            Retrun:
                Rhino.Geometry.Surface
        """
        
        interval = rg.Interval(0, 1)
        srf.SetDomain(0, interval)
        srf.SetDomain(1, interval)
        
        return srf
