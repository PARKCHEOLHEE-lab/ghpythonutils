from scaler.minmaxscaler import MinMaxScaler
from utils.utils import NumericHelper

from scriptcontext import sticky as st
import Rhino.Geometry as rg
import Rhino

class TextHelper:
    
    custom_display = Rhino.Display.CustomDisplay(True)
    
    @staticmethod
    def get_text(
        string,
        height,
        string_place_origin=rg.Point3d(0, 0, 0), 
        string_place_plane=rg.Plane.WorldXY
    ):
        string_place_plane.Origin = origin
        string_object = Rhino.Display.Text3d(string, string_place_plane, height)
        
        
        custom_display.AddText(
            Rhino.Display.Text3d("test", plane, 2), 
            Rhino.Display.ColorHSL(0, 0, 0)
        )
        return


class ScorePolygon(MinMaxScaler, NumericHelper):
    """
    To use the inherited module, refer the link below.
    https://github.com/PARKCHEOLHEE-lab/GhPythonUtils
    """
    
    tolerance = 1e-3
    
    def __init__(self, origin, rad=5, scoredict={}):
        self.origin = origin
        self.rad = rad
        self.scoredict = scoredict
        self.score_length = len(self.scoredict.values())
        
        self.is_added_dummies = False
        if self.score_length <= 2:
            self.scoredict = self.get_scoredict_included_dummies(self.scoredict)
            self.score_length = len(self.scoredict.values())
            self.is_added_dummies = True

        MinMaxScaler.__init__(self)
        NumericHelper.__init__(self)
        
        self.circles = self.get_base_circles(self.origin, self.rad)
        self.outer_circle = self.circles[-1:]
        self.inner_circle = self.circles[:-1]
        
        self.main_polygon, self.main_polygon_points = self.get_polygon(
            self.outer_circle, self.score_length, is_return_points=True
        )
        
        self.inner_polygons = self.get_polygon(
            self.inner_circle, self.score_length
        )
        
        self.scorepolygon, self.scorepolygon_vertices = self.get_scorepolygon(
            self.origin, 
            self.rad, 
            self.main_polygon, 
            self.scoredict, 
            self.is_added_dummies
        )
        
    def get_scoredict_included_dummies(self, scoredict):
        needed_dummies_count = 3 - self.score_length
        for dummy_count in range(needed_dummies_count):
            scoredict["dummy_{}".format(dummy_count)] = self.tolerance
        return scoredict
        
    def get_base_circles(self, origin, rad):
        circle_count = 5
        sub_circle_rad = rad / circle_count
        
        circles = []
        for count in range(1, circle_count + 1):
            each_rad = sub_circle_rad * count
            circle = rg.Circle(origin, each_rad).ToNurbsCurve()
            circles.append(circle)
        
        return circles
        
    def get_polygon(self, circles, score_length, is_return_points=False):
        polygons = []
        all_divided_points = []
        for circle in circles:
            divided_score_length = circle.DivideByCount(score_length, True)
            divided_points = []
            
            for d_length in divided_score_length:
                divided_points.append(circle.PointAt(d_length))
            
            divided_points.append(circle.PointAtStart)
            all_divided_points.append(divided_points[:-1])
            polygon = rg.PolylineCurve(divided_points)
            polygons.append(polygon)
        
        if is_return_points:
            return polygons, all_divided_points
            
        return polygons
        
    def get_scorepolygon(
        self, origin, rad, main_polygon, scoredict, is_added_dummies
    ):
        score_values = scoredict.values()
        normalized_score_values = self.get_normalized_data(
            score_values, range_scale=rad
        )
        
        sorted_score_values = sorted(score_values)
        second_minimum_value = sorted(list(set(sorted_score_values)))[1]
        ratio = min(score_values) / second_minimum_value
        
        sorted_normalized_score_values = sorted(normalized_score_values)
        normalized_second_minimum_value = sorted(list(set(normalized_score_values)))[1]
        
        print(ratio, second_minimum_value)
        print(normalized_second_minimum_value)
        scorepolygon_vertices = []
        for segment, normalized_score_value in zip(
            main_polygon[0].DuplicateSegments(), normalized_score_values
        ):  
            
            if (
                self.is_close(0, normalized_score_value) 
                and not is_added_dummies
            ):
                normalized_score_value = normalized_second_minimum_value * ratio
            
            origin_to_each_vertex = rg.Line(origin, segment.PointAtStart)
            scorepolygon_vertices.append(
                origin_to_each_vertex.PointAtLength(normalized_score_value)
            )
        scorepolygon_vertices.append(scorepolygon_vertices[0])
        
        scorepolygon_curve = rg.PolylineCurve(scorepolygon_vertices)
        scorepolygon = rg.Brep.CreatePlanarBreps(scorepolygon_curve)
        
        if scorepolygon is None:
            return scorepolygon_curve, scorepolygon_vertices[:-1]
        
        return scorepolygon, scorepolygon_vertices[:-1]

if __name__ == "__main__":
    import random
    
    scoredict = {
        "1": -random.random(),
        "2": -random.random(),
        "3": random.random(),
        "4": random.random(),
        "5": random.random(),
        "6": random.random(),
    }
    
    scoredict = {
        "R2": -8.5,
        "R3": -2,
        "R1": -1,
        "R4": -3.7,
        "R5": -8.5,
    }
    
    scp = ScorePolygon(origin, rad=7, scoredict=scoredict)
    a, b, c, d = (
        scp.outer_circle, 
        scp.main_polygon, 
        scp.inner_polygons, 
        scp.scorepolygon
    )
    
    e = scp.main_polygon_points[0]
    f = scp.scoredict.keys()
    
    g = scp.scorepolygon_vertices
    h = scp.scoredict.values()
    
    print(len(g))
    
    if "custom_display" not in globals():
        custom_display = Rhino.Display.CustomDisplay(True)
    
    custom_display.Clear()
    
    if not textviz:
        custom_display.Dispose
        del custom_display
    
    else:
        plane = rg.Plane.WorldXY
        plane.Origin = origin
        text = Rhino.Display.Text3d("Test", plane, 1)
        color = Rhino.Display.ColorHSL(0, 0, 0)
        
        custom_display.AddText(text, color)

