from scaler.minmaxscaler import MinMaxScaler
from utils.utils import NumericHelper
import Rhino.Geometry as rg


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
        
        self.main_polygon = self.get_polygon(
            self.outer_circle, self.score_length
        )
        
        self.inner_polygons = self.get_polygon(
            self.inner_circle, self.score_length
        )
        
        self.scorepolygon = self.get_scorepolygon(
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
        
    def get_polygon(self, circles, score_length):
        polygons = []
        for circle in circles:
            divided_score_length = circle.DivideByCount(score_length, True)
            divided_points = []
            
            for d_length in divided_score_length:
                divided_points.append(circle.PointAt(d_length))
            
            divided_points.append(circle.PointAtStart)
            polygon = rg.PolylineCurve(divided_points)
            polygons.append(polygon)
        
        return polygons
        
    def get_scorepolygon(
        self, origin, rad, main_polygon, scoredict, is_added_dummies
    ):
        score_values = scoredict.values()
        normalized_score_values = self.get_normalized_data(
            score_values, range_scale=rad
        )
        
        print(normalized_score_values)
        
        sorted_score_values = sorted(score_values)
        second_minimum_value = sorted(list(set(sorted_score_values)))[1]
        ratio = max(min(score_values) / second_minimum_value, 0)
        
        sorted_normalized_score_values = sorted(normalized_score_values)
        normalized_second_minimum_value = sorted(list(set(normalized_score_values)))[1]
        
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
            return scorepolygon_curve
        
        return scorepolygon

if __name__ == "__main__":
    import random
    
#    scoredict = {
#        "1": random.random(),
#        "2": random.random(),
#        "3": random.random(),
#        "4": random.random(),
#        "5": random.random(),
#        "6": random.random(),
#    }
    
    scoredict = {
        "2": -3,
        "3": -2,
        "1": -1,
        "4": -3.7,
        "4": -8.5,
    }
    
    scp = ScorePolygon(origin, rad=7, scoredict=scoredict)
    a, b, c, d = (
        scp.outer_circle, 
        scp.main_polygon, 
        scp.inner_polygons, 
        scp.scorepolygon
    )
    
