import Rhino.Geometry as rg
import Rhino.Display as rd
from scaler.minmaxscaler import MinMaxScaler
from utils.utils import NumericHelper, VisualizeHelper, ColorHelper



class ScorePolygon(MinMaxScaler, NumericHelper, VisualizeHelper, ColorHelper):
    """
    To use the inherited module, refer the link below.
    https://github.com/PARKCHEOLHEE-lab/GhPythonUtils
    """
    
    tolerance = 1e-3
    circle_count = 6
    
    def __init__(
        self, 
        origin, 
        rad=5, 
        scoredict={}, 
        toggle=False, 
        color=rd.ColorHSL(0.7, 0.333, 1, 0.5)
    ):
        """Score polygon to visualize given scores of Each element

        Args:
            origin (Rhino.Geometry.Point3d): Origin of Score polygon
            rad (int, optional): Size of the score polygon. Defaults to 5.
            scoredict (dict, optional): Dictionary to visualize. Defaults to {}.
            toggle (bool, optional): Whether visualization. Defaults to False.
            color (Rhino.Display.ColorHSL, optional): Score polygon color. Defaults to rd.ColorHSL(0.7, 0.333, 1, 0.5).
        """

        self.origin = origin
        self.rad = rad
        self.scoredict = scoredict
        self.score_length = len(self.scoredict.values())
        self.toggle = toggle
        self.text_size = rad * 0.05
        self.color = color
        
        if self.score_length <= 2:
            self.scoredict = self.get_scoredict_included_dummies(self.scoredict)
            self.score_length = len(self.scoredict.values())

        MinMaxScaler.__init__(self)
        NumericHelper.__init__(self)
        VisualizeHelper.__init__(self)
        ColorHelper.__init__(self)
        
        self.sub_circle_rad = self.rad / self.circle_count
        self.circles = self.get_base_circles(self.origin, self.sub_circle_rad)
        self.outer_circle = self.circles[-1:]
        self.inner_circle = self.circles[:-1]
        
        self.main_polygon, self.main_polygon_points = self.get_polygon(
            self.outer_circle, self.score_length, is_return_points=True
        )
        
        self.inner_polygons = self.get_polygon(
            self.inner_circle, self.score_length
        )
        
        (
            self.scorepolygon, 
            self.scorepolygon_vertices, 
            self.sublines
        ) = self.get_scorepolygon(
            self.origin, 
            self.rad, 
            self.main_polygon[0], 
            self.scoredict, 
            self.sub_circle_rad,
        )
        
        self.visualize(
            self.scoredict, 
            self.scorepolygon,
            self.scorepolygon_vertices, 
            self.main_polygon_points[0], 
            self.main_polygon + self.inner_polygons,
            self.sublines,
            self.outer_circle[0],
            self.toggle,
            self.text_size,
            self.color,
        )
        
    def get_scoredict_included_dummies(self, scoredict):
        """Create dummy elements when length of `scoredict` less than 3 

        Args:
            scoredict (dict): Dictionary to visualize.

        Returns:
            dict: Dictionary to visualize, included dummies.
        """

        needed_dummies_count = 3 - self.score_length
        for dummy_count in range(needed_dummies_count):
            scoredict["dummy_{}".format(dummy_count)] = self.tolerance
        return scoredict
        
    def get_base_circles(self, origin, sub_circle_rad):
        """Create circles to make score polygons

        Args:
            origin (Rhino.Geometry.Point3d): Origin of Score polygon
            sub_circle_rad (float): Interval to make circles 

        Returns:
            List[Rhino.Geometry.Circle]: Base circles for making score polygons
        """

        circles = []
        for count in range(1, self.circle_count + 1):
            each_rad = sub_circle_rad * count
            circle = rg.Circle(origin, each_rad).ToNurbsCurve()
            circles.append(circle)
        
        return circles
        
    def get_polygon(self, circles, score_length, is_return_points=False):
        """Create polygon via separated points of given circles

        Args:
            circles (List[Rhino.Geometry.Circle]): Circles to make polygon
            score_length (int): length of values of `scoredict`
            is_return_points (bool, optional): Whether return polygon vertices. Defaults to False.

        Returns:
            Rhino.Geometry.PolylineCurve: Polygon created
        """
        
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
        self, 
        origin, 
        rad, 
        main_polygon, 
        scoredict, 
        sub_circle_rad
    ):
        """Create score polygon

        Args:
            origin (Rhino.Geometry.Point3d): Origin of Score polygon
            rad (int): Size of the score polygon.
            main_polygon (Rhino.Geometry.PolylineCurve): Most outer polygon
            scoredict (dict): Dictionary to visualize
            sub_circle_rad (float): Interval to make circles 

        Returns:
            Tuple[
                Rhino.Geometry.Brep,
                List[Rhino.Geometry.Point3d],
                List[Rhino.Geometry.Line]
            ]: For visualization data
        """
        
        score_values = scoredict.values()
        normalized_score_values = self.get_normalized_data(
            score_values, range_scale=rad - sub_circle_rad * 2
        )
        
        sublines = []
        scorepolygon_vertices = []
        for segment, normalized_score_value in zip(
            main_polygon.DuplicateSegments(), normalized_score_values
        ):  
            
            origin_to_each_vertex_curve = rg.Line(
                origin, segment.PointAtStart
            ).ToNurbsCurve()
            
            origin_to_each_vertex_curve_subline = rg.Line(
                origin_to_each_vertex_curve.PointAtLength(sub_circle_rad),
                origin_to_each_vertex_curve.PointAtLength(
                    origin_to_each_vertex_curve.GetLength() - sub_circle_rad
                )
            )
            
            sublines.append(origin_to_each_vertex_curve_subline)
            
            scorepolygon_vertices.append(
                origin_to_each_vertex_curve_subline.PointAtLength(
                    normalized_score_value
                )
            )
            
        scorepolygon_vertices.append(scorepolygon_vertices[0])
        scorepolygon_curve = rg.PolylineCurve(scorepolygon_vertices)
        scorepolygon = rg.Brep.CreatePlanarBreps(scorepolygon_curve)[0]
        
        if scorepolygon is None:
            return scorepolygon_curve, scorepolygon_vertices
        
        return scorepolygon, scorepolygon_vertices, sublines
        
    def visualize(
        self, 
        scoredict, 
        scorepolygon,
        scorepolygon_vertices, 
        main_polygon_points, 
        polygons,
        sublines,
        outer_circle,
        toggle, 
        text_size,
        color,
    ):
        """Visualization function

        Args:
            scoredict (dict): Dictionary to visualize.
            scorepolygon (Rhino.Geometry.Brep): Score polygon
            scorepolygon_vertices (List[Rhino.Geometry.Point3d]): Vertices of score polygon
            main_polygon_points (List[Rhino.Geometry.Point3d]): Vertices of Most outer polygon
            polygons (List[Rhino.Geometry.PolylineCurve]): All polygons
            sublines (List[Rhino.Geometry.Line]): Lines that connects the origin and each vertex
            outer_circle (Rhino.Geometry.Circle): Most outer circle
            toggle (bool): Whether visualization
            text_size (float): Text size
            color (Rhino.Display.ColorHSL): Score polygon color
        """
        
        for edge in scorepolygon.Edges:
            self.visualize_curve(
                edge.ToNurbsCurve(), 
                toggle, 
                color=color,
                thickness=5
            )
        
#        self.visualize_polygon(
#            scorepolygon_vertices, toggle, fill_color=color, draw_edge=True
#        )
        
        for curve in sublines + polygons + [outer_circle]:
            if isinstance(curve, rg.Line):
                curve = curve.ToNurbsCurve()
                
            self.visualize_curve(
                curve.ToNurbsCurve(), toggle, color=self.COLOR_GRAY
            )

        for key, main_polygon_point in zip(
            scoredict.keys(), main_polygon_points
        ):
            self.visualize_text(
                key, 
                text_size * 1.5, 
                toggle, 
                color=self.COLOR_GRAY,
                string_place_origin=main_polygon_point
            )
    
        for value, scorepolygon_vertex in zip(
            scoredict.values(), scorepolygon_vertices
        ):
            self.visualize_text(
                str(value), 
                text_size, 
                toggle, 
                color=self.COLOR_BLACK, 
                string_place_origin=scorepolygon_vertex
            )