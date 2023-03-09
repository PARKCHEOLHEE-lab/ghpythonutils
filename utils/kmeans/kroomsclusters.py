from utils.kmeans.kmeans import KMeans
from utils.utils import LineHelper
import Rhino.Geometry as rg
import copy


class Room:
    """
    _description_
    """
    
    
    def __init__(self, boundary, path):
        self.boundary = boundary
        self.path = path


class Boundary:
    """
    _description_
    """
    
    def __init__(self, boundary, target_area):
        self.boundary = boundary
        self.boundary_area = rg.AreaMassProperties.Compute(boundary).Area
        self.target_area = target_area
        
        self._gen_estimated_k()
        
    def _gen_estimated_k(self):
        """The K is given floor area divided target area."""
        self.k = int(self.boundary_area / self.target_area)
        
    def _gen_shortest_path(self):
        pass


class KRoomsCluster(KMeans, LineHelper):
    """
    To use the inherited moduels, refer the link below.
    https://github.com/PARKCHEOLHEE-lab/GhPythonUtils
    """
    
    def __init__(
        self, floor, core, hall, target_area, axis=None, grid_size=None
    ):
        self.floor = floor
        self.core = core
        self.hall = hall
        self.sorted_hall_segments = self.get_sorted_segment(self.hall)
        self.target_area = target_area
        self.axis = axis
        self.grid_size = grid_size
        
        KMeans.__init__(self)
        LineHelper.__init__(self)
        
    def predict(self):
        self._gen_given_axis_aligned_obb()
        self._gen_boundaries()
        self._gen_estimated_grid_size()
        self._gen_grid()
        self._gen_grid_2()
        
    def _gen_boundaries(self):
        boundaries = rg.Curve.CreateBooleanDifference(
            self.floor, self.core
        )
        
        self.boundaries = []
        for boundary in boundaries:
            boundary_object = Boundary(boundary, self.target_area)
            self.boundaries.append(boundary_object)
            
    def _gen_given_axis_aligned_obb(self):
        if self.axis is None:
            self.axis = self.hall.DuplicateSegments()[0]
            
        self.obb = self.get_2d_obb_from_line(
            self.axis, self.floor
        ).ToNurbsCurve()
        
        if (
            self.obb.ClosedCurveOrientation() 
            != rg.CurveOrientation.CounterClockwise
        ):
           self.obb.Reverse()
        
    def _gen_estimated_grid_size(self):
        if self.grid_size is None:
            hall_shortest_segment = self.sorted_hall_segments[0]
            self.grid_size = hall_shortest_segment.GetLength()
        
    def _gen_grid(self):
        self.x_segment, self.y_segment = self.obb.DuplicateSegments()[:2]
        
        x_segment_divided_length = self.x_segment.DivideByLength(
            self.grid_size, includeEnds=True
        )
        
        y_count = int(self.y_segment.GetLength() / self.grid_size) + 1
        y_vector = (
            self.y_segment.PointAtLength(self.grid_size) 
            - self.y_segment.PointAtStart
        )
        
        self.grid = []
        for xi, x_length in enumerate(x_segment_divided_length):
            next_xi = (xi + 1) % len(x_segment_divided_length)
            next_x_length = x_segment_divided_length[next_xi]
            
            start_point = self.x_segment.PointAt(x_length)
            next_point = self.x_segment.PointAt(next_x_length)
            
            if xi == len(x_segment_divided_length) - 1:
                next_point = self.x_segment.PointAtEnd
                
            offset_start_point = start_point + y_vector
            offset_next_point = next_point + y_vector
                
            rectangle = rg.PolylineCurve(
                [
                    start_point, 
                    next_point, 
                    offset_next_point, 
                    offset_start_point, 
                    start_point
                ]
            )
            
            for y in range(y_count):
                copied_rectangle = copy.copy(rectangle)
                copied_rectangle.Translate(y_vector * y)
                
                cleanup_rectangles = []
                for boundary in self.boundaries:
                    clean_up_rectangle = list(
                        rg.Curve.CreateBooleanIntersection(
                            boundary.boundary, copied_rectangle.ToNurbsCurve()
                        )
                    )
                    
                    cleanup_rectangles.extend(clean_up_rectangle)
                    
                self.grid.extend(cleanup_rectangles)
        
    def _gen_grid_2(self):
        self.initial_rectangle = self.get_2d_offset_polygon(
            self.sorted_hall_segments[0], self.grid_size
        )
        

if __name__ == "__main__":
    krooms = KRoomsCluster(
        floor=floor,
        core=core,
        hall=hall,
        target_area=40,
        grid_size=None
    )
    
    krooms.predict()

    a = krooms.initial_rectangle
    b = [b.boundary for b in krooms.boundaries]