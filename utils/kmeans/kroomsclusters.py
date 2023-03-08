from utils.kmeans.kmeans import KMeans
from utils.utils import LineHelper
import Rhino.Geometry as rg
import copy

class KRoomsCluster(KMeans, LineHelper):
    """
    To use the inherited moduels, refer the link below.
    https://github.com/PARKCHEOLHEE-lab/GhPythonUtils
    """
    
    def __init__(self, floor, core, hall, target_area, axis=None):
        self.floor = floor
        self.core = core
        self.hall = hall
        self.target_area = target_area
        self.axis = axis
        
        KMeans.__init__(self)
        LineHelper.__init__(self)
        
    def predict(self):
        self._gen_given_axis_aligned_obb()
        self._gen_boundary()
        self._gen_estimated_grid_size()
        self._gen_grid()
        
    def _gen_boundary(self):
        self.boundary = rg.Curve.CreateBooleanDifference(self.floor, self.core)[0]
        
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
        hall_shortest_segment = self.get_shortest_segment(self.hall)
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
                
                cleanup_rectangle = rg.Curve.CreateBooleanIntersection(
                    self.boundary, copied_rectangle.ToNurbsCurve()
                )
                if len(cleanup_rectangle) == 1:
                    self.grid.append(cleanup_rectangle[0])
        
    def _gen_estimated_k(self):
        """The K is given floor area divided target area."""
        self.k = 0
        
    def _gen_shortest_path(self):
        return


if __name__ == "__main__":
    krooms = KRoomsCluster(
        floor=floor,
        core=core,
        hall=hall,
        target_area=None,
    )
    
    krooms.predict()

    a = krooms.grid
    b = krooms.boundary