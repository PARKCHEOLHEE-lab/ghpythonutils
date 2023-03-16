from utils.kmeans.kmeans import KMeans
from utils.utils import PointHelper, LineHelper, ConstsCollection
import Rhino.Geometry as rg
import copy
import math
from ghpythonlib.components import ShortestWalk


class Room:
    """
    _description_
    """

    def __init__(self, boundary, cells, path):
        self.boundary = boundary
        self.cells = cells
        self.path = path

    def _gen_corridor(self):
        pass


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


class KRoomsCluster(KMeans, PointHelper, LineHelper, ConstsCollection):
    """
    To use the inherited moduels, refer the link below.
    https://github.com/PARKCHEOLHEE-lab/GhPythonUtils
    """

    def __init__(self, floor, core, hall, target_area, axis=None):
        self.floor = floor
        self.core = core
        self.hall = hall
        self.sorted_hall_segments = self.get_sorted_segment(self.hall)
        self.target_area = target_area
        self.axis = axis

        KMeans.__init__(self)
        PointHelper.__init__(self)
        LineHelper.__init__(self)
        ConstsCollection.__init__(self)

    def get_predicted_rooms(self):
        self._gen_given_axis_aligned_obb()
        self._gen_boundaries()
        self._gen_estimated_grid_size()
        self._gen_grid()
        self._gen_predicted_rooms()
        self._gen_network()
        self._gen_connected_rooms_to_corridor()

        return self.rooms

    def _gen_boundaries(self):
        boundaries = rg.Curve.CreateBooleanDifference(self.floor, self.core)

        self.boundaries = []
        for boundary in boundaries:
            boundary_object = Boundary(boundary, self.target_area)
            self.boundaries.append(boundary_object)

    def _gen_given_axis_aligned_obb(self):
        if self.axis is None:
            self.axis = self.core.DuplicateSegments()[0]

        self.obb = self.get_2d_obb_from_line(
            self.axis, self.floor
        ).ToNurbsCurve()

        if (
            self.obb.ClosedCurveOrientation()
            != rg.CurveOrientation.CounterClockwise
        ):
            self.obb.Reverse()

    def _gen_estimated_grid_size(self):
        hall_shortest_segment = self.sorted_hall_segments[0]
        self.grid_size = hall_shortest_segment.GetLength()
        self.grid_size_x = self.sorted_hall_segments[-1].GetLength()

    def _gen_grid(self):
        base_rectangle = self.get_2d_offset_polygon(
            self.sorted_hall_segments[0], self.grid_size_x / 4
        )

        base_rectangle.Translate(
            (
                self.sorted_hall_segments[0].PointAtStart
                - self.sorted_hall_segments[0].PointAtEnd
            )
            / 2
        )

        x_segment, y_segment, _, _ = base_rectangle.DuplicateSegments()
        anchor = base_rectangle.ToPolyline().CenterPoint()
        plane = rg.Plane(
            origin=anchor,
            xDirection=x_segment.PointAtEnd - x_segment.PointAtStart,
            yDirection=y_segment.PointAtEnd - x_segment.PointAtStart,
        )

        counts = []
        for vi, plane_element in enumerate(plane):
            if isinstance(plane_element, rg.Point3d):
                continue

            grid_size = x_segment.GetLength()
            if vi == 2:
                grid_size = y_segment.GetLength()

            projected_points = [
                self.get_projected_point_on_curve(
                    anchor, plane_element, self.obb
                ),
                self.get_projected_point_on_curve(
                    anchor, -plane_element, self.obb
                ),
            ]

            for projected_point in projected_points:
                if projected_point is None:
                    continue

                count = (
                    int(
                        math.ceil(
                            projected_point.DistanceTo(anchor) / grid_size
                        )
                    )
                    + 1
                )

                counts.append(count)

        x_vectors = [plane.XAxis, -plane.XAxis]
        y_vectors = [plane.YAxis, -plane.YAxis]

        x_grid = [base_rectangle]
        for x_count, x_vector in zip(counts[:2], x_vectors):
            for xc in range(1, x_count):
                copied_rectangle = copy.copy(base_rectangle)
                vector = x_vector * x_segment.GetLength() * xc
                copied_rectangle.Translate(vector)
                x_grid.append(copied_rectangle)

        all_grid = [] + x_grid
        for rectangle in x_grid:
            for y_count, y_vector in zip(counts[2:], y_vectors):
                for yc in range(y_count):
                    copied_rectangle = copy.copy(rectangle)
                    vector = y_vector * y_segment.GetLength() * yc
                    copied_rectangle.Translate(vector)
                    all_grid.append(copied_rectangle)

        self.grid = []
        for grid in all_grid:
            for boundary in self.boundaries:
                tidied_grid = list(
                    rg.Curve.CreateBooleanIntersection(
                        boundary.boundary, grid, self.TOLERANCE
                    )
                )

                self.grid.extend(tidied_grid)

        self.grid_centroids = [
            rg.AreaMassProperties.Compute(g).Centroid for g in self.grid
        ]

    def _gen_predicted_rooms(self):
        self.points = self.grid_centroids
        self.k = self.boundaries[0].k

        _, indices = self.predict(get_indices=True)

        self.rooms = []
        for each_indices in indices:
            room = []
            for index in each_indices:
                room.append(self.grid[index])

            self.rooms.extend(rg.Curve.CreateBooleanUnion(room))

    def _gen_network(self):
        self.network = []
        self.network_length = []
        for room in self.rooms:
            for segment in room.DuplicateSegments():
                self.network.append(segment)
                self.network_length.append(segment.GetLength())

        start_point_candidates = []
        for grid in self.grid:
            intersections = rg.Intersect.Intersection.CurveCurve(
                grid, self.hall, self.TOLERANCE, self.TOLERANCE
            )

            for intsc in intersections:
                rg.Intersect.IntersectionEvent.PointA2
                start_point_candidates.append(intsc.PointA)
                start_point_candidates.append(intsc.PointA2)

        start_point_candidates = list(
            rg.Point3d.CullDuplicates(start_point_candidates, self.TOLERANCE)
        )

        hall_vertices = [s.PointAtStart for s in self.hall.DuplicateSegments()]
        corner_circles = []
        for hall_vertex in hall_vertices:
            corner_circle = rg.Circle(
                hall_vertex, self.sorted_hall_segments[0].GetLength() / 3
            ).ToNurbsCurve()

            corner_circles.append(corner_circle)

        self.start_points = []
        for candidate in start_point_candidates:
            outside_count = 0
            for circle in corner_circles:
                outside_count += int(
                    circle.Contains(candidate) == rg.PointContainment.Outside
                )
            if outside_count == len(hall_vertices):
                self.start_points.append(candidate)

    def _gen_connected_rooms_to_corridor(self):
        return


if __name__ == "__main__":
    krooms = KRoomsCluster(
        floor=floor,
        core=core,
        hall=hall,
        target_area=70,
    )

    c = krooms.get_predicted_rooms()
    a = krooms.grid
    b = krooms.rooms
    d = krooms.start_points
