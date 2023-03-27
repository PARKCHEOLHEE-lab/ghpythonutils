# pylint: disable-all

import copy
import math

import Rhino.Geometry as rg
from ghpythonlib.components import ShortestWalk

from ghpythonutils.utils.dijkstra.dijkstra import Dijkstra
from ghpythonutils.utils.kmeans.kmeans import KMeans
from ghpythonutils.utils.utils import ConstsCollection, LineHelper, PointHelper


class Room(ConstsCollection):
    """Dataclass to each room"""

    def __init__(self, cells, floor_boundary):
        self.cells = cells
        self.floor_boundary = floor_boundary
        self.room = rg.Curve.CreateBooleanUnion(self.cells)[0]
        self._gen_boundary_points()

        ConstsCollection.__init__(self)

    def is_connected_to_corridor(self, corridor):
        intersections = rg.Intersect.Intersection.CurveCurve(
            self.room, corridor, self.TOLERANCE, self.TOLERANCE
        )
        return len(intersections) > 0

    def _gen_boundary_points(self):
        self.boundary_points = []
        for cell in self.cells:
            cell_vertices = LineHelper.get_curve_vertices(cell)
            for cell_vertex in cell_vertices:
                is_overlap = (
                    rg.Curve.Contains(self.room, cell_vertex, rg.Plane.WorldXY)
                    == rg.PointContainment.Coincident
                )

                for boundary in self.floor_boundary:
                    is_floor_overlap = (
                        rg.Curve.Contains(
                            boundary, cell_vertex, rg.Plane.WorldXY
                        )
                        == rg.PointContainment.Coincident
                    )

                    if is_overlap and not is_floor_overlap:
                        self.boundary_points.append(cell_vertex)


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
        assert self.target_area > 0, "Target area is less than zero"
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
        self._is_disjoint_floor_and_core()

        self._gen_given_axis_aligned_obb()
        self._gen_boundaries()
        self._gen_estimated_grid_size()
        self._gen_grid()
        self._gen_predicted_rooms()
        self._gen_corridor()
        #        self._gen_connected_rooms_to_corridor()

        return [room.room for room in self.rooms]

    def _is_disjoint_floor_and_core(self):
        is_disjoint = (
            rg.Curve.PlanarClosedCurveRelationship(
                self.floor, self.core, rg.Plane.WorldXY, self.TOLERANCE
            )
            == rg.RegionContainment.Disjoint
        )

        assert not is_disjoint, "Given floor and core are disjoint"

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

    def _gen_boundaries(self):
        self.diff_floor = rg.Curve.CreateBooleanDifference(
            self.floor, self.core
        )

        self.boundaries = []
        for boundary in self.diff_floor:
            boundary_object = Boundary(boundary, self.target_area)
            self.boundaries.append(boundary_object)

    def _gen_estimated_grid_size(self):
        hall_shortest_segment = self.sorted_hall_segments[0]
        self.grid_size = hall_shortest_segment.GetLength()

    def _gen_grid(self):
        self.base_rectangles = [
            self.get_2d_offset_polygon(seg, self.grid_size)
            for seg in self.sorted_hall_segments[:2]
        ]

        counts = []
        planes = []

        for ri, base_rectangle in enumerate(self.base_rectangles):
            x_vector = (
                rg.AreaMassProperties.Compute(base_rectangle).Centroid
                - rg.AreaMassProperties.Compute(self.hall).Centroid
            )

            y_vector = copy.copy(x_vector)
            y_transform = rg.Transform.Rotation(
                math.pi * 0.5,
                rg.AreaMassProperties.Compute(base_rectangle).Centroid,
            )
            y_vector.Transform(y_transform)

            base_rectangle.Translate(
                (
                    self.sorted_hall_segments[0].PointAtStart
                    - self.sorted_hall_segments[0].PointAtEnd
                )
                / 2
            )

            base_rectangle.Translate(
                self.get_normalized_vector(x_vector) * -self.grid_size / 2
            )

            anchor = rg.AreaMassProperties.Compute(base_rectangle).Centroid
            plane = rg.Plane(
                origin=anchor,
                xDirection=x_vector,
                yDirection=y_vector,
            )

            x_proj = self.get_projected_point_on_curve(
                anchor, plane.XAxis, self.obb
            )

            x_count = (
                int(math.ceil(x_proj.DistanceTo(anchor) / self.grid_size)) + 1
            )

            y_projs = [
                self.get_projected_point_on_curve(
                    anchor, plane.YAxis, self.obb
                ),
                self.get_projected_point_on_curve(
                    anchor, -plane.YAxis, self.obb
                ),
            ]

            y_count = [
                int(math.ceil(y_proj.DistanceTo(anchor) / self.grid_size)) + 1
                for y_proj in y_projs
            ]

            planes.append(plane)
            counts.append([x_count] + y_count)

        x_grid = []
        for base_rectangle, count, plane in zip(
            self.base_rectangles, counts, planes
        ):
            xc, _, _ = count

            for x in range(xc):
                copied_rectangle = copy.copy(base_rectangle)
                vector = plane.XAxis * self.grid_size * x
                copied_rectangle.Translate(vector)
                x_grid.append(copied_rectangle)

        y_vectors = [planes[0].YAxis, -planes[0].YAxis]
        y_counts = counts[0][1:]
        all_grid = [] + x_grid
        for rectangle in x_grid:
            for y_count, y_vector in zip(y_counts, y_vectors):
                for yc in range(1, y_count):
                    copied_rectangle = copy.copy(rectangle)
                    vector = y_vector * self.grid_size * yc
                    copied_rectangle.Translate(vector)
                    all_grid.append(copied_rectangle)

        union_all_grid = rg.Curve.CreateBooleanUnion(all_grid, self.TOLERANCE)

        for y_count, y_vector in zip(y_counts, y_vectors):
            for yc in range(1, y_count):
                copied_hall = copy.copy(self.hall)
                copied_hall.Translate(
                    (
                        self.sorted_hall_segments[0].PointAtStart
                        - self.sorted_hall_segments[0].PointAtEnd
                    )
                    / 2
                )

                vector = y_vector * self.grid_size * yc
                copied_hall.Translate(vector)
                all_grid.extend(
                    rg.Curve.CreateBooleanDifference(
                        copied_hall, union_all_grid
                    )
                )

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
            cells = []
            for index in each_indices:
                cells.append(self.grid[index])

            self.rooms.append(Room(cells, self.diff_floor))

    def _gen_corridor(self):
        self.start_point_candidates = [
            s.PointAtLength(s.GetLength() / 2)
            for s in self.sorted_hall_segments[:2]
        ]

        self.inaccessible_rooms = []
        for ri, room in enumerate(self.rooms):
            is_connected_to_corridor = room.is_connected_to_corridor(self.hall)
            if not is_connected_to_corridor:
                self.inaccessible_rooms.append(room)

        hall_centroid = rg.AreaMassProperties.Compute(self.hall).Centroid
        sorted_inaccessible_rooms = sorted(
            self.inaccessible_rooms,
            key=lambda r: rg.AreaMassProperties.Compute(
                r.room
            ).Centroid.DistanceTo(hall_centroid),
            reverse=True,
        )

        self.network = []
        for grid_segment in self.get_removed_overlapped_curves(self.grid):
            intersections = rg.Intersect.Intersection.CurveCurve(
                grid_segment, self.core, self.TOLERANCE, self.TOLERANCE
            )

            for intersect in intersections:
                if not intersect.IsOverlap:
                    self.network.append(grid_segment)

            if len(intersections) < 1:
                self.network.append(grid_segment)

        self.corridor = [self.hall]
        self.asdf = []
        for inaccessible_room in sorted_inaccessible_rooms:
            if inaccessible_room.is_connected_to_corridor(self.corridor[0]):
                continue

            ic = rg.AreaMassProperties.Compute(inaccessible_room.room).Centroid
            start_point = sorted(
                self.start_point_candidates, key=lambda p: p.DistanceTo(ic)
            )[0]

            room_points = rg.PointCloud(inaccessible_room.boundary_points)
            closest_index = room_points.ClosestPoint(start_point)
            target_point = inaccessible_room.boundary_points[closest_index]

            print(target_point)

            dijkstra = Dijkstra(self.network, start_point, target_point)
            corridor_line = rg.Curve.JoinCurves(dijkstra.shortest_path)[0]

            each_corridor = self.get_2d_buffered_linestring(
                corridor_line, self.grid_size
            )[0]

            self.corridor.append(each_corridor)

        self.corridor = rg.Curve.CreateBooleanUnion(self.corridor)


#
#        self.krooms = []
#        for room in self.rooms:
#            self.krooms.extend(
#                list(
#                    rg.Curve.CreateBooleanDifference(
#                        room.room, self.corridor[0]
#                    )
#                )
#            )

if __name__ == "__main__":
    krsc = KRoomsCluster(
        floor=floor,
        core=core,
        hall=hall,
        target_area=50,
    )

    krooms = krsc.get_predicted_rooms()
    grid = krsc.grid

    import ghpythonlib.treehelpers as gt

    d = krsc.corridor
    e = gt.list_to_tree([kroom.boundary_points for kroom in krsc.rooms])
