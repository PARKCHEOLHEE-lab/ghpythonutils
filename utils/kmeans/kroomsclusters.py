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

    def __init__(self, cells):
        self.cells = cells
        self.room = rg.Curve.CreateBooleanUnion(self.cells)[0]

        ConstsCollection.__init__(self)

    def is_connected_to_corridor(self, corridor):
        intersections = rg.Intersect.Intersection.CurveCurve(
            self.room, corridor, self.TOLERANCE, self.TOLERANCE
        )
        return len(intersections) > 0


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
        #        self._gen_network()
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
        boundaries = rg.Curve.CreateBooleanDifference(self.floor, self.core)

        self.boundaries = []
        for boundary in boundaries:
            boundary_object = Boundary(boundary, self.target_area)
            self.boundaries.append(boundary_object)

    def _gen_estimated_grid_size(self):
        hall_shortest_segment = self.sorted_hall_segments[0]
        self.grid_size = hall_shortest_segment.GetLength()

    #        self.grid_size_x = self.sorted_hall_segments[-1].GetLength()

    def _gen_grid(self):
        self.base_rectangles = [
            self.get_2d_offset_polygon(seg, self.grid_size)
            for seg in self.sorted_hall_segments[:2]
        ]

        counts = []
        planes = []

        copied_hall = copy.copy(self.hall)

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

            if ri == 0:
                copied_hall.Translate(
                    (
                        self.sorted_hall_segments[0].PointAtStart
                        - self.sorted_hall_segments[0].PointAtEnd
                    )
                    / 2
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
        for rectangle in x_grid + [copied_hall]:
            for y_count, y_vector in zip(y_counts, y_vectors):
                for yc in range(1, y_count):
                    copied_rectangle = copy.copy(rectangle)
                    vector = y_vector * self.grid_size * yc
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
            cells = []
            for index in each_indices:
                cells.append(self.grid[index])

            self.rooms.append(Room(cells))


#    def _gen_network(self):
#        self.network = []
#        self.network_length = []
#
#        for room in self.rooms:
#            for segment in room.room.DuplicateSegments():
#                self.network.append(segment)
#                self.network_length.append(segment.GetLength())
#
#        start_point_candidates = []
#        for grid in self.grid:
#            intersections = rg.Intersect.Intersection.CurveCurve(
#                grid, self.hall, self.TOLERANCE, self.TOLERANCE
#            )
#
#            for intsc in intersections:
#                start_point_candidates.append(intsc.PointA)
#                start_point_candidates.append(intsc.PointA2)
#
#        self.start_point_candidates = list(
#            rg.Point3d.CullDuplicates(start_point_candidates, self.TOLERANCE)
#        )
#
#        hall_vertices = [s.PointAtStart for s in self.hall.DuplicateSegments()]
#        corner_circles = []
#        for hall_vertex in hall_vertices:
#            corner_circle = rg.Circle(
#                hall_vertex, self.sorted_hall_segments[0].GetLength() / 3
#            ).ToNurbsCurve()
#
#            corner_circles.append(corner_circle)
#
#    def _gen_connected_rooms_to_corridor(self):
#        self.inaccessible_rooms = []
#        for ri, room in enumerate(self.rooms):
#            is_connected_to_corridor = room.is_connected_to_corridor(self.hall)
#            if not is_connected_to_corridor:
#                self.inaccessible_rooms.append(room)
#
#        hall_centroid = rg.AreaMassProperties.Compute(self.hall).Centroid
#        sorted_inaccessible_rooms = sorted(
#            self.inaccessible_rooms,
#            key=lambda r: rg.AreaMassProperties.Compute(r.room).Centroid.DistanceTo(
#                hall_centroid
#            ),
#            reverse=True,
#        )
#
#        self.intscs = []
#
#        self.corridor = list(rg.Curve.CreateBooleanUnion([self.hall]))
#        for inaccessible_room in sorted_inaccessible_rooms:
#            if inaccessible_room.is_connected_to_corridor(self.corridor[0]):
#                continue
#
#            ic = rg.AreaMassProperties.Compute(inaccessible_room.room).Centroid
#            start_point = sorted(
#                self.start_point_candidates, key=lambda p: p.DistanceTo(ic)
#            )[0]
#
#            _, start, end = inaccessible_room.room.ClosestPoints(self.hall)
#
#            corridor_curve, _, _, _ = ShortestWalk.ShortestWalk(
#                self.network,
#                self.network_length,
#                rg.Line(start, end).ToNurbsCurve(),
#            )
#
#            extended_corridor = corridor_curve.Extend(
#                rg.CurveEnd.Both, 1.4, rg.CurveExtensionStyle.Line
#            )
#
#            corridor = self.get_2d_buffered_linestring(
#                extended_corridor, -1.4, True
#            )[0]
#
#            intscs = rg.Curve.CreateBooleanIntersection(self.hall, corridor)
#            self.intscs.append(extended_corridor)
#
#
#            if len(intscs) != 0:
#                corridor = self.get_2d_buffered_linestring(
#                    extended_corridor, 1.4, True
#                )[0]
#
#            self.corridor = list(
#                rg.Curve.CreateBooleanUnion(self.corridor + [corridor])
#            )
#
#            print(self.corridor)
#
#            self.start_point_candidates.extend(
#                [extended_corridor.PointAtStart, extended_corridor.PointAtEnd]
#            )
#
#
#            break
#
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
