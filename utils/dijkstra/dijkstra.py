import heapq

import Rhino.Geometry as rg

from ghpythonutils.utils.utils import (
    ConstsCollection,
    LineHelper,
    NumericHelper,
    PointHelper,
)


class Node:
    """Each vertex of the graph"""

    def __init__(self, node_index):
        self.node_index = node_index

        self.adjacent = {}
        self.distance = ConstsCollection.INF
        self.is_visited = False
        self.previous_vertex = None

    def __repr__(self):
        return (
            str(self.node_index)
            + "th vertex"
            + ": adjacent ->"
            + str([v.node_index for v in self.adjacent])
        )

    def get_adjacent_cost(self, vertex):
        """Get cost(edge length) from self to vertex

        Args:
            vertex (Node): Adjacent node

        Returns:
            float: Cost
        """

        return self.adjacent[vertex]

    def set_adjacent(self, vertex, cost):
        """Append the adjacent node to self node

        Args:
            vertex (Node): Adjacent node
            cost (float): Cost
        """

        self.adjacent[vertex] = cost


class Graph(PointHelper, LineHelper, NumericHelper):
    """Graph to node and edge composed"""

    def __init__(self, input_curves):
        self.input_curves = input_curves

        PointHelper.__init__(self)
        LineHelper.__init__(self)
        NumericHelper.__init__(self)

        self._generate()

    def __iter__(self):
        return iter(self.graph_dict.values())

    def _generate(self):
        self._gen_network()
        self._gen_graph()

    def _gen_network(self):
        """Generate network data with duplicates lines and vertices removed
        The structure of network data is like to -> self.network: List[List[int], Rhino.Geometry.Curve]
        """

        cleanup_curves = [
            curve for curve in self.input_curves if curve is not None
        ]

        self.curves, self.vertices = self.get_removed_overlapped_curves(
            cleanup_curves, is_needed_for_points=True
        )

        self.network = []
        for curve in self.curves:
            curve_start_point = curve.PointAtStart
            curve_end_point = curve.PointAtEnd

            data = []
            for node_index, vertex in enumerate(self.vertices):
                if self.is_same_points(curve_start_point, vertex):
                    data.append(node_index)

                elif self.is_same_points(curve_end_point, vertex):
                    data.append(node_index)

                if len(data) == 2:
                    break

            self.network.append([data, curve])

    def _gen_graph(self):
        """Generate the graph through the network data to node and edge composed"""

        self.graph_dict = {}
        for node_index in range(len(self.vertices)):
            self.graph_dict[node_index] = Node(node_index)

        for data in self.network:
            (start_node_index, end_node_index), edge_geom = data
            cost = edge_geom.GetLength()
            self.graph_dict[start_node_index].set_adjacent(
                self.graph_dict[end_node_index], cost
            )
            self.graph_dict[end_node_index].set_adjacent(
                self.graph_dict[start_node_index], cost
            )

    def get_node(self, node_index):
        """Get node object from `graph_dict`

        Args:
            node_index (int): index

        Returns:
            Node: Node object about given index
        """

        return self.graph_dict[node_index]


class Dijkstra(Graph):
    """Class to get the shortest path from given curves"""

    def __init__(self, input_curves, start_point, target_point):
        self.input_curves = input_curves
        self.start_point = start_point
        self.target_point = target_point

        Graph.__init__(self, input_curves)
        self._run()

    def _run(self):
        self._gen_estimated_path()
        self._relax()
        self._gen_shortest_path()

    def _gen_estimated_path(self):
        """Estimate indices about the given start point and target point"""

        for node_index, vertex in enumerate(self.vertices):
            is_start_vertex = isinstance(
                self.start_point, rg.Point3d
            ) and self.is_same_points(self.start_point, vertex)
            is_target_vertex = isinstance(
                self.target_point, rg.Point3d
            ) and self.is_same_points(self.target_point, vertex)

            if is_start_vertex:
                self.start_index = node_index
            elif is_target_vertex:
                self.target_index = node_index

    def _relax(self):
        """Calculate the distance between all nodes from the starting point"""

        start_vertex = self.get_node(self.start_index)
        start_vertex.distance = 0

        unvisited_queue = [(v.distance, v) for v in self]
        heapq.heapify(unvisited_queue)

        while len(unvisited_queue) > 0:
            current_distance, current_vertex = heapq.heappop(unvisited_queue)
            current_vertex.is_visited = True

            for adjacent_vertex in current_vertex.adjacent:
                if adjacent_vertex.is_visited:
                    continue

                distance_sum = (
                    current_distance
                    + current_vertex.get_adjacent_cost(adjacent_vertex)
                )

                if distance_sum < adjacent_vertex.distance:
                    adjacent_vertex.distance = distance_sum
                    adjacent_vertex.previous_vertex = current_vertex

            unvisited_queue = [
                (v.distance, v) for v in self if not v.is_visited
            ]
            heapq.heapify(unvisited_queue)

    def _gen_shortest_path(self):
        """Generate the shortest path via relaxed graph data"""

        target_vertex = self.get_node(self.target_index)
        shortest_path_indices = [target_vertex.node_index]
        shortest_path_costs = []

        while target_vertex.previous_vertex is not None:
            shortest_path_indices.append(
                target_vertex.previous_vertex.node_index
            )
            shortest_path_costs.append(
                target_vertex.get_adjacent_cost(target_vertex.previous_vertex)
            )
            target_vertex = target_vertex.previous_vertex

        edge_indices = []
        for i in range(1, len(shortest_path_indices)):
            edge_indices.append(
                [shortest_path_indices[i - 1], shortest_path_indices[i]]
            )

        self.shortest_path = []
        if self.start_index in shortest_path_indices:
            for path, cost in zip(edge_indices, shortest_path_costs):
                for data in self.network:
                    (start_node_index, end_node_index), edge_geom = data

                    is_same_path = path == [start_node_index, end_node_index]
                    is_same_path_reversed = (
                        path == [start_node_index, end_node_index][::-1]
                    )
                    is_same_cost = NumericHelper.is_close(
                        cost, edge_geom.GetLength()
                    )

                    if (is_same_path or is_same_path_reversed) and is_same_cost:
                        self.shortest_path.append(edge_geom)
                        break
