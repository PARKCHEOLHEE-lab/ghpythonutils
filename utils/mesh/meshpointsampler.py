import math
import random

import Rhino.Geometry as rg


class MeshPointSamplerHelper:
    def _compute_triangle_area(self, triangle):
        """Compute the area of triangle

        Args:
            triangle (List[Point3d]): Vertices of a triangle 

        Returns:
            float: The area
        """

        side_a = math.sqrt(
            (triangle[1][0] - triangle[2][0]) ** 2
            + (triangle[1][1] - triangle[2][1]) ** 2
            + (triangle[1][2] - triangle[2][2]) ** 2
        )
        side_b = math.sqrt(
            (triangle[0][0] - triangle[2][0]) ** 2
            + (triangle[0][1] - triangle[2][1]) ** 2
            + (triangle[0][2] - triangle[2][2]) ** 2
        )
        side_c = math.sqrt(
            (triangle[0][0] - triangle[1][0]) ** 2
            + (triangle[0][1] - triangle[1][1]) ** 2
            + (triangle[0][2] - triangle[1][2]) ** 2
        )

        s = (side_a + side_b + side_c) / 2

        area = math.sqrt(s * (s - side_a) * (s - side_b) * (s - side_c))

        return area

    def _compute_sample_point(self, triangle):
        """Compute a point on the triangle

        Args:
            triangle (List[Point3d]): Vertices of a triangle

        Returns:
            Point3d: A sampled point
        """

        u = random.uniform(0, 1)
        v = random.uniform(0, 1 - u)
        w = 1 - u - v

        random_point = [
            u * triangle[0][0] + v * triangle[1][0] + w * triangle[2][0],
            u * triangle[0][1] + v * triangle[1][1] + w * triangle[2][1],
            u * triangle[0][2] + v * triangle[1][2] + w * triangle[2][2],
        ]

        return rg.Point3d(*random_point)

    @staticmethod
    def get_mesh_vertices(mesh):
        """Get the all vertices of a mesh

        Args:
            mesh (Mesh): Mesh to get the all vertices

        Returns:
            List[List[Point3d]]: All vertices of a mesh
        """

        mesh_vertices = []

        for face_index in range(mesh.Faces.Count):
            face = mesh.Faces[face_index]
            vertices = []

            for i in range(3):
                vertex_index = face[i]
                vertex = mesh.Vertices[vertex_index]
                vertices.append(rg.Point3d(vertex))

            mesh_vertices.append(vertices)

        return mesh_vertices


class MeshPointSampler(MeshPointSamplerHelper):
    def __init__(self, mesh, output_size, seed=0):
        self.mesh = mesh
        self.output_size = output_size
        self.seed = seed

        random.seed(self.seed)

    def sampling(self):
        """Compute random sampled points as much as `output_size`

        Returns:
            List[Point3d]: Weighted sampled random points by `area`
        """

        mesh_vertices = self.get_mesh_vertices(self.mesh)
        assert all(
            len(vertices) == 3 for vertices in mesh_vertices
        ), "Given triangle's vertices are invalid."

        triangle_areas = []

        for i in range(len(mesh_vertices)):
            triangle_areas.append(self._compute_triangle_area(mesh_vertices[i]))

        cum_weights = [
            sum(triangle_areas[: i + 1]) for i in range(len(triangle_areas))
        ]

        sampled_faces = []
        for _ in range(self.output_size):
            rand_num = random.uniform(0, cum_weights[-1])

            index = 0
            while rand_num > cum_weights[index]:
                index += 1

            sampled_faces.append(mesh_vertices[index])

        sampled_points = [[0, 0, 0] for _ in range(self.output_size)]

        for i in range(len(sampled_faces)):
            sampled_points[i] = self._compute_sample_point(sampled_faces[i])

        return sampled_points


if __name__ == "__main__":
    sampler = MeshPointSampler(mesh, 1024)
    sampled_points = sampler.sampling()
