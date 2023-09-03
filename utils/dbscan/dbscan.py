class DBScanEnums:
    UNCLASSIFIED = 0
    NOISE = -1


class DBScanHelper(DBScanEnums):
    def _get_random_points(self):
        import random

        import Rhino.Geometry as rg

        random.seed(777)
        num_points = 500
        random_points = []

        for _ in range(num_points):
            x = random.uniform(0, 10)
            y = random.uniform(0, 10)
            z = random.uniform(0, 2)
            point = rg.Point3d(x, y, z)
            random_points.append(point)

        return random_points

    def _get_neighbors_indices(self, points, point_idx, epsilon):
        neighbors_indices = []
        for i, point in enumerate(points):
            if point.DistanceTo(points[point_idx]) <= epsilon:
                neighbors_indices.append(i)

        return neighbors_indices

    def _expand_cluster(
        self, points, labels, point_idx, cluster_id, epsilon, min_samples
    ):
        neighbors_indices = self._get_neighbors_indices(
            points, point_idx, epsilon
        )

        is_noise = len(neighbors_indices) < min_samples

        if is_noise:
            labels[point_idx] = self.NOISE

        else:
            labels[point_idx] = cluster_id
            for neighbor_idx in neighbors_indices:
                if labels[neighbor_idx] == self.UNCLASSIFIED:
                    labels[neighbor_idx] = cluster_id

                    self._expand_cluster(
                        points,
                        labels,
                        neighbor_idx,
                        cluster_id,
                        epsilon,
                        min_samples,
                    )

        return is_noise


class DBScan(DBScanHelper, DBScanEnums):
    def __init__(self, points=None, epsilon=1, min_samples=10):
        self.points = points
        self.epsilon = epsilon
        self.min_samples = min_samples

        if self.points is None:
            self.points = self._get_random_points()

    def clustering(self):
        num_points = len(self.points)
        self.labels = [self.UNCLASSIFIED] * num_points

        cluster_id = 0
        for i in range(num_points):
            if self.labels[i] == self.UNCLASSIFIED:
                is_noise = self._expand_cluster(
                    self.points,
                    self.labels,
                    i,
                    cluster_id + 1,
                    self.epsilon,
                    self.min_samples,
                )

                if not is_noise:
                    cluster_id += 1


if __name__ == "__main__":
    dbscan = DBScan()
    dbscan.clustering()
    points = dbscan.points
    labels = dbscan.labels
