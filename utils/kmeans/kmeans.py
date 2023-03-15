import Rhino.Geometry as rg

from utils.utils import PointHelper


class KMeans(PointHelper):
    def __init__(self, points=None, k=3, iteration_count=20, random_seed=0):
        """KMeansCluster simple implementation using Rhino Geometry

        Args:
            points (Rhino.Geometry.Point3d, optional): Points to classify. Defaults to None. if points is None, make random points 
            k (int, optional): Number to classify. Defaults to 3.
            iteration_count (int, optional): Clusters candidates creation count. Defaults to 20.
            random_seed (int, optional): Random seed number to fix. Defaults to 0.
        """
        
        PointHelper.__init__(self)
        
        self.points = points
        self.k = k
        self.iteration_count = iteration_count
        self.threshold = 0.1
        
        import random
        self.random = random
        self.random.seed(random_seed)
        
    def predict(self, get_indices=False):
        """Classify given points by input k

        Args:
            get_indices (bool, optional): If True, return together with indices. Defaults to False.

        Raises:
            Exception: When given points count is 0
            Exception: When dimension of given points is difference

        Returns:
            List[List[Rhino.Geometry.Point3d]], List[List[int]]: get_indices: True
            List[List[Rhino.Geometry.Point3d]]: get_indices: False
        """
        
        if self.points is None:
            num_points = 50
            dimensions = 3
            minimum = 0
            maximum = 100
            
            self.points = [
                self.get_random_point(dimensions, minimum, maximum) 
                for i in range(num_points)
            ]
            
        if len(self.points) <= 0:
            raise Exception("There is no Given points")
            
        if not all(len(point) == 3 for point in self.points):
            raise Exception("Given points have unequal dimensions")
        
        best_clusters, best_indices = self.iterative_kmeans(
            self.points,
            self.k,
            self.threshold,
            self.iteration_count
        )
        
        if get_indices:
            return best_clusters, best_indices
            
        return best_clusters
        
    def iterative_kmeans(self, points, k, threshold, iteration_count):
        """Cluster random sampling as much as iteration_count

        Args:
            points (Rhino.Geometry.Point3d): Initialized given points
            k (int): Initialized given k
            threshold (float): Initialized threshold
            iteration_count (int):  Initialized given iteration_count

        Returns:
            Tuple[List[List[Rhino.Geometry.Point3d]], List[List[int]]]: Bestest score clusters, Bestest score indices
        """

        clusters_candidates = []
        indices_candidates = []
        clusters_distortion_costs = []
        for _ in range(iteration_count):
            clusters, indices = self.kmeans(points, k, threshold)
            clusters_distortion_cost = self.get_distortion_cost(clusters)
            
            clusters_candidates.append(clusters)
            indices_candidates.append(indices)
            clusters_distortion_costs.append(clusters_distortion_cost)
        
        best_clusters_index = clusters_distortion_costs.index(min(clusters_distortion_costs))
        
        best_clusters = clusters_candidates[best_clusters_index]
        best_indices = indices_candidates[best_clusters_index]
        
        return best_clusters, best_indices
        
    def kmeans(self, points, k, threshold):
        """Clusters by each iteration

        Args:
            points (Rhino.Geometry.Point3d): Initialized given points
            k (int): Initialized given k
            threshold (float): Initialized threshold

        Returns:
            Tuple[List[List[Rhino.Geometry.Point3d]], List[List[int]]]: Clusters by each iteration, Indices by each iteration
        """

        centroids = self.random.sample(points, k)
        
        while True:
            clusters = [[] for _ in centroids]
            indices = [[] for _ in centroids]
            
            for pi, point in enumerate(points):
                point_to_centroid_distance = [point.DistanceTo(centroid) for centroid in centroids]
                nearest_centroid_index = point_to_centroid_distance.index(min(point_to_centroid_distance))
                
                clusters[nearest_centroid_index].append(point)
                indices[nearest_centroid_index].append(pi)
            
            shift_distance = 0.0
            for ci, current_centroid in enumerate(centroids):
                if len(clusters[ci]) == 0:
                    continue
                
                updated_centroid = self.get_centroid(clusters[ci])
                shift_distance = max(updated_centroid.DistanceTo(current_centroid), shift_distance)
                
                centroids[ci] = updated_centroid
            
            if shift_distance < threshold:
                break
            
        return clusters, indices
        
    def get_distortion_cost(self, clusters):
        """The degree of distortion of the each clusters.

        Args:
            clusters (List[List[Rhino.Geometry.Point3d]]): Each clusters by each iteration

        Returns:
            float: distortion cost
        """
        
        total_distance = 0
        for cluster in clusters:
            centroid = self.get_centroid(cluster)
            centroid_to_all_points = sum(point.DistanceTo(centroid) for point in cluster)
            
            total_distance += centroid_to_all_points
        
        distortion_cost = total_distance / len(self.points)
        
        return distortion_cost
        
    def get_centroid(self, points):
        """Inherited from utils.utils.PointHelper"""
        return self.get_points_cloud_centroid(points)
        
    def get_random_point(self, dimensions, minimum, maximum):
        coords = [self.random.uniform(minimum, maximum) for _ in range(dimensions)]
        if dimensions == 2:
            coords += [0]
        
        return rg.Point3d(*coords)