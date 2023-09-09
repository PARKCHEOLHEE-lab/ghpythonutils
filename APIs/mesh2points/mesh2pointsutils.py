import os
import glob
import math
import random

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms, utils

random.seed(777)

def point_cloud_show(*args: np.ndarray, labels: List[str] = None) -> None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    if labels is not None:
        assert len(args) == len(
            labels
        ), "The length between the given `args` and `labels` is different."

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Define a colormap to assign unique colors to each iteration
    cmap = plt.get_cmap("plasma")
    num_args = len(args)
    colors = [cmap(i / num_args) for i in range(num_args)]

    for i, point_cloud in enumerate(args):
        x, y, z = point_cloud.T

        label = f"{labels[i]}" if labels is not None else f"point cloud {i + 1}"

        ax.scatter(x, y, z, c=colors[i], marker="o", s=10, label=label)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    # Add a legend to differentiate iterations
    ax.legend()

    plt.show()


def load_point_cloud(file_path: str) -> np.array:
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()

        num_vertices, _, _ = map(int, lines[1].split())

        # Parse vertices
        vertices = []
        for line in lines[2 : 2 + num_vertices]:
            x, y, z = map(float, line.strip().split())
            vertices.append([x, y, z])

        # Parse faces
        faces = []
        for line in lines[2 + num_vertices :]:
            parts = line.strip().split()
            if len(parts) > 3:  # Ignore faces with less than 3 vertices
                num_vertices_in_face = int(parts[0])
                face_indices = list(map(int, parts[1:]))
                if len(face_indices) == num_vertices_in_face:
                    faces.append(face_indices)

        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)

        return vertices, faces

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


class PointSampler:
    def __init__(self, output_size: int):
        self.output_size = output_size

    def _compute_triangle_area(
        self, point_1: np.ndarray, point_2: np.ndarray, point_3: np.ndarray
    ) -> float:
        side_a = math.sqrt(
            (point_2[0] - point_3[0]) ** 2
            + (point_2[1] - point_3[1]) ** 2
            + (point_2[2] - point_3[2]) ** 2
        )
        side_b = math.sqrt(
            (point_1[0] - point_3[0]) ** 2
            + (point_1[1] - point_3[1]) ** 2
            + (point_1[2] - point_3[2]) ** 2
        )
        side_c = math.sqrt(
            (point_1[0] - point_2[0]) ** 2
            + (point_1[1] - point_2[1]) ** 2
            + (point_1[2] - point_2[2]) ** 2
        )

        s = (side_a + side_b + side_c) / 2

        formula = s * (s - side_a) * (s - side_b) * (s - side_c)
        area = 0
        if formula > 0:
            area = math.sqrt(formula)

        return area

    def _compute_sample_point(
        self, point_1: np.ndarray, point_2: np.ndarray, point_3: np.ndarray
    ):
        u = random.uniform(0, 1)
        v = random.uniform(0, 1 - u)
        w = 1 - u - v

        random_point = [
            u * point_1[0] + v * point_2[0] + w * point_3[0],
            u * point_1[1] + v * point_2[1] + w * point_3[1],
            u * point_1[2] + v * point_2[2] + w * point_3[2],
        ]

        return random_point

    def __call__(self, mesh: Union[List[np.ndarray], Tuple[np.ndarray]]):
        vertices, faces = mesh
        vertices = np.array(vertices)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = self._compute_triangle_area(
                vertices[faces[i][0]],
                vertices[faces[i][1]],
                vertices[faces[i][2]],
            )

        sampled_faces = random.choices(
            faces, weights=areas, cum_weights=None, k=self.output_size
        )

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = self._compute_sample_point(
                vertices[sampled_faces[i][0]],
                vertices[sampled_faces[i][1]],
                vertices[sampled_faces[i][2]],
            )

        return sampled_points

    
class Normalize:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud


class RnadomRotationZ:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T

        return  rot_pointcloud

    
class RandomNoise:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))
    
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud


class PointCloudData(Dataset):
    def __init__(
        self, data_dir: str, folder: str = "train", transform: transforms = None
    ):
        self.data_dir = data_dir
        self.folder = folder
        self.transform = transform
        self.categories = [
            category for category in os.listdir(data_dir) if "." not in category
        ]
        self.labels = {
            category: i for i, category in enumerate(self.categories)
        }

    def __len__(self):
        length = 0

        for category in self.categories:
            category_dir = glob.glob(
                os.path.join(self.data_dir, category, f"{self.folder}/*")
            )
            category_dir = [
                name for name in category_dir if name.endswith(".off")
            ]

            length += len(category_dir)

        return length

    def __getitem__(self, idx):
        num_samples_per_category = len(self) // len(self.categories)
        category_idx = idx // num_samples_per_category
        sample_idx_within_category = idx % num_samples_per_category

        category = self.categories[category_idx]
        category_dir = glob.glob(
            os.path.join(self.data_dir, category, f"{self.folder}/*")
        )
        category_dir = [name for name in category_dir if name.endswith(".off")]

        filename = category_dir[sample_idx_within_category]

        point_cloud = load_point_cloud(filename)

        if self.transform:
            point_cloud = self.transform(point_cloud)

        label = self.labels[category]

        return point_cloud, label
