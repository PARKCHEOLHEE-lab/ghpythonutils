import glob
import math
import os
import random
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils

from APIs.mesh2points.mesh2pointsutils import point_cloud_show

random.seed(777)


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


class TNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class PointNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(__file__, "../data/ModelNet10"))

    sample_index = 0
    data_sofa_sample = glob.glob(
        os.path.abspath(os.path.join(data_dir, "sofa/train/*"))
    )[sample_index]
    sofa_sample_vertices, sofa_sample_faces = load_point_cloud(data_sofa_sample)

    sofa_sample_point_cloud = PointSampler(1024)(
        (sofa_sample_vertices, sofa_sample_faces)
    )
    point_cloud_show(
        sofa_sample_vertices,
        sofa_sample_point_cloud,
        labels=["sofa_sample_vertices", "sofa_sample_point_cloud"],
    )

    train_dataset = PointCloudData(data_dir=data_dir, transform=None)
    valid_dataset = PointCloudData(
        data_dir=data_dir, folder="test", transform=None
    )

    classes = {
        label: category for category, label in train_dataset.labels.items()
    }

    print("Train dataset size:", len(train_dataset))
    print("Valid dataset size:", len(valid_dataset))
    print("Number of classes:", len(classes))

    sample_point_cloud, sample_point_cloud_class = train_dataset[0]
    print("Sample pointcloud shape:", sample_point_cloud.shape)
    print("Sample pointcloud class:", classes[sample_point_cloud_class])
